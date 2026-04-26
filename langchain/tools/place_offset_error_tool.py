"""
Place Offset Error Tool

Offsets a place operation's target frame sideways so a new object lands
beside an already-placed one rather than on top of it.

Pose lookup priority:
  1. LookupTransform on the original place_frame_id (reliable — TF always published)
  2. GetObjectPose on the colliding object (fallback)

The original place_frame_id is stored in the YAML under
'_original_place_frame_id' on first offset, so successive attempts
always offset from the same base position (not cascading).
"""

import sys
import time
import yaml
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from b_scene_interfaces.srv import GetObjectPose, PublishPoseAsTF, LookupTransform
from geometry_msgs.msg import Pose
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

PACKAGE_ROOT = Path(__file__).parent.parent.parent

# Offset directions (dx_m, dy_m) per attempt
OFFSET_SEQUENCE = [
    ( 0.12,  0.00),
    ( 0.00,  0.12),
    (-0.12,  0.00),
    ( 0.00, -0.12),
    ( 0.15,  0.15),
]
PREPLACE_LIFT = 0.15   # Z above place frame for safe approach


class SceneQueryNode(Node):
    def __init__(self):
        super().__init__('place_offset_node_' + str(int(time.time())))
        self._pose_client    = self.create_client(GetObjectPose,  '/get_object_pose')
        self._tf_pub_client  = self.create_client(PublishPoseAsTF, '/publish_pose_as_tf')
        self._tf_look_client = self.create_client(LookupTransform, '/lookup_transform')

    def _call_sync(self, client, request, timeout_sec=5.0):
        if not client.wait_for_service(timeout_sec=5.0):
            return None
        future = client.call_async(request)
        start = time.time()
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - start > timeout_sec:
                return None
        return future.result()

    def lookup_frame_pose(self, frame_id: str) -> Optional[Tuple[float, float, float, dict]]:
        """Return (x, y, z, orientation_dict) of frame_id in world frame, or None."""
        req = LookupTransform.Request()
        req.from_frame = 'world'
        req.to_frame = frame_id
        resp = self._call_sync(self._tf_look_client, req)
        if resp and resp.success:
            t = resp.transform.transform.translation
            r = resp.transform.transform.rotation
            return t.x, t.y, t.z, {'x': r.x, 'y': r.y, 'z': r.z, 'w': r.w}
        return None

    def get_object_pose(self, object_id: str) -> Optional[Tuple[float, float, float, dict]]:
        """Return (x, y, z, orientation_dict) of object, or None."""
        req = GetObjectPose.Request()
        req.id = object_id
        resp = self._call_sync(self._pose_client, req)
        if resp and resp.result.state == 1:
            p = resp.pose.pose.position
            o = resp.pose.pose.orientation
            return p.x, p.y, p.z, {'x': o.x, 'y': o.y, 'z': o.z, 'w': o.w}
        return None

    def publish_tf_frame(self, x, y, z, orientation: dict, parent_frame: str, child_frame: str) -> bool:
        req = PublishPoseAsTF.Request()
        req.pose.position.x = x
        req.pose.position.y = y
        req.pose.position.z = z
        req.pose.orientation.x = orientation['x']
        req.pose.orientation.y = orientation['y']
        req.pose.orientation.z = orientation['z']
        req.pose.orientation.w = orientation['w']
        req.frame_id = parent_frame
        req.child_frame_id = child_frame
        req.is_static = True
        resp = self._call_sync(self._tf_pub_client, req)
        return bool(resp and resp.success)


_scene_node: Optional[SceneQueryNode] = None


def _get_scene_node() -> SceneQueryNode:
    global _scene_node
    if _scene_node is None:
        if not rclpy.ok():
            rclpy.init()
        _scene_node = SceneQueryNode()
    return _scene_node


def _get_operation_key(config: dict, operation_index: int) -> Optional[str]:
    numbered = f"MoveAndPlace_{operation_index}"
    if numbered in config:
        return numbered
    if operation_index == 1 and "MoveAndPlace" in config:
        return "MoveAndPlace"
    return None


class PlaceOffsetErrorInput(BaseModel):
    yaml_path: str = Field(description="Path to task YAML file")
    colliding_object: str = Field(description="ID of the already-placed blocking object (e.g. 'part_1')")
    operation_index: int = Field(description="Index of the failing MoveAndPlace operation (1, 2, 3...)")
    attempt: int = Field(default=1, description="Retry attempt — selects offset direction")


def place_offset_error_func(
    yaml_path: str,
    colliding_object: str,
    operation_index: int,
    attempt: int = 1,
) -> Dict:
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        op_key = _get_operation_key(config, operation_index)
        if op_key is None:
            return {'success': False, 'message': f'No MoveAndPlace op found for index {operation_index}'}

        op = config[op_key]

        # Use stored original frame so successive attempts offset from the same base
        original_place_frame = op.get('_original_place_frame_id') or op.get('place_frame_id', '')
        if not original_place_frame:
            return {'success': False, 'message': 'Cannot determine original place_frame_id from YAML'}

        node = _get_scene_node()

        # Try TF lookup first (most reliable — frame is always published)
        pose_result = node.lookup_frame_pose(original_place_frame)
        source = f"TF lookup of '{original_place_frame}'"

        if pose_result is None:
            # Fallback: get pose of the colliding placed object
            pose_result = node.get_object_pose(colliding_object)
            source = f"GetObjectPose of '{colliding_object}'"

        if pose_result is None:
            return {
                'success': False,
                'message': (
                    f"Could not determine base position for offset. "
                    f"TF lookup of '{original_place_frame}' and GetObjectPose of '{colliding_object}' both failed."
                ),
            }

        bx, by, bz, orientation = pose_result

        # Select offset direction
        dx, dy = OFFSET_SEQUENCE[(attempt - 1) % len(OFFSET_SEQUENCE)]

        # New frame names (unique per operation + attempt)
        place_frame    = f"offset_place_op{operation_index}_a{attempt}"
        preplace_frame = f"offset_preplace_op{operation_index}_a{attempt}"

        # Publish place frame (same Z as original)
        ok_place = node.publish_tf_frame(
            bx + dx, by + dy, bz,
            orientation, 'world', place_frame,
        )
        # Publish preplace frame (lifted for safe approach)
        ok_preplace = node.publish_tf_frame(
            bx + dx, by + dy, bz + PREPLACE_LIFT,
            orientation, 'world', preplace_frame,
        )

        if not (ok_place and ok_preplace):
            return {'success': False, 'message': 'Failed to publish offset TF frames via /publish_pose_as_tf'}

        # Update YAML — store original frame for future attempts
        op['_original_place_frame_id'] = original_place_frame
        op['place_frame_id']  = place_frame
        op['preplace_frame']  = preplace_frame
        op['postplace_frame'] = preplace_frame

        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return {
            'success': True,
            'message': (
                f"Offset applied to {op_key} (attempt {attempt}, source: {source}): "
                f"dx={dx:.3f}m dy={dy:.3f}m from '{original_place_frame}'. "
                f"New frames: place='{place_frame}', preplace='{preplace_frame}'"
            ),
            'place_frame': place_frame,
            'preplace_frame': preplace_frame,
        }

    except FileNotFoundError:
        return {'success': False, 'message': f'YAML not found: {yaml_path}'}
    except Exception as e:
        return {'success': False, 'message': f'Exception: {e}'}


place_offset_error = StructuredTool.from_function(
    func=place_offset_error_func,
    name="place_offset_error",
    description="""Fix a place collision with an already-placed task object by offsetting the target frame.

Uses TF lookup of the original place_frame_id (reliable) with GetObjectPose as fallback.
Stores the original frame in YAML so successive attempts always offset from the same base.

Inputs:
  - yaml_path: path to task YAML
  - colliding_object: ID of the blocking placed object (e.g. "part_1")
  - operation_index: which MoveAndPlace failed (1, 2, 3...)
  - attempt: retry count — selects offset direction (+X, +Y, -X, -Y, diagonal)
""",
    args_schema=PlaceOffsetErrorInput,
    return_direct=False,
)

__all__ = ['place_offset_error', 'place_offset_error_func']
