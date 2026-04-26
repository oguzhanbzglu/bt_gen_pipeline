"""
Reload Scene Tool

Reloads the MoveIt planning scene by calling /load_workspace.
Must be called after reset_workspace because reset removes objects from the
MoveIt planning scene while TF frames remain published — causing SpawnObjects
in the next BT to skip reloading (it sees the frame and assumes the scene is ready).
"""

import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from b_scene_interfaces.srv import AddObjects
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class LoadWorkspaceClient(Node):
    """ROS2 Service Client for /load_workspace (type: AddObjects)."""

    def __init__(self):
        super().__init__('load_workspace_client_' + str(int(time.time())))
        self._client = self.create_client(AddObjects, '/load_workspace')

    def call_load_workspace(self, scene_file_path: str, timeout_sec: float = 30.0) -> Dict:
        if not self._client.wait_for_service(timeout_sec=5.0):
            return {
                'success': False,
                'message': 'Service /load_workspace not available',
            }

        request = AddObjects.Request()
        request.file_path = scene_file_path
        request.as_marker = False

        self.get_logger().info(f'Reloading scene from: {scene_file_path}')
        start_time = time.time()
        future = self._client.call_async(request)

        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time >= timeout_sec:
                return {
                    'success': False,
                    'message': f'/load_workspace timed out after {timeout_sec}s',
                }

        response = future.result()
        return {
            'success': response.success,
            'message': response.message,
        }


_load_client_instance: Optional[LoadWorkspaceClient] = None


def _get_load_client() -> LoadWorkspaceClient:
    global _load_client_instance
    if _load_client_instance is None:
        if not rclpy.ok():
            rclpy.init()
        _load_client_instance = LoadWorkspaceClient()
    return _load_client_instance


class ReloadSceneInput(BaseModel):
    yaml_path: str = Field(
        description="Path to the task YAML file containing scene.file_path"
    )
    timeout_sec: float = Field(
        default=30.0,
        description="Max seconds to wait for /load_workspace service"
    )


def reload_scene_func(yaml_path: str, timeout_sec: float = 30.0) -> Dict:
    """
    Reload the MoveIt planning scene after a workspace reset.

    reset_workspace removes objects from the MoveIt planning scene but TF frames
    remain published. Without this step, SpawnObjects in the next BT skips
    reloading (frame already exists) and the robot operates on an empty scene.

    Reads scene.file_path from the task YAML and calls /load_workspace.
    """
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        return {'success': False, 'message': f'YAML not found: {yaml_path}'}
    except Exception as e:
        return {'success': False, 'message': f'Failed to read YAML: {e}'}

    if config is None:
        return {'success': False, 'message': f'YAML file is empty or invalid: {yaml_path}'}
    scene_file_path = config.get('scene', {}).get('file_path', '')
    if not scene_file_path:
        return {'success': False, 'message': 'No scene.file_path found in YAML'}

    try:
        client = _get_load_client()
        result = client.call_load_workspace(scene_file_path, timeout_sec)
        if result['success']:
            result['message'] = f"Scene reloaded from: {scene_file_path}"
        return result
    except Exception as e:
        return {'success': False, 'message': f'Exception: {e}'}


reload_scene = StructuredTool.from_function(
    func=reload_scene_func,
    name="reload_scene",
    description="""Reload the MoveIt planning scene after reset_workspace.

IMPORTANT: Always call this after reset_workspace during collision recovery.
reset_workspace removes scene objects from MoveIt but TF frames remain published.
Without this, the next BT execution skips scene loading and operates on an empty scene.

Input:
- yaml_path: path to the task YAML (reads scene.file_path from it)

Calls /load_workspace to re-add all objects to the MoveIt planning scene.
""",
    args_schema=ReloadSceneInput,
    return_direct=False,
)

__all__ = ['reload_scene']
