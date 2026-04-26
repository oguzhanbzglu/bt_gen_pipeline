"""
Smart Collision Recovery Tool

Simple and robust collision recovery strategy:

  PRIORITY RULE: ALWAYS increase goal_tolerances FIRST for ANY failure
  
  Why: Collision detection from rosout is unreliable (move_group logs don't reach bt_executor)
  
  Strategy:
    - Increases THREE YAML parameters simultaneously:
      * goal_tolerances (for ReachPreGraspOmpl - Ompl motion)
      * pick_goal_tolerances (for MoveGraspLin during pick - Lin motion)
      * place_goal_tolerances (for MoveGraspLin during place - Lin motion)
    - Aggressive increments: position +5cm, orientation +0.3rad per attempt
    - Position tolerance MAX limit: 15cm (0.15m)
    - Orientation tolerance: NO limit (can keep increasing)
  
  When position tolerance reaches MAX (15cm):
    - Ompl motion / Fixture collision → switch to increasing approach_distance
    - Lin motion / Task collision / Unknown → keep increasing orientation tolerance only
"""

import re
import sys
import yaml
import json
from pathlib import Path
from typing import Tuple
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

PACKAGE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "langchain" / "tools"))

from collision_error_tool import collision_error_func
from place_offset_error_tool import place_offset_error_func


def _get_task_object_ids(config: dict) -> set:
    ids = set()
    for key, value in config.items():
        if isinstance(value, dict) and ('PickAndMove' in key or 'MoveAndPlace' in key):
            obj_id = value.get('object_id', '')
            if obj_id:
                ids.add(obj_id)
    return ids


def _detect_stacking_operation(config: dict, task_collision: str) -> int:
    """
    Find which MoveAndPlace operation should be offset because the colliding
    task object occupies its target place_frame_id.

    Returns the operation index (1, 2, ...) of the op that needs offsetting.
    Falls back to the last MoveAndPlace if no exact match found.
    """
    # place_frame_id of the already-placed colliding object
    colliding_place_frame = None
    for key, value in config.items():
        if isinstance(value, dict) and 'MoveAndPlace' in key:
            if value.get('object_id') == task_collision:
                colliding_place_frame = value.get('place_frame_id')
                break

    if colliding_place_frame:
        # Find the OTHER operation targeting the same frame
        for key, value in config.items():
            if isinstance(value, dict) and 'MoveAndPlace' in key:
                if value.get('object_id') == task_collision:
                    continue
                if value.get('place_frame_id') == colliding_place_frame:
                    match = re.search(r'MoveAndPlace_(\d+)', key)
                    return int(match.group(1)) if match else 1

    # Fallback: return the last MoveAndPlace index
    max_idx = 1
    for key in config:
        if 'MoveAndPlace' in key:
            match = re.search(r'MoveAndPlace_(\d+)', key)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    return max_idx


_BASE_APPROACH_DISTANCE = 0.1
_APPROACH_DISTANCE_STEP = 0.05  # add 5cm per attempt
_BASE_GOAL_TOLERANCES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [x, y, z, roll, pitch, yaw]
_GOAL_TOLERANCE_POSITION_STEP = 0.05  # add 5cm per attempt to position (x,y,z)
_GOAL_TOLERANCE_ORIENTATION_STEP = 0.3  # add 0.3 rad (~17 degrees) per attempt to orientation (r,p,y)
_MAX_GOAL_TOLERANCE_POSITION = 0.15  # maximum 15cm for position tolerances


def _increase_approach_distance(yaml_path: str, attempt: int) -> dict:
    """Increase approach_distance when we don't know what collided."""
    try:
        print(f"[_increase_approach_distance] Reading YAML from: {yaml_path}")
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        current = config.get('behavior_tree', {}).get('approach_distance', _BASE_APPROACH_DISTANCE)
        new_dist = round(_BASE_APPROACH_DISTANCE + _APPROACH_DISTANCE_STEP * attempt, 3)
        if 'behavior_tree' not in config:
            config['behavior_tree'] = {}
        config['behavior_tree']['approach_distance'] = new_dist
        print(f"[_increase_approach_distance] Writing to YAML: {yaml_path}")
        print(f"[_increase_approach_distance] New approach_distance: {new_dist}")
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"[_increase_approach_distance] YAML written successfully")
        return {
            'success': True,
            'message': f'No collision objects detected. Increased approach_distance: {current} → {new_dist}',
        }
    except Exception as e:
        return {'success': False, 'message': f'Failed to increase approach_distance: {e}'}


def _increase_goal_tolerances(yaml_path: str, attempt: int) -> dict:
    """Increase goal_tolerances for collisions.
    
    Updates THREE parameters in YAML:
    - goal_tolerances: used by ReachPreGraspOmpl (Ompl motion)
    - pick_goal_tolerances: used by MoveGraspLin during pick (Lin motion)
    - place_goal_tolerances: used by MoveGraspLin during place (Lin motion)
    
    goal_tolerances format: [x, y, z, roll, pitch, yaw]
    - x, y, z: position tolerances in meters (max: 0.15m)
    - roll, pitch, yaw: orientation tolerances in radians (no max limit)
    """
    try:
        print(f"[_increase_goal_tolerances] Reading YAML from: {yaml_path}")
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get current goal_tolerances (array of 6 values: [x, y, z, r, p, y])
        current = config.get('behavior_tree', {}).get('goal_tolerances', _BASE_GOAL_TOLERANCES[:])
        
        # Ensure it's a list
        if not isinstance(current, list):
            current = _BASE_GOAL_TOLERANCES[:]
        
        # Calculate new tolerances
        # Increase position tolerances (x, y, z) with max limit
        pos_increment = round(_GOAL_TOLERANCE_POSITION_STEP * attempt, 4)
        pos_increment = min(pos_increment, _MAX_GOAL_TOLERANCE_POSITION)  # Cap at max
        
        # Increase orientation tolerances (roll, pitch, yaw) - no limit
        orient_increment = round(_GOAL_TOLERANCE_ORIENTATION_STEP * attempt, 4)
        
        new_tolerances = [
            pos_increment,     # x
            pos_increment,     # y
            pos_increment,     # z
            orient_increment,  # roll
            orient_increment,  # pitch
            orient_increment   # yaw
        ]
        
        # Check if position tolerance hit the max
        at_max = (pos_increment >= _MAX_GOAL_TOLERANCE_POSITION)
        
        if 'behavior_tree' not in config:
            config['behavior_tree'] = {}
        
        # Update ALL three goal_tolerances parameters
        config['behavior_tree']['goal_tolerances'] = new_tolerances
        config['behavior_tree']['pick_goal_tolerances'] = new_tolerances
        config['behavior_tree']['place_goal_tolerances'] = new_tolerances
        
        print(f"[_increase_goal_tolerances] Writing to YAML: {yaml_path}")
        print(f"[_increase_goal_tolerances] New goal_tolerances (Ompl): {new_tolerances}")
        print(f"[_increase_goal_tolerances] New pick_goal_tolerances (Lin pick): {new_tolerances}")
        print(f"[_increase_goal_tolerances] New place_goal_tolerances (Lin place): {new_tolerances}")
        if at_max:
            print(f"[_increase_goal_tolerances] WARNING: Position tolerance at maximum ({_MAX_GOAL_TOLERANCE_POSITION}m)")
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"[_increase_goal_tolerances] YAML written successfully")
        
        return {
            'success': True,
            'message': f'Increased goal_tolerances (Ompl + Lin) from {current} to {new_tolerances} ([x,y,z,r,p,y])',
            'at_max': at_max,
        }
    except Exception as e:
        return {'success': False, 'message': f'Failed to increase goal_tolerances: {e}'}


class SmartCollisionRecoveryInput(BaseModel):
    yaml_path: str = Field(description="Path to task YAML file")
    colliding_objects: str = Field(
        default="",
        description="Comma-separated colliding object names from call_transfer_object result"
    )
    bt_error_node: str = Field(
        default="",
        description="bt_error_node JSON string from call_transfer_object result (optional hint)"
    )
    error_message: str = Field(
        default="",
        description="Error message from BT execution result"
    )
    attempt: int = Field(default=1, description="Retry attempt number (1, 2, 3...)")


def smart_collision_recovery_func(
    yaml_path: str,
    colliding_objects: str = "",
    bt_error_node: str = "",
    error_message: str = "",
    attempt: int = 1,
) -> str:
    """
    Route collision recovery with simple priority rules:
      - ALWAYS increase goal_tolerances first (for ALL failures)
      - When position tolerance reaches MAX:
        * Ompl/Fixture → increase approach_distance
        * Lin/Task/Unknown → keep increasing orientation tolerance
    """
    print(f"\n[smart_collision_recovery] Called with:")
    print(f"  yaml_path: {yaml_path}")
    print(f"  colliding_objects: '{colliding_objects}' (type: {type(colliding_objects)})")
    print(f"  bt_error_node: '{bt_error_node}' (type: {type(bt_error_node)})")
    print(f"  error_message: '{error_message}' (type: {type(error_message)})")
    print(f"  attempt: {attempt}")
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return json.dumps({'success': False, 'message': f'Cannot read YAML: {e}'})

    # Parse bt_error_node to determine motion type
    error_node_info = None
    motion_type = None  # 'Lin', 'Ompl', or None
    
    print(f"[smart_collision_recovery] bt_error_node is empty: {not bt_error_node}")
    print(f"[smart_collision_recovery] bt_error_node bool: {bool(bt_error_node)}")
    
    # Try to get motion type from bt_error_node first
    if bt_error_node:
        try:
            error_node_info = json.loads(bt_error_node) if isinstance(bt_error_node, str) else bt_error_node
            print(f"[smart_collision_recovery] error_node_info: {error_node_info}")
            error_node = error_node_info.get('error_node', '')
            print(f"[smart_collision_recovery] Parsed error_node: {error_node}")
            if 'MoveGraspLin' in error_node:
                motion_type = 'Lin'
                print(f"[smart_collision_recovery] Detected Lin motion from bt_error_node")
            elif 'ReachPreGraspOmpl' in error_node or 'Ompl' in error_node:
                motion_type = 'Ompl'
                print(f"[smart_collision_recovery] Detected Ompl motion from bt_error_node")
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[smart_collision_recovery] Failed to parse bt_error_node: {e}")
    
    # If bt_error_node didn't work, try to infer from error_message
    if not motion_type and error_message:
        print(f"[smart_collision_recovery] Trying to infer motion type from error_message")
        if 'MoveGraspLin' in error_message or 'Lin' in error_message:
            motion_type = 'Lin'
            print(f"[smart_collision_recovery] Inferred Lin motion from error_message")
        elif 'ReachPreGraspOmpl' in error_message or 'Ompl' in error_message or 'OMPL' in error_message:
            motion_type = 'Ompl'
            print(f"[smart_collision_recovery] Inferred Ompl motion from error_message")
    
    if not motion_type:
        print(f"[smart_collision_recovery] Could not determine motion type - will use default heuristics")

    task_objects = _get_task_object_ids(config)
    objects_list = [o.strip() for o in colliding_objects.split(',') if o.strip()]

    task_collisions    = [o for o in objects_list if o in task_objects]
    fixture_collisions = [o for o in objects_list if o not in task_objects]
    
    print(f"[smart_collision_recovery] task_objects: {task_objects}")
    print(f"[smart_collision_recovery] objects_list: {objects_list}")
    print(f"[smart_collision_recovery] task_collisions: {task_collisions}")
    print(f"[smart_collision_recovery] fixture_collisions: {fixture_collisions}")

    # ── PRIORITY RULE: ALWAYS increase goal_tolerances first for ANY failure ──
    # This applies to ALL collision types AND unknown failures
    # Only after goal_tolerances max is reached, try other strategies
    
    has_any_collision = bool(task_collisions or fixture_collisions)
    
    # ALWAYS try goal_tolerances first, even if no collision objects detected
    # (collision detection from rosout is unreliable)
    if True:  # Always try goal_tolerances first
        if has_any_collision:
            print(f"[smart_collision_recovery] Collision detected - using goal_tolerances strategy")
        else:
            print(f"[smart_collision_recovery] No collision detected but trying goal_tolerances first anyway")
        
        collision_info = []
        if task_collisions:
            collision_info.append(f"task objects: {task_collisions}")
        if fixture_collisions:
            collision_info.append(f"fixtures: {fixture_collisions}")
        if not collision_info:
            collision_info.append("unknown failure")
        
        # Try to increase goal_tolerances
        result = _increase_goal_tolerances(yaml_path, attempt)
        at_max = result.get('at_max', False)
        
        # If position tolerance NOT at max, keep increasing goal_tolerances
        if not at_max:
            print(f"[smart_collision_recovery] CASE 1: Increase goal_tolerances (attempt {attempt})")
            result['recovery_type'] = 'goal_tolerances'
            result['reason'] = (
                f"Collision detected ({', '.join(collision_info)}). "
                f"Increasing goal_tolerances (attempt {attempt})."
            )
            return json.dumps(result)
        
        # Position tolerance at MAX - try other strategies based on motion type and collision type
        print(f"[smart_collision_recovery] Goal_tolerances at maximum - trying alternative strategies")
        
        # ── For Ompl motion: increase approach_distance ──────────────────────
        if motion_type == 'Ompl' or (fixture_collisions and not task_collisions):
            print(f"[smart_collision_recovery] CASE 2: Ompl/Fixture collision - trying approach_distance")
            result = _increase_approach_distance(yaml_path, attempt)
            result['recovery_type'] = 'approach_distance'
            result['reason'] = (
                f"Ompl motion or fixture collision ({', '.join(collision_info)}). "
                f"Goal_tolerances at max, increasing approach_distance (attempt {attempt})."
            )
            return json.dumps(result)
        
        # ── For Lin motion or task objects: keep increasing goal_tolerances (orientation can still increase) ──
        print(f"[smart_collision_recovery] CASE 3: Lin motion or unknown - continue with goal_tolerances")
        result['recovery_type'] = 'goal_tolerances_continue'
        result['reason'] = (
            f"Lin motion, task object collision, or unknown failure ({', '.join(collision_info)}). "
            f"Position tolerance at max but orientation can still increase (attempt {attempt})."
        )
        return json.dumps(result)


smart_collision_recovery = StructuredTool.from_function(
    func=smart_collision_recovery_func,
    name="smart_collision_recovery",
    description="""Recover from a BT execution collision with aggressive goal_tolerances strategy.

Strategy:
  ANY collision detected:
    - Always increase goal_tolerances first (position: +5cm, orientation: +0.3rad per attempt)
    - Position tolerance MAX: 15cm, Orientation: NO limit
    - When position tolerance reaches MAX:
      * Ompl motion / Fixture collision → increase approach_distance
      * Lin motion / Task collision → continue increasing orientation tolerance
  
  No collision detected:
    - Increase approach_distance as fallback

Inputs:
  - yaml_path: path to task YAML
  - colliding_objects: from call_transfer_object result (comma-separated)
  - bt_error_node: JSON string from call_transfer_object result (optional)
  - error_message: error message from BT execution (optional)
  - attempt: retry count (1, 2, 3...)
""",
    args_schema=SmartCollisionRecoveryInput,
    return_direct=False,
)

__all__ = ['smart_collision_recovery']
