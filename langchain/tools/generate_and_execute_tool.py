"""
Generate and Execute BT Tool

Single tool for the LLM to call. Owns the full loop in Python:
  1. Generate BT XML from YAML
  2. Execute on robot
  3. On failure → smart_collision_recovery → reset → reload_scene → regenerate → retry
  4. Repeat up to MAX_RETRIES times
  5. Return final result

The LLM only needs to call this once per task. No multi-step orchestration needed.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

PACKAGE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "langchain" / "tools"))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from xml_gen_tools import xml_generator_func
from bt_executor_tool import call_transfer_object_func
from reset_workspace_tool import reset_workspace_func
from reload_scene_tool import reload_scene_func
from smart_collision_recovery_tool import smart_collision_recovery_func

MAX_RETRIES = 5


class GenerateAndExecuteInput(BaseModel):
    yaml_path: str = Field(
        description="Absolute path to the task YAML configuration file"
    )
    timeout_sec: float = Field(
        default=60.0,
        description="Max seconds to wait for each BT execution attempt"
    )


def generate_and_execute_func(yaml_path: str, timeout_sec: float = 60.0) -> Dict:
    """
    Full generate → execute → recover → retry loop in Python.

    Steps per attempt:
      1. xml_generator  — generate BT XML from YAML
      2. call_transfer_object — execute on robot
      3. (on failure) smart_collision_recovery — patch YAML
      4. (on failure) reset_workspace — return robot to home
      5. (on failure) reload_scene — re-add objects to MoveIt scene
      Then retry from step 1.

    Returns final success/failure with attempt history.
    """
    history = []

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n[generate_and_execute] Attempt {attempt}/{MAX_RETRIES}")

        # ── Step 1: Generate XML ──────────────────────────────────────────────
        gen_result = xml_generator_func(yaml_content=yaml_path)
        if not gen_result.get('success'):
            msg = f"Attempt {attempt}: XML generation failed — {gen_result.get('error')}"
            history.append({'attempt': attempt, 'stage': 'xml_generator', 'result': msg})
            print(f"[generate_and_execute] {msg}")
            break  # YAML is broken, no point retrying

        print(f"[generate_and_execute] XML generated: {gen_result.get('output_path')}")

        # ── Step 2: Execute BT ────────────────────────────────────────────────
        exec_result = call_transfer_object_func(yaml_path=yaml_path, timeout_sec=timeout_sec)
        history.append({
            'attempt': attempt,
            'stage': 'call_transfer_object',
            'success': exec_result.get('success'),
            'message': exec_result.get('message'),
            'colliding_objects': exec_result.get('colliding_objects', ''),
        })

        if exec_result.get('success'):
            print(f"[generate_and_execute] SUCCESS on attempt {attempt}")
            return {
                'success': True,
                'message': f"BT executed successfully on attempt {attempt}.",
                'attempts': attempt,
                'history': history,
            }

        # ── Step 3: Collision recovery ────────────────────────────────────────
        colliding_objects = exec_result.get('colliding_objects', '')
        bt_error_node     = json.dumps(exec_result.get('bt_error_node')) if exec_result.get('bt_error_node') else ''
        error_message     = exec_result.get('error_logs', '')
        print(f"[generate_and_execute] Execution failed. colliding_objects='{colliding_objects}'")
        print(f"[generate_and_execute] error_message='{error_message}'")

        if attempt < MAX_RETRIES:
            recovery_raw = smart_collision_recovery_func(
                yaml_path=yaml_path,
                colliding_objects=colliding_objects,
                bt_error_node=bt_error_node,
                error_message=error_message,
                attempt=attempt,
            )
            recovery = json.loads(recovery_raw) if isinstance(recovery_raw, str) else recovery_raw
            history.append({
                'attempt': attempt,
                'stage': 'smart_collision_recovery',
                'recovery_type': recovery.get('recovery_type'),
                'success': recovery.get('success'),
                'message': recovery.get('message'),
            })
            print(f"[generate_and_execute] Recovery ({recovery.get('recovery_type')}): {recovery.get('message')}")

            # ── Step 4: Reset workspace ───────────────────────────────────────
            reset_result = reset_workspace_func(
                do_not_move_robot=False,
                move_duration=5.0,
                timeout_sec=60.0,
            )
            print(f"[generate_and_execute] Reset workspace: {reset_result.get('message')}")

            # ── Step 5: Reload scene ──────────────────────────────────────────
            reload_result = reload_scene_func(yaml_path=yaml_path, timeout_sec=30.0)
            print(f"[generate_and_execute] Reload scene: {reload_result.get('message')}")

    return {
        'success': False,
        'message': f"BT execution failed after {MAX_RETRIES} attempts.",
        'attempts': MAX_RETRIES,
        'history': history,
    }


generate_and_execute = StructuredTool.from_function(
    func=generate_and_execute_func,
    name="generate_and_execute",
    description="""Generate a Behavior Tree from a YAML file and execute it on the robot.

Handles the full loop automatically:
  - Generates BT XML from the YAML configuration
  - Executes it on the robot via /transfer_object action
  - On failure: applies smart collision recovery (offset placement or allowed_touch_links)
  - Resets and reloads the scene, then retries — up to 5 times
  - Stops immediately when execution succeeds

Input:
  - yaml_path: absolute path to the task YAML file (e.g. /home/.../case_2.yaml)
  - timeout_sec: max seconds per execution attempt (default: 60)

Returns: success status, number of attempts, and attempt history.
""",
    args_schema=GenerateAndExecuteInput,
    return_direct=False,
)

__all__ = ['generate_and_execute']
