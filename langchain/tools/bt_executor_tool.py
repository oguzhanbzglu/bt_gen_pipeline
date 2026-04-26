"""
Behavior Tree Executor Tool

Executes Behavior Trees via ROS2 action server (/transfer_object).
Integrates RosoutSubscriber to capture MoveIt collision errors during execution.
"""

import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from b_robots_interfaces.action import TransferObject
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Import RosoutSubscriber from scripts/
PACKAGE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))
from rosout_callback import RosoutSubscriber


class TransferObjectClient(Node):
    """ROS2 Action Client for executing Behavior Trees."""

    def __init__(self):
        super().__init__('transfer_object_client_' + str(int(time.time())))
        self._action_client = ActionClient(self, TransferObject, '/transfer_object')

    def _spin_until_done(self, future, timeout_sec: float, rosout_node: RosoutSubscriber):
        """Spin this node and the rosout subscriber until the future completes or timeout."""
        start = time.time()
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.02)
            rclpy.spin_once(rosout_node, timeout_sec=0.0)
            if time.time() - start >= timeout_sec:
                break

    def execute_sync(self, btree_id: str, rosout_node: RosoutSubscriber, timeout_sec: float = 60.0) -> Dict:
        """
        Send BT execution goal and wait for result.
        Spins the rosout subscriber in parallel to capture collision logs.
        """
        rosout_node.reset()

        goal_msg = TransferObject.Goal()
        goal_msg.type = btree_id

        self.get_logger().info('Waiting for /transfer_object action server...')
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            return {
                'success': False,
                'message': 'Action server /transfer_object not available',
                'error_logs': 'Cannot connect to action server. Is bt_executor running?',
                'colliding_objects': '',
                'btree_id': btree_id,
                'execution_time': 0.0,
            }

        self.get_logger().info(f'Sending goal: {btree_id}')
        send_goal_future = self._action_client.send_goal_async(goal_msg)

        self._spin_until_done(send_goal_future, timeout_sec=5.0, rosout_node=rosout_node)
        goal_handle = send_goal_future.result()

        if goal_handle is None:
            return {
                'success': False,
                'message': 'Failed to send goal — no response from server',
                'error_logs': 'Goal sending failed. Server may be busy or unavailable.',
                'colliding_objects': '',
                'btree_id': btree_id,
                'execution_time': 0.0,
            }

        if not goal_handle.accepted:
            return {
                'success': False,
                'message': 'Goal rejected by action server',
                'error_logs': f'BT ID "{btree_id}" may not exist or be invalid.',
                'colliding_objects': '',
                'btree_id': btree_id,
                'execution_time': 0.0,
            }

        self.get_logger().info('Goal accepted, executing BT...')
        start_time = time.time()
        get_result_future = goal_handle.get_result_async()

        self._spin_until_done(get_result_future, timeout_sec=timeout_sec, rosout_node=rosout_node)
        execution_time = time.time() - start_time

        result = get_result_future.result()

        if result is None:
            return {
                'success': False,
                'message': f'Execution timed out after {timeout_sec}s',
                'error_logs': 'BT execution timed out.',
                'colliding_objects': rosout_node.get_colliding_objects_str(),
                'bt_error_node': rosout_node.get_result().get('bt_error_node'),
                'btree_id': btree_id,
                'execution_time': execution_time,
            }

        result_data = result.result
        colliding_objects = rosout_node.get_colliding_objects_str()

        rosout_result = rosout_node.get_result()
        bt_error_node = rosout_result.get('bt_error_node')
        
        print(f"[bt_executor] rosout_result: {rosout_result}")
        print(f"[bt_executor] colliding_objects: '{colliding_objects}'")
        print(f"[bt_executor] bt_error_node: {bt_error_node}")

        if result_data.success:
            return {
                'success': True,
                'message': getattr(result_data, 'return_message', 'Execution completed successfully'),
                'error_logs': '',
                'colliding_objects': '',
                'bt_error_node': bt_error_node,
                'btree_id': btree_id,
                'execution_time': execution_time,
            }
        else:
            error_msg = getattr(result_data, 'return_message', 'Unknown error')
            return {
                'success': False,
                'message': f'Execution failed: {error_msg}',
                'error_logs': error_msg,
                'colliding_objects': colliding_objects,
                'bt_error_node': bt_error_node,
                'btree_id': btree_id,
                'execution_time': execution_time,
            }


# Singletons — created once, reused across tool calls
_client_instance: Optional[TransferObjectClient] = None
_rosout_instance: Optional[RosoutSubscriber] = None


def _get_nodes():
    global _client_instance, _rosout_instance
    if _client_instance is None:
        if not rclpy.ok():
            rclpy.init()
        _client_instance = TransferObjectClient()
        _rosout_instance = RosoutSubscriber()
    return _client_instance, _rosout_instance


class BTExecutorInput(BaseModel):
    yaml_path: str = Field(description="Path to the YAML config file containing behavior_tree.name")
    timeout_sec: float = Field(default=60.0, description="Max seconds to wait for execution")


def call_transfer_object_func(yaml_path: str, timeout_sec: float = 60.0) -> Dict:
    """
    Execute a Behavior Tree via /transfer_object action server.

    Reads YAML to extract the BT ID, sends execution request, waits for result.
    Captures MoveIt collision objects from /rosout during execution.

    Returns:
        success (bool), message (str), error_logs (str),
        colliding_objects (str, comma-separated), btree_id (str), execution_time (float)
    """
    try:
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
    except FileNotFoundError:
        return {
            'success': False,
            'message': f'YAML file not found: {yaml_path}',
            'error_logs': f'File not found: {yaml_path}',
            'colliding_objects': '',
            'btree_id': None,
            'execution_time': 0.0,
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Failed to read YAML: {str(e)}',
            'error_logs': str(e),
            'colliding_objects': '',
            'btree_id': None,
            'execution_time': 0.0,
        }

    try:
        yaml_dict = yaml.safe_load(yaml_content)
        btree_id = yaml_dict.get('behavior_tree', {}).get('name')
        if not btree_id:
            return {
                'success': False,
                'message': 'Could not extract BT ID from YAML',
                'error_logs': 'YAML must contain behavior_tree.name',
                'colliding_objects': '',
                'btree_id': None,
                'execution_time': 0.0,
            }
    except yaml.YAMLError as e:
        return {
            'success': False,
            'message': f'Failed to parse YAML: {str(e)}',
            'error_logs': str(e),
            'colliding_objects': '',
            'btree_id': None,
            'execution_time': 0.0,
        }

    try:
        client, rosout = _get_nodes()
        return client.execute_sync(btree_id, rosout, timeout_sec)
    except Exception as e:
        return {
            'success': False,
            'message': f'Exception during execution: {str(e)}',
            'error_logs': str(e),
            'colliding_objects': '',
            'btree_id': btree_id,
            'execution_time': 0.0,
        }


call_transfer_object = StructuredTool.from_function(
    func=call_transfer_object_func,
    name="call_transfer_object",
    description="""Execute a Behavior Tree via ROS2 /transfer_object action server.

Reads yaml_path to extract the BT ID, executes it on the robot, and waits for result.
Captures MoveIt collision contacts from /rosout during execution.

Returns:
- success: whether execution succeeded
- error_logs: failure message from bt_executor
- colliding_objects: comma-separated object names detected in collision (e.g. "part_1, table")
- execution_time: seconds taken
""",
    args_schema=BTExecutorInput,
    return_direct=False,
)

__all__ = ['call_transfer_object']
