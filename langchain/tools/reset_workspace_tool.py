"""
Reset Workspace Tool

This module provides a LangChain-compatible tool for resetting the robot workspace
via ROS2 service call (/reset_workspace).

Uses langchain_core for modern LangChain/LangGraph integration.
Implements singleton pattern for ROS2 node to avoid repeated creation/destruction.
"""

import rclpy
from rclpy.node import Node
from b_scene_interfaces.srv import ResetWorkspace
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Dict, Optional
import time


class ResetWorkspaceClient(Node):
    """ROS2 Service Client for resetting the workspace"""

    def __init__(self):
        super().__init__('reset_workspace_client_' + str(int(time.time())))
        self._client = self.create_client(ResetWorkspace, '/reset_workspace')

    def call_reset_service(self, do_not_move_robot: bool, move_duration: float, 
                          timeout_sec: float = 10.0) -> Dict:
        """
        Synchronously call the /reset_workspace service.

        Args:
            do_not_move_robot: If true, skip robot movement
            move_duration: Duration for movement operation
            timeout_sec: Maximum time to wait for service response

        Returns:
            Dict with success status, message, and execution time
        """
        # Wait for service availability
        if not self._client.wait_for_service(timeout_sec=5.0):
            return {
                'success': False,
                'message': 'Service /reset_workspace not available',
                'service_name': '/reset_workspace',
                'execution_time': 0.0
            }

        # Create request
        request = ResetWorkspace.Request()
        request.do_not_move_robot = do_not_move_robot
        request.move_duration = move_duration

        # Call service
        self.get_logger().info(f'Calling /reset_workspace with do_not_move_robot={do_not_move_robot}, '
                              f'move_duration={move_duration}')
        
        start_time = time.time()
        future = self._client.call_async(request)
        
        # Wait for service to complete with periodic spinning and logging
        self.get_logger().info(f'Waiting for service to complete (timeout: {timeout_sec}s)...')
        last_log_time = start_time
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Log progress every 2 seconds
            current_time = time.time()
            if current_time - last_log_time >= 2.0:
                elapsed = current_time - start_time
                self.get_logger().info(f'Waiting for /reset_workspace... ({elapsed:.1f}s elapsed)')
                last_log_time = current_time
            
            # Check for timeout
            if time.time() - start_time >= timeout_sec:
                self.get_logger().warn(f'Service call timed out after {timeout_sec} seconds')
                break
        
        execution_time = time.time() - start_time
        
        if not future.done():
            return {
                'success': False,
                'message': f'Service call timed out after {timeout_sec} seconds',
                'service_name': '/reset_workspace',
                'execution_time': execution_time
            }

        # Parse response
        response = future.result()
        
        if hasattr(response, 'success'):
            return {
                'success': response.success,
                'message': getattr(response, 'message', 'Workspace reset completed'),
                'service_name': '/reset_workspace',
                'execution_time': execution_time
            }
        else:
            return {
                'success': True,
                'message': 'Service call completed (no response details)',
                'service_name': '/reset_workspace',
                'execution_time': execution_time
            }


# Singleton instance - reuse across tool calls
_reset_client_instance: Optional[ResetWorkspaceClient] = None


def get_reset_client() -> ResetWorkspaceClient:
    """
    Get or create singleton ROS2 client instance.
    This avoids repeated node creation/destruction which can cause memory leaks.
    """
    global _reset_client_instance

    if _reset_client_instance is None:
        # Initialize rclpy if needed
        if not rclpy.ok():
            rclpy.init()
        _reset_client_instance = ResetWorkspaceClient()

    return _reset_client_instance


class ResetWorkspaceInput(BaseModel):
    """Input schema for reset workspace tool"""
    do_not_move_robot: bool = Field(
        default=False,
        description="If true, skip robot movement during workspace reset"
    )
    move_duration: float = Field(
        default=0.0,
        description="Duration in seconds for movement operation"
    )
    timeout_sec: float = Field(
        default=30.0,
        description="Maximum time in seconds to wait for service response (default: 30)"
    )


def reset_workspace_func(do_not_move_robot: bool = False, move_duration: float = 0.0,
                        timeout_sec: float = 30.0) -> Dict:
    """
    Reset the robot workspace by calling the /reset_workspace ROS2 service.

    This tool sends a reset command to the robot's workspace system. Use it when:
    - Before starting a new task sequence
    - After a collision or error to reset the workspace state
    - When switching between different workspace configurations

    Args:
        do_not_move_robot: If true, the robot will not move during reset
        move_duration: Duration for any movement operations (0.0 for automatic)
        timeout_sec: Maximum time to wait for the service call to complete

    Returns:
        Dict with:
        - success (bool): Whether the reset succeeded
        - message (str): Status message from the service
        - service_name (str): The service endpoint used
        - execution_time (float): How long the service call took
    """
    try:
        client = get_reset_client()
        result = client.call_reset_service(do_not_move_robot, move_duration, timeout_sec)
        return result
    except Exception as e:
        return {
            'success': False,
            'message': f'Exception during service call: {str(e)}',
            'service_name': '/reset_workspace',
            'execution_time': 0.0
        }


reset_workspace = StructuredTool.from_function(
    func=reset_workspace_func,
    name="reset_workspace",
    description="""Reset the robot workspace state.

Use this tool to reset or clear the robot workspace before starting new operations.
This is useful when:
- Starting a fresh task sequence
- Recovering from errors or collisions
- Switching workspace configurations

Parameters:
- do_not_move_robot: If true, skip robot movement (default: false)
- move_duration: Movement duration in seconds (default: 0.0 for auto)
- timeout_sec: Service timeout in seconds (default: 10)

Returns success status and execution details.
""",
    args_schema=ResetWorkspaceInput,
    return_direct=False
)


# Export tool for easy import
__all__ = ['reset_workspace']
