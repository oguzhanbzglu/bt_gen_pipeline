import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Log
import re
import json


class RosoutSubscriber(Node):

    def __init__(self):
        super().__init__('rosout_subscriber')
        self.sub = self.create_subscription(Log, '/rosout', self.rosout_callback, 10)
        self.collision_objects_ = []
        self.error_node_ = None
        self.last_bt_error_msg = None

    def reset(self):
        self.collision_objects_ = []
        self.error_node_ = None
        self.last_bt_error_msg = None

    def extract_collision_objects(self, message: str):
        """Extract collision object names from a MoveIt log message."""
        pattern = r"between '([^']+)'(?: \(type '[^']+'\))? and '([^']+)'(?: \(type '[^']+'\))?"
        match = re.search(pattern, message)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def extract_bt_error_node(self, error_msg: str, failure_msg: str):
        match = re.search(r"Error in '([^']+)'", error_msg)
        if match:
            error_node_name = match.group(1)
            nodes_match = re.search(r"failure nodes:\s*(.+)", failure_msg)
            if nodes_match:
                nodes = [n.strip() for n in nodes_match.group(1).split(',')]
                for node in nodes:
                    if error_node_name in node:
                        return node
        return None

    def rosout_callback(self, msg: Log):
        logger = msg.name
        message = msg.msg

        # Collision detection — match any MoveIt collision-related logger
        COLLISION_LOGGERS = (
            'collision_detection',
            'moveit_collision_detection',
            'planning_scene',
            'move_group',
        )
        if any(kw in logger for kw in COLLISION_LOGGERS):
            obj1, obj2 = self.extract_collision_objects(message)
            if obj1 and obj2:
                pair = {'object1': obj1, 'object2': obj2}
                if pair not in self.collision_objects_:
                    self.collision_objects_.append(pair)
                    self.get_logger().info(f"Collision detected: '{obj1}' and '{obj2}'")

        # BT error node detection
        if logger == 'bt_executor_node':
            print(f"[rosout_callback] bt_executor_node message: {message[:200]}")  # First 200 chars
            if "Error in '" in message:
                self.last_bt_error_msg = message
                print(f"[rosout_callback] Captured error message: {message[:200]}")
            if 'failure nodes:' in message and self.last_bt_error_msg:
                print(f"[rosout_callback] Found 'failure nodes:' in message")
                exact_node = self.extract_bt_error_node(self.last_bt_error_msg, message)
                tree_match = re.search(r"Tree '([^']+)'", message)
                tree_name = tree_match.group(1) if tree_match else None
                print(f"[rosout_callback] Extracted node: {exact_node}, tree: {tree_name}")
                if exact_node:
                    self.error_node_ = {'error_node': exact_node, 'bt_id': tree_name}
                    print(f"[rosout_callback] Set error_node_: {self.error_node_}")

    def get_result(self) -> dict:
        return {
            'collide_objects': self.collision_objects_,
            'bt_error_node': self.error_node_,
        }

    def get_colliding_objects_str(self) -> str:
        """Return comma-separated list of all unique colliding object names."""
        names = set()
        for pair in self.collision_objects_:
            names.add(pair['object1'])
            names.add(pair['object2'])
        return ', '.join(sorted(names))


def main():
    rclpy.init()
    node = RosoutSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
