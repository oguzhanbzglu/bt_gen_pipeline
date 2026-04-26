import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import PoseStamped


class TCPPosePublisher(Node):

    def __init__(self):
        super().__init__('tcp_pose_publisher')

        # TF buffer and listener (IMPORTANT: spin_thread=True)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Publisher
        self.publisher = self.create_publisher(PoseStamped, '/tcp_pose', 10)

        # Timer (20 Hz)
        self.timer = self.create_timer(0.05, self.publish_pose)

        self.get_logger().info("TCP Pose Publisher started.")

    def publish_pose(self):
        target_frame = 'base_link'
        source_frame = 'centric_gripper_tcp_link'

        try:
            # Check if transform is available
            if self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                Time()
            ):
                # Lookup transform (latest available)
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.5)
                )

                # Convert to PoseStamped
                pose_msg = PoseStamped()
                pose_msg.header = transform.header

                pose_msg.pose.position.x = transform.transform.translation.x
                pose_msg.pose.position.y = transform.transform.translation.y
                pose_msg.pose.position.z = transform.transform.translation.z

                pose_msg.pose.orientation = transform.transform.rotation

                # Publish
                self.publisher.publish(pose_msg)

                self.get_logger().info("Publishing TCP pose...", throttle_duration_sec=2.0)

            else:
                self.get_logger().warn(
                    f"Transform not available: {target_frame} -> {source_frame}",
                    throttle_duration_sec=2.0
                )

        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {str(e)}",
                throttle_duration_sec=2.0
            )


def main(args=None):
    rclpy.init(args=args)

    node = TCPPosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()