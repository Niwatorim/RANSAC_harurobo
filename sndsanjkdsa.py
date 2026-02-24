import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from can_comm.ransac import get_rings
import sys


class LidarTest(Node):
    def __init__(self):
        super().__init__('lidar_test')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',  
            self.lidar_callback,
            10)

        self.publish = self.create_publisher(Twist, '/cmd_vel', 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/ransac_markers', 10)

        self.target_distance_to_wall = 0.5 #how far from wall should the robot be
        self.kp = 1.0
        self.get_logger().info("lidar test node started, see in RViz")

    def lidar_callback(self, msg):
        # angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        # ranges = np.array(msg.ranges)

        # valid_mask = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        # ranges = ranges[valid_mask]
        # angles = angles[valid_mask]

        # x = ranges * np.cos(angles)
        # y = ranges * np.sin(angles)

        angle,x,y = get_rings(msg)

        self.get_logger().info(f"Detected {angle}, x:  {x} , y: {y}")



        # self.publish_markers(poles, walls, msg.header)

        # for i, pole in enumerate(poles):
        #     self.get_logger().info(
        #         f"Pole {i+1}: Center=({pole['cx']:.3f}, {pole['cy']:.3f}"
        #         f"Radius={pole['radius']:.3f}m"
        #     )
       
        # for i, wall in enumerate(walls):
        #     self.get_logger().info(
        #         f"  Wall {i+1}: Angle={wall['angle']:.2f} degrees"
        #     )
   
    def publish_markers(self, poles, walls, header):
        marker_array = MarkerArray()
        my_markers = []

        #delete previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        my_markers.append(delete_marker)

        marker_id = 0

        for pole in poles:
            m = Marker()
            m.header = header
            m.id = marker_id
            marker_id += 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            #center of pole
            m.pose.position.x = float(pole['cx'])
            m.pose.position.y = float(pole['cy'])
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0

            # Scale (Radius * 2 = Diameter. Z is height)
            m.scale.x = float(pole['radius'] * 2)
            m.scale.y = float(pole['radius'] * 2)
            m.scale.z = 0.5  # Make it 0.5 meters tall in RViz

            # Color (Red, semi-transparent)
            m.color.r = 1.0
            m.color.a = 0.8

            my_markers.append(m)
       
        for wall in walls:
            m = Marker()
            m.header = header
            m.ns = "walls"
            m.id = marker_id
            marker_id += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0

            # Line width
            m.scale.x = 0.05

            # Color (Blue)
            m.color.b = 1.0
            m.color.a = 0.8

            # The 'wall' array has two sets of (x,y) coordinates [start point, end point]
            p1 = Point()
            p1.x = float(wall['wall'][0, 0])
            p1.y = float(wall['wall'][0, 1])
            p1.z = 0.0

            p2 = Point()
            p2.x = float(wall['wall'][1, 0])
            p2.y = float(wall['wall'][1, 1])
            p2.z = 0.0

            m.points.append(p1)
            m.points.append(p2)

            my_markers.append(m)
       
        marker_array.markers = my_markers
       
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    lidar_subscriber = LidarTest()
    rclpy.spin(lidar_subscriber)
    lidar_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
