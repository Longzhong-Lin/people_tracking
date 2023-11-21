#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class VelocityEstimator:
    def __init__(self):
        rospy.init_node('velocity_estimator', anonymous=True)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.odom_sub = rospy.Subscriber('/Odometry', Odometry, self.odom_callback)
        self.cmd_vel_sub = rospy.Subscriber('/robot_3/cmd_vel', Twist, self.cmd_vel_callback)
        self.last_time = None
        self.last_position = None
        self.last_orientation = None
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.cmd_time = None
    
    def cmd_vel_callback(self, msg):
        # Extract linear and angular velocities from cmd_vel message
        self.cmd_v = msg.linear.x
        self.cmd_w = msg.angular.z
        self.cmd_time = rospy.Time.now().to_sec()

    def odom_callback(self, msg):
        # Extract current position and orientation
        current_position = np.array([msg.pose.pose.position.x,
                                     msg.pose.pose.position.y,
                                     msg.pose.pose.position.z])
        orientation_q = msg.pose.pose.orientation
        current_orientation = np.array([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

        current_time = msg.header.stamp.to_sec()
        if self.last_time is not None and self.last_time != current_time:
            dt = current_time - self.last_time

            # Calculate linear velocity in XY plane
            delta_position = current_position - self.last_position
            linear_velocity_xy = np.linalg.norm(delta_position[:2]) / dt

            # Calculate angular velocity around z-axis
            current_angle = euler_from_quaternion(current_orientation)
            last_angle = euler_from_quaternion(self.last_orientation)
            angular_velocity_z = - (current_angle[2] - last_angle[2]) / dt

            # Fill the Odometry message and publish it
            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = msg.header.frame_id
            odom_msg.child_frame_id = msg.child_frame_id

            odom_msg.pose = msg.pose

            if self.cmd_time is not None and 0 < (rospy.Time.now().to_sec() - self.cmd_time) < 0.2:
                odom_msg.twist.twist.linear.x = self.cmd_v
                odom_msg.twist.twist.angular.z = self.cmd_w
            else:
                odom_msg.twist.twist.linear.x = 0
                odom_msg.twist.twist.angular.z = 0
            
            odom_msg.twist.twist.linear.y = 0
            odom_msg.twist.twist.linear.z = 0
            odom_msg.twist.twist.angular.x = 0
            odom_msg.twist.twist.angular.y = 0

            self.odom_pub.publish(odom_msg)

        # Update last known position and time
        self.last_time = current_time
        self.last_position = current_position
        self.last_orientation = current_orientation

if __name__ == '__main__':
    try:
        ve = VelocityEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
