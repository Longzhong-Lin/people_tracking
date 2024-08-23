#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class VelocityEstimator:
    def __init__(self):
        rospy.init_node('velocity_estimator', anonymous=True)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.odom_sub = rospy.Subscriber('/Odometry', Odometry, self.odom_callback)
        self.last_time = None
        self.last_position = None
        self.last_orientation = None

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

            odom_msg.twist.twist.linear.x = linear_velocity_xy
            odom_msg.twist.twist.linear.y = 0
            odom_msg.twist.twist.linear.z = 0
            odom_msg.twist.twist.angular.x = 0
            odom_msg.twist.twist.angular.y = 0
            odom_msg.twist.twist.angular.z = angular_velocity_z

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
