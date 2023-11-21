#!/usr/bin/python

import rosbag
import rospy
import cv2
import sys
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from matplotlib import pyplot as plt

def compute_distance(bag_file, interval=0.1, debug=False):
    record_positions = []
    last_time = None
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic == "/odom":
                if last_time is None:
                    last_time = t.to_sec()
                elif t.to_sec() - last_time > interval:
                    record_positions.append((msg.pose.pose.position.x, msg.pose.pose.position.y))
                    last_time = t.to_sec()
    
    if debug:
        plt.plot([x for x, y in record_positions], [y for x, y in record_positions])
        plt.show()
        plt.close()
        
    distance = 0
    for i in range(1, len(record_positions)):
        x0, y0 = record_positions[i-1]
        x1, y1 = record_positions[i]
        distance += ((x1 - x0)**2 + (y1 - y0)**2)**0.5
    return distance

if __name__ == '__main__':
    rospy.init_node('distance_extractor')

    # Read command line arguments
    if len(sys.argv) < 2:
        rospy.logerr("Usage: python get_res_from_bag.py <bag_file>")
        sys.exit(1)

    bag_file = sys.argv[1]

    try:
        distance = compute_distance(bag_file)
        rospy.loginfo("Distance: %f" % distance)
    except rospy.ROSInterruptException:
        pass
