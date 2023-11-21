#!/usr/bin/python

import rosbag
import rospy
import cv2
import sys
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageExtractor:
    def __init__(self, bag_file, output_folder, num_images):
        self.bag_file = bag_file
        self.output_folder = output_folder
        self.num_images = num_images
        self.bridge = CvBridge()

    def extract_images(self):
        target_topic = "/img/TRACK/result"
        
        # total number of images
        total_number = 0
        with rosbag.Bag(self.bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages():
                if topic == target_topic:
                    total_number += 1
        rospy.loginfo("Total number of images: {}".format(total_number))
        
        # extract images
        total_img_cnt = 0
        extract_img_cnt = 0
        extract_interval = max(total_number // self.num_images, 1)
        rospy.loginfo("Extract every {} images".format(extract_interval))
        
        with rosbag.Bag(self.bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages():
                if topic == target_topic:
                    total_img_cnt += 1
                    if total_img_cnt % extract_interval == 0:
                        extract_img_cnt += 1                    
                        try:
                            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                            image_name = "img_{:04d}.png".format(extract_img_cnt)
                            image_path = os.path.join(self.output_folder, image_name)
                            cv2.imwrite(image_path, cv_image)
                            rospy.loginfo("Saved image: {}".format(image_path))
                        except CvBridgeError as e:
                            rospy.logerr("Error converting image: {}".format(e))

if __name__ == '__main__':
    rospy.init_node('image_extractor')

    # Read command line arguments
    if len(sys.argv) < 4:
        rospy.logerr("Usage: python get_res_from_bag.py <bag_file> <output_folder> <num_images>")
        sys.exit(1)

    bag_file = sys.argv[1]
    output_folder = sys.argv[2]
    num_images = int(sys.argv[3])
    
    os.makedirs(output_folder, exist_ok=True)

    try:
        image_extractor = ImageExtractor(bag_file, output_folder, num_images)
        image_extractor.extract_images()
    except rospy.ROSInterruptException:
        pass

