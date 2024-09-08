#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

# 定义比例因子
scale_v = 1.0  # 线速度比例因子
scale_omega = 1.5  # 角速度比例因子

# 回调函数，当订阅到 /robot_3/cmd_vel_dwa 消息时调用
def cmd_vel_dwa_callback(msg):
    # 创建一个新的Twist消息对象
    modified_msg = Twist()

    # 对linear.x和angular.z分别乘以系数
    modified_msg.linear.x = msg.linear.x * scale_v
    modified_msg.angular.z = msg.angular.z * scale_omega

    # 打印变换前后的v和omega
    print(f"v: {msg.linear.x:.2f} -> {modified_msg.linear.x:.2f} |", \
          f"omega: {msg.angular.z:.2f} -> {modified_msg.angular.z:.2f}")

    # 发布新的消息到 /robot_3/cmd_vel
    cmd_vel_pub.publish(modified_msg)

# 主函数
if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('cmd_vel_scaler')

    # 创建发布者，发布到 /robot_3/cmd_vel 话题
    cmd_vel_pub = rospy.Publisher('/robot_3/cmd_vel', Twist, queue_size=10)

    # 创建订阅者，订阅 /robot_3/cmd_vel_dwa 话题
    rospy.Subscriber('/robot_3/cmd_vel_dwa', Twist, cmd_vel_dwa_callback)

    # 保持节点运行，等待回调函数处理
    rospy.spin()
