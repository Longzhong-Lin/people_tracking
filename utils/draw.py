import numpy as np
import cv2
import rospy
from geometry_msgs.msg import Twist


pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

maxr=1.2 #最大角速度 单位 rad/s 
maxv=0.8 #最大线速度,单位 m/s


def control_publisher(vel,angle):
    twist = Twist()
    twist.linear.x = vel*1
    twist.angular.z = angle*1
    pub_vel.publish(twist)



def draw_boxes(img, bbox, identities, Id, history, develop):#img: 视频帧，bbox: 所有的选框坐标构成的数组，identities: 系统识别的ID， Id: 用户选择的id    
    
    count=0
    v=0
    r=0
    if Id==0:#未选择任何目标，停止
        if develop:
            cv2.putText(img,"Stop",(20,100), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,255], 3)
            cv2.putText(img,"Enter the object ID:",(20,125), cv2.FONT_HERSHEY_PLAIN, 2, [255,0,0], 2)
        v=0
        r=0
        control_publisher(v,r)
        history.reset()#清空历史记录
    for i,box in enumerate(bbox):
        id = int(identities[i]) if identities is not None else 0
        speed =0
        rad=0
        if Id == 0 or Id == id:#接下去如Id为0,则标出所有供选择;若Id为目标id，则检测位置。
        
            count+=1
            x1,y1,x2,y2 = [int(j) for j in box]
            
            
            color=[255,128,128]
            if Id!=0:
                center=((x1+x2)//2,(y1+y2)//2)
                
                history.add(center[0])## 只记录X位置到历史中
                
                angle=(center[0]/img.shape[1]-0.5)*2 # range -1 to 1
                rad = -angle*maxr
                
                cv2.circle(img,center,2,[0,0,255],-1)#画出选框中心点
                top=y1/img.shape[0]
                if top<0.05:
                    v=-0.2
                    r=rad
                    control_publisher(v,r)
                    color=[0,0,255]
                    if develop:
                        cv2.putText(img,"Too Close!",(20,100), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,255], 3)
                elif top<0.15:#too close
                    
                    v=0
                    r=rad
                    control_publisher(v,r)
                    color=[0,0,255]
                    if develop:
                        cv2.putText(img,"Too Close!",(20,100), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,255], 3)
                    
                elif top >0.3:#far
                    
                    v=maxv
                    r=rad
                    control_publisher(v,r)
                    color=[0,255,0]
                    if develop:
                        cv2.putText(img,"Far",(20,100), cv2.FONT_HERSHEY_PLAIN, 2, [0,128,0], 3)
                    
                else:
                    v=(top-0.15)/0.15*maxv
                    r=rad
                    control_publisher(v,r)
                    color=[0,128,255]
                    if develop:
                        cv2.putText(img,"Close",(20,100), cv2.FONT_HERSHEY_PLAIN, 2, [0,128,255], 3)
                      
                if develop:
                    cv2.putText(img,"Selected ID {:}".format(Id),(20,125), cv2.FONT_HERSHEY_PLAIN, 2, [255,0,0], 2)
                    cv2.putText(img,"Press \"Enter\" to reset ID",(20,150), cv2.FONT_HERSHEY_PLAIN, 2, [255,0,0], 2)   

            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            
            if develop:
                cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
                cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
                cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    if count==0 and Id != 0:#if无目标被检测出，且处于跟踪状态，then报告目标丢失
        
        if develop:
            cv2.putText(img,"Miss Person".format(Id),(20,100), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,255], 3)
            cv2.putText(img,"Press \"Enter\" to reset ID",(20,125), cv2.FONT_HERSHEY_PLAIN, 2, [255,0,0], 2)
        control_publisher(0,0)
        v=0
        r=0
            #history.add(None)
        
    if develop:
        cv2.putText(img,"{:.02f} m/s".format(v),(0,250), cv2.FONT_HERSHEY_PLAIN, 2, [255,0,0], 3)
        cv2.putText(img,"{:.02f} rad/s".format(r),(0,300), cv2.FONT_HERSHEY_PLAIN, 2, [255,0,0], 3)    
        #cv2.putText(img,"{:.02f} m/s".format(v/1000),(0,250), cv2.FONT_HERSHEY_PLAIN, 2, [255,0,0], 3)
        #cv2.putText(img,"{:.02f} rad/s".format(r/1000),(0,300), cv2.FONT_HERSHEY_PLAIN, 2, [255,0,0], 3)
    return img
