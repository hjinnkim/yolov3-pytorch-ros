#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
from object_msgs.msg import Object, ObjectArray, ObjectDistance, ObjectDistanceArray
from cv_bridge import CvBridge, CvBridgeError

class yolo_distance_node():
    def __init__(self):
        
        rospy.init_node("yolo_distance_node")

        self.indexsub = rospy.Subscriber('/yolov3_detect/objects', ObjectArray, self.objCb, queue_size=5)
        # self.img_sub = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.rgbCb, queue_size=5)
        self.depthsub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depthCb, queue_size=5)
        self.distancepub = rospy.Publisher('/yolov3_detect/distances', ObjectDistanceArray, queue_size=5)

        self.bridge = CvBridge()

        self.objs = []
        self.distances = []


    def objCb(self, objmsg):
        self.objs = [[obj.class_name, (int(obj.xmin_ymin_xmax_ymax[1]+obj.xmin_ymin_xmax_ymax[3])/2, int(obj.xmin_ymin_xmax_ymax[0]+obj.xmin_ymin_xmax_ymax[2])/2)] for obj in objmsg.Objects]
        

    # def rgbCb(self, rgbmsg):
    #     try:
    #         img = self.bridge.compressed_imgmsg_to_cv2(rgbmsg)
    #         for obj in self.objs:
    #             cv2.circle(img, obj[1], radius=5, color=(0, 0, 255), thickness=-1)
    #         cv2.imshow('test', img)
    #         cv2.waitKey(1)

    #     except CvBridgeError as e:
    #         print(e)

    def depthCb(self, depthmsg):
        try:
            # img = self.bridge.compressed_imgmsg_to_cv2(depthmsg)
            img = self.bridge.imgmsg_to_cv2(depthmsg, desired_encoding='passthrough')

            del self.distances[:]

            for obj in self.objs:
                object = ObjectDistance()
                object.class_name = obj[0]
                object.distance = img.item(obj[1])
                self.distances.append(object)

            self.distancepub.publish(self.distances)

            
        except CvBridgeError as e:
            print(e)
    
if __name__=='__main__':
    test_node = yolo_distance_node()
    rospy.spin()