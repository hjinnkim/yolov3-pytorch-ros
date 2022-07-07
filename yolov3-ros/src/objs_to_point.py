#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import ros_numpy

from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud, PointCloud2, ChannelFloat32
from object_msgs.msg import Object, ObjectArray

class yolo_point_node():
    def __init__(self):
        
        rospy.init_node("yolo_point_node")

        self.indexsub = rospy.Subscriber('/yolov3_detect/objects', ObjectArray, self.objCb, queue_size=5)
        # self.img_sub = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.rgbCb, queue_size=5)
        self.depthsub = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.pointCb, queue_size=5)
        # self.distancepub = rospy.Publisher('/yolov3_detect/distances', ObjectDistanceArray, queue_size=5)
        self.pointpub = rospy.Publisher('/yolov3_detect/points', PointCloud, queue_size=5)

        self.objs = []


    def objCb(self, objmsg):
        self.objs = [[obj.class_name, (int(obj.xmin_ymin_xmax_ymax[1]+obj.xmin_ymin_xmax_ymax[3])/2, int(obj.xmin_ymin_xmax_ymax[0]+obj.xmin_ymin_xmax_ymax[2])/2)] for obj in objmsg.Objects]
        

    def pointCb(self, pcl2msg):
        assert isinstance(pcl2msg, PointCloud2)

        pcl_dummy = PointCloud()
        pcl = PointCloud()
        pcl.header = pcl2msg.header

        width = pcl2msg.width
        # pclList = point_cloud2.read_points_list(pcl2msg, field_names=('x', 'y', 'z'))
        pclnp = ros_numpy.numpify(pcl2msg)

        # for obj in self.objs:
        #     Point = pclList[width*obj[1][0]+obj[1][1]]
        #     point = Point32(x=Point.x, y=Point.y, z=Point.z)
        #     # point.x = Point.x
        #     # point.y = Point.y
        #     # point.z = Point.z
        #     pcl.points.append(point)
        #     pcl.channels.append(ChannelFloat32(name=obj[0]))

        for obj in self.objs:
            Point = pclnp[obj[1][0], obj[1][1]]
            point = Point32(x=Point['x'], y=Point['y'], z=Point['z'])
            pcl.points.append(point)
            pcl.channels.append(ChannelFloat32(name=obj[0]))
        
        # self.pointpub.publish(pcl_dummy)
        self.pointpub.publish(pcl)

if __name__=='__main__':
    test_node = yolo_point_node()
    rospy.spin()