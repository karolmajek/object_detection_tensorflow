#!/usr/bin/env python
from __future__ import print_function
import roslib
# roslib.load_manifest('object_detection_tensorflow')
import sys
import rospy
import cv2
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from object_detection_tensorflow_msgs.msg import BBox, BBoxArray
from collections import deque
import PIL.ImageColor as ImageColor

#Disable Tensroflow log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SKIP_FRAMES = 0
MAX_DEQUE_SIZE=30

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]
def class2color(cl):
    cl = cl%len(STANDARD_COLORS)
    color = STANDARD_COLORS[cl]
    return ImageColor.getrgb(color)

class object_detection_tensorflow_viz:
    def __init__(self):
        self.images_deque = deque()
        self.seq = 0
        self.counter = 0
        self.bridge = CvBridge()

        self.image_sub = None
        self.image_pub = rospy.Publisher("detections/image_raw/compressed", CompressedImage, queue_size=5)
        self.threshold = rospy.get_param('~threshold', 0.5)
        self.rotate = rospy.get_param('~rotate', False)
        self.debug = rospy.get_param('~debug', False)

        self.detections_topic = rospy.get_param('~detections_topic', "/detections")
        self.detections_sub = rospy.Subscriber(self.detections_topic, BBoxArray, self.detections_callback, queue_size=100)

    def image_callback(self, msg):
        if self.counter < SKIP_FRAMES:
            self.counter = self.counter + 1
            print("Skipping %d/%d" % (self.counter, SKIP_FRAMES))
            return
        else:
            self.counter = 0
        self.images_deque.append(msg)
        if len(self.images_deque)>MAX_DEQUE_SIZE:
            self.images_deque.popleft()
    def detections_callback(self, msg):
        if self.image_sub is None:
            print("Subscribing to:",msg.camera_topic)
            self.image_sub = rospy.Subscriber(msg.camera_topic, Image, self.image_callback, queue_size=1)

        img_msg = None
        for i,_img_msg in enumerate(self.images_deque):
            if _img_msg.header.stamp==msg.header.stamp:
                img_msg = _img_msg
                break
        if img_msg is not None:
            try:
                image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
                if self.rotate:
                    #Rotate90
                    image = np.transpose(image, (1, 0, 2))
                    image = image[::-1,:,:]

                for bbox in msg.bboxes:
                    h,w,_ = image.shape
                    cv2.rectangle(image,(int(bbox.ymin*w),int(bbox.xmin*h)),
                                  (int(bbox.ymax*w),int(bbox.xmax*h)),
                                  class2color(bbox.id),2)
                    cv2.putText(image, "%s %.1f%%"%(bbox.name,100.0*bbox.score),
                                (int(bbox.ymin*w),int(bbox.xmin*h)+20), cv2.FONT_HERSHEY_DUPLEX, 0.7,class2color(bbox.id))

                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"
                msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
                self.image_pub.publish(msg)
            except CvBridgeError as e:
                print(e)

def main(args):
    rospy.init_node('object_detection_tensorflow_viz', anonymous=True)
    ic = object_detection_tensorflow_viz()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
