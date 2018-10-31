#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('object_detection_tensorflow')
import sys
import rospy
import cv2
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import NavSatFix

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
# from tqdm import tqdm

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SKIP_PUBLISH=5

latitude = "0"
longitude = "0"
stamp = "0"

class image_converter:
    def __init__(self):
        with open("/home/robot/detections/opi.csv",'w') as f:
            pass
        self.ready = False
        self.counter = 1000
        self.bridge = CvBridge()

        self.camera_topic = rospy.get_param('~camera_topic',
                                            "/husky/ladybug/preview")
        self.image_sub = rospy.Subscriber(self.camera_topic, Image,
                                          self.callback, queue_size=1,
                                          buff_size=100000000)
        self.subGPS = rospy.Subscriber("/fix", NavSatFix, self.onGPSData, queue_size=10)

        # self.image_pub = rospy.Publisher("detections/ladybug/compressed", CompressedImage, queue_size=5)
        # self.image_pub2 = rospy.Publisher("detections/ladybug/rows/compressed", CompressedImage, queue_size=5)
        self.image_pub_cam0 = rospy.Publisher("detections/ladybug/cam0/compressed", CompressedImage, queue_size=5)
        self.image_pub_cam1 = rospy.Publisher("detections/ladybug/cam1/compressed", CompressedImage, queue_size=5)
        self.image_pub_cam2 = rospy.Publisher("detections/ladybug/cam2/compressed", CompressedImage, queue_size=5)
        self.image_pub_cam3 = rospy.Publisher("detections/ladybug/cam3/compressed", CompressedImage, queue_size=5)
        self.image_pub_cam4 = rospy.Publisher("detections/ladybug/cam4/compressed", CompressedImage, queue_size=5)
        self.image_pub_cam = [self.image_pub_cam0, self.image_pub_cam1,
                              self.image_pub_cam2, self.image_pub_cam3,
                              self.image_pub_cam4]
        self.model_name = rospy.get_param('~model_name')
        self.path_to_ckpt = '/home/robot/erl2018_ws/src/object_detection_tensorflow/models/' + self.model_name + '/frozen_inference_graph.pb'
        self.path_to_labels =rospy.get_param('~path_to_labels', '/home/robot/erl2018_ws/src/object_detection_tensorflow/data/mscoco_label_map.pbtxt')
        self.num_classes = int(rospy.get_param('~num_classes', 90))

        if self.path_to_ckpt == '' or self.path_to_labels == '':
            print("\n\nProvide requiered args: path_to_ckpt, path_to_labels")
            print("path_to_ckpt:",self.path_to_ckpt)
            print("path_to_labels:",self.path_to_labels)
            print("Shutting down.")
            exit(-1)

        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
         max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)





        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=self.detection_graph)

            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()

            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        self.ready = True
        print('ready')

    def onGPSData(self, data):
        global longitude, latitude, stamp
        latitude = data.latitude
        longitude = data.longitude
        stamp = str(int(int(str(data.header.stamp))/1000000))

    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():

            if 'detection_masks' in self.tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                self.tensor_dict['detection_masks_reframed'] = tf.expand_dims(
                detection_masks_reframed, 0)
            # Run inference
            output_dict = self.sess.run(self.tensor_dict,
                 feed_dict={self.image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks_reframed' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks_reframed'][0]
            return output_dict

    def callback(self, data):
        if not self.ready:
            print("not ready!")
            return
        if self.counter < SKIP_PUBLISH:
            self.counter = self.counter + 1
            print("Skipping %d/%d" % (self.counter, SKIP_PUBLISH))
            return
        else:
            self.counter = 0
        try:
            image_np = self.bridge.imgmsg_to_cv2(data, "rgb8")
            print("image:",image_np.shape)


            box_found = False

            results = []
            print(image_np.shape)
            # for i in tqdm(list(range(5))):
            for i in range(5):
                image = image_np[:, 600*i:600*(i+1), :]

                output_dict = self.run_inference_for_single_image(image)


                _box_found = np.any(output_dict['detection_scores']>0.3)

                if _box_found:
                    box_found = True

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                  image,
                  output_dict['detection_boxes'],
                  output_dict['detection_classes'],
                  output_dict['detection_scores'],
                  self.category_index,
                  min_score_thresh=0.3,
                  instance_masks=output_dict.get('detection_masks'),
                  use_normalized_coordinates=True,
                  line_thickness=8)
                results.append(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"
                msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
                self.image_pub_cam[i].publish(msg)

            if stamp!="0" and box_found:
                result = np.concatenate(results, axis=1)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                fname = "/home/robot/detections/%s.jpg"%str(time.time())
                cv2.imwrite(fname, result)
                with open("/home/robot/detections/opi.csv",'a') as f:
                    f.write(fname.split('/')[-1])
                    f.write("\t")
                    f.write(stamp)
                    f.write("\t")
                    f.write(str(latitude))
                    f.write("\t")
                    f.write(str(longitude))
                    f.write("\n")
            # image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            # msg = CompressedImage()
            # msg.header.stamp = rospy.Time.now()
            # msg.format = "jpeg"
            # msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
            # self.image_pub.publish(msg)

            # row1 = np.concatenate(results[0:2], axis=1)
            # empty =np.zeros_like((494, 600, 3))
            # print(empty.shape, results[3].shape, results[4].shape)
            # row2 = np.concatenate([empty, results[3], results[4]], axis=1)
            # print(row1.shape, row2.shape)
            # result = np.concatenate([row1, row2], axis=2)
            # image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            # msg = CompressedImage()
            # msg.header.stamp = rospy.Time.now()
            # msg.format = "jpeg"
            # msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
            # self.image_pub2.publish(msg)


        except CvBridgeError as e:
            print(e)


def main(args):
    rospy.init_node('object_detection_tensorflow', anonymous=True)
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
