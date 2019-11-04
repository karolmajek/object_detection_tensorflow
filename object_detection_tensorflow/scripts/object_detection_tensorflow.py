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
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

from object_detection_tensorflow_msgs.msg import BBox, BBoxArray

#Disable Tensroflow log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SKIP_FRAMES = 0

class ObjectDetectionTensorflow:
    def __init__(self):
        self.seq = 0
        self.ready = False
        self.counter = 0
        self.bridge = CvBridge()

        self.camera_topic = rospy.get_param('~camera_topic',
                                            "/image_raw")
        self.image_sub = rospy.Subscriber(self.camera_topic, Image,
                                          self.callback, queue_size=1)
        self.render = rospy.get_param('~render', True)
        if self.render:
            self.image_pub = rospy.Publisher("detections/image_raw/compressed", CompressedImage, queue_size=5)
        self.model_name = rospy.get_param('~model_name')
        self.models_dir = rospy.get_param('~models_dir')
        self.path_to_ckpt = self.models_dir + '/' + self.model_name + '/frozen_inference_graph.pb'
        self.path_to_labels =rospy.get_param('~path_to_labels')
        self.num_classes = rospy.get_param('~num_classes', 90)
        self.threshold = rospy.get_param('~threshold', 0.5)
        self.rotate = rospy.get_param('~rotate', False)
        self.debug = rospy.get_param('~debug', False)
        self.bbox_pub = rospy.Publisher(self.camera_topic+"/detections", BBoxArray, queue_size=5)

        print("path_to_ckpt:",self.path_to_ckpt)
        print("path_to_labels:",self.path_to_labels)

        if self.path_to_ckpt == '' or self.path_to_labels == '':
            print("\n\nProvide requiered args: path_to_ckpt, path_to_labels")
            print("Shutting down.")
            exit(-1)

        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
         max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        print("Category map loaded:")
        for i,n in zip (self.category_index.keys(),[str(_['name']) for _ in self.category_index.values()]):
            print("%4d %s"%(i,n))

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            print("Loading model")

            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                print("Parsing")
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=self.detection_graph)

            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            print("Outputs:")
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    print("  "+key)
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        self.ready = True
        print("Model loaded. Waiting for messages on topic:",self.camera_topic)

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
            'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks_reframed' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks_reframed'][0]
            return output_dict

    def callback(self, data):
        if not self.ready:
            # print("Received image, but not yet ready, still loading...")
            return
        if self.counter < SKIP_FRAMES:
            self.counter = self.counter + 1
            print("Skipping %d/%d" % (self.counter, SKIP_FRAMES))
            return
        else:
            self.counter = 0
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            if self.rotate:
                #Rotate90
                image = np.transpose(image, (1, 0, 2))
                image = image[::-1,:,:]


            output_dict = self.run_inference_for_single_image(image)

            bboxes=BBoxArray()
            bboxes.camera_topic = self.camera_topic
            bboxes.header.stamp = data.header.stamp
            bboxes.header.frame_id = data.header.frame_id
            debug_list=[]
            for box, cl, score in zip(output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores']):
                if score>=self.threshold:
                    b = BBox()
                    b.header.seq = self.seq
                    self.seq = self.seq+1
                    b.header.stamp = data.header.stamp
                    b.header.frame_id = data.header.frame_id
                    b.xmin = box[0]
                    b.ymin = box[1]
                    b.xmax = box[2]
                    b.ymax = box[3]
                    b.score = score
                    b.name = str(self.category_index[cl]['name'])
                    b.id = cl
                    b.camera_topic = self.camera_topic
                    debug_list.append(self.category_index[cl]['name']+" (%.1f) "%(score*100))
                    # print(box,cl,score,self.category_index[cl]['name'])
                    bboxes.bboxes.append(b)
            self.bbox_pub.publish(bboxes)
            if self.debug:
                print(' '.join(sorted(debug_list)))
            if self.render:
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                  image,
                  output_dict['detection_boxes'],
                  output_dict['detection_classes'],
                  output_dict['detection_scores'],
                  self.category_index,
                  instance_masks=output_dict.get('detection_masks'),
                  min_score_thresh=self.threshold,
                  use_normalized_coordinates=True,
                  line_thickness=8)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"
                msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
                self.image_pub.publish(msg)
        except CvBridgeError as e:
            print(e)


def main(args):
    rospy.init_node('object_detection_tensorflow', anonymous=True)
    ic = ObjectDetectionTensorflow()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
