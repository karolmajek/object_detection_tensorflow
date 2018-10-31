cmake_minimum_required(VERSION 2.8.3)
project(object_detection_tensorflow_viz)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  std_msgs
  object_detection_tensorflow_msgs)

catkin_package(
  CATKIN_DEPENDS
    std_msgs
    object_detection_tensorflow_msgs
)
