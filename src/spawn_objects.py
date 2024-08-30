#!/usr/bin/env python

import rospy
import copy
import open3d as o3d
import os
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import numpy as np
import pickle
from matplotlib.colors import to_rgb
import tf2_ros
from conversion_utils import convertCloudFromOpen3dToRos


ROOT_DIR = os.path.abspath(__file__+'/../..')
SCAN_DIR = ROOT_DIR+'/images/test_order/'

OUTPUT_DIR = ROOT_DIR+'/output/recycle/'

COLORS = ['red', 'green', 'blue', 'magenta', 'cyan','yellow', 'green', 'cyan', 'magenta', 'cyan']

publisher_centroid = rospy.Publisher("/pcl_centroids", MarkerArray, queue_size=100)
publisher_maximum = rospy.Publisher("/pcl_maximum", MarkerArray, queue_size=100)
publisher_names = rospy.Publisher("/pcl_names", MarkerArray, queue_size=100)

def get_R_and_T(trans):
    Tx_base = trans.transform.translation.x
    Ty_base = trans.transform.translation.y
    Tz_base = trans.transform.translation.z
    T = np.array([Tx_base, Ty_base, Tz_base])
    # Quaternion coordinates
    qx = trans.transform.rotation.x
    qy = trans.transform.rotation.y
    qz = trans.transform.rotation.z
    qw = trans.transform.rotation.w
    
    # Rotation matrix
    R = 2*np.array([[pow(qw,2) + pow(qx,2) - 0.5, qx*qy-qw*qz, qw*qy+qx*qz],[qw*qz+qx*qy, pow(qw,2) + pow(qy,2) - 0.5, qy*qz-qw*qx],[qx*qz-qw*qy, qw*qx+qy*qz, pow(qw,2) + pow(qz,2) - 0.5]])
    return R, T

def set_marker(point, color, id, R, T):

    point = point/1000
    transform = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    point = np.dot(transform, point) #in xtion
    R = np.transpose(R)
    point = np.dot(R, point-T) #in map

    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time(0)
    marker.pose.position.x = point[0]
    marker.pose.position.y = point[1]
    marker.pose.position.z = point[2]
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.type = marker.SPHERE
    color = to_rgb(color)
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0
    marker.scale.x = 0.01
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    marker.id = id
    
    return marker

def set_names(marker_, color, id, label, R, T):
    marker_ = marker_/1000
    transform = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    marker_ = np.dot(transform, marker_) #in xtion
    R = np.transpose(R)
    marker_ = np.dot(R, marker_-T) #in map

    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time(0)
    marker.pose.position.x = marker_[0]
    marker.pose.position.y = marker_[1]
    marker.pose.position.z = marker_[2]
    color = to_rgb(color)
    marker.pose.position.z += 0.1
    marker.type = marker.TEXT_VIEW_FACING
    marker.text = label
    marker.scale.z = 0.1
    marker.id = id

    return marker
    

def listener():
    
    # pcd = o3d.io.read_point_cloud(CONFIG_DIR+'colored_pcl.pcd')

    with open(OUTPUT_DIR + "detection.pkl", 'rb') as f:
        detections = pickle.load(f)


    trans = tf_buffer.lookup_transform("xtion_rgb_optical_frame", "map",  rospy.Time(0), rospy.Duration(2.0))
    Rx2m, Tx2m = get_R_and_T(trans)
    # pcd_converted = convertCloudFromOpen3dToRos(pcd, 'xtion_rgb_optical_frame')

    with open(OUTPUT_DIR + 'colors_dict.pkl', 'rb') as f:
        color_dict = pickle.load(f)

    array_centroids = MarkerArray()
    array_maximum = MarkerArray()
    array_name = MarkerArray()

    id = 0
    id_label = 0
    for id_color, list_points in color_dict.items():
        label = detections[id_label]['label']
        for id_point, point in enumerate(list_points):
            if id_point % 20 == 0 and  "yellow" not in COLORS[id_color]:
                new_marker = set_marker(point, COLORS[id_color], id, Rx2m, Tx2m)
                array_centroids.markers.append(new_marker)
                id += 1

        centroid = np.mean(list_points, axis=0)
        new_list = []
        for point in list_points:
            if np.linalg.norm(point - centroid) < 100:
                new_list.append(point)
        
        if new_list == []:
            new_centroid = centroid
            new_list = list_points
        else:
            new_centroid = np.mean(new_list, axis=0)

        label_pos = new_centroid

        label = detections[id_label]['label']
        label = label.encode('utf-8')
        array_name.markers.append(set_names(label_pos, COLORS[id_color], id_label, label, Rx2m, Tx2m))
        id_label+=1


    while not rospy.is_shutdown():
        publisher_centroid.publish(array_name)
        publisher_centroid.publish(array_centroids)

        rate.sleep()


if __name__ == '__main__':

    rospy.init_node('spawn_clusters_points', anonymous=True)
    rate=rospy.Rate(10)

    
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    listener()