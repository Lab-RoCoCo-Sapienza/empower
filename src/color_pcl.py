#!/usr/bin/env python

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from PIL import Image as PILImage

import cPickle as pickle  # For Python 2.x
from open3d_ros_helper import open3d_ros_helper as orh
import open3d as o3d
import cv2
import io
import numpy as np 
import base64
import os
import time
import tf2_ros
import tf2_py as tf2
import rospy

from matplotlib.colors import to_rgb

from paths import *

USE_CASE = rospy.get_param('/use_case')

SCAN_DIR = IMAGES_DIR+USE_CASE+'/'
DUMP_DIR = OUTPUT_DIR+USE_CASE+'/'

COLORS = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']*3

def rgb_to_bgr(rgb_color):
    r, g, b = rgb_color
    return [b, g, r]

def listener():

    pcd = o3d.io.read_point_cloud(DUMP_DIR+"depth_pointcloud.pcd")

    pcd.colors = o3d.utility.Vector3dVector(np.tile(to_rgb('gray'), (len(pcd.points), 1)))

    image = cv2.imread(SCAN_DIR+'scan.jpg')
    h, w, _ = image.shape

    masks = []
    masks_flipped = []
    with open(DUMP_DIR+"detection.pkl", 'rb') as f:
        detections = pickle.load(f)

    for key in detections.keys():
        mask = detections[key]['mask']
        masks.append(mask[:,:,0])
        masks_flipped.append(cv2.flip(mask[:,:,0],1))


    camera_info = rospy.wait_for_message("/xtion/depth/camera_info", CameraInfo)
    proj_matrix = camera_info.K   

    fx = proj_matrix[0]
    fy = proj_matrix[4]
    cx = proj_matrix[2]
    cy = proj_matrix[5]

    colors_dict = {}

    for idx,point in enumerate(pcd.points):
        if point[2] < 0:
            x_ = point[0]
            y_ = point[1]
            z_ = point[2]
            x = int((fx * x_ / z_) + cx)
            y = int((fy * y_ / z_) + cy)

            for id_color, (mask, mask_flipped) in enumerate(zip(masks, masks_flipped)):
                if 0 <= x < w and 0 <= y < h and mask[y, x] != 0:
                    image[y,x] = rgb_to_bgr([int(color*255) for color in to_rgb(COLORS[id_color])])

                
                if 0 <= x < w and 0 <= y < h and mask_flipped[y, x] != 0:
                    if id_color not in colors_dict.keys():
                        colors_dict[id_color] = []
                    pcd.colors[idx] = [int(color*255) for color in to_rgb(COLORS[id_color])]   
                    colors_dict[id_color].append(point)              

    cv2.imwrite(DUMP_DIR+'colored_image.png', image)

    # Uncomment to visualize pcd
    # o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud(DUMP_DIR+'colored_pcl.pcd', pcd)
    with open(DUMP_DIR+'colors_dict.pkl', 'wb') as f:
        pickle.dump(colors_dict, f)
    return



if __name__ == '__main__':
    rospy.init_node('color_pcl')
    
    listener()