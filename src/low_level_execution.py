#!/usr/bin/env python

import rospy
import open3d as o3d
import os
import numpy as np
import copy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from actionlib import SimpleActionClient
import math
from math import pi
import rospy
import sys
from geometry_msgs.msg import Pose, Quaternion
import moveit_commander
import moveit_msgs.msg
import tf2_ros
import tf2_py as tf2
import pickle
from primitive_actions import *
from paths import *

USE_CASE = rospy.get_param('/use_case')
SPEECH = rospy.get_param('/speech')

DUMP_DIR = OUTPUT_DIR + USE_CASE +'/'

rospy.init_node('listener', anonymous=True)
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

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

def say_phrase(phrase):
    client = SimpleActionClient('/tts', TtsAction)
    client.wait_for_server()
    goal = TtsGoal()
    goal.rawtext.text = phrase
    goal.rawtext.lang_id = "en_GB"
    client.send_goal_and_wait(goal)

def extract_labels_per_step(plan):

    labels = []
    step = step.split(' ')[1:]
    labels.append(step[0])
    for i in range(len(step)-1):
        labels.append(step[i]+' '+step[i+1])
    labels.append(step[-1])

    return labels

def apply_direction_to_goal(goal_pose, direction):

    if direction == 'right':
        goal_pose.pose.position.y += 0.3
    elif direction == 'left':
        goal_pose.pose.position.y -= 0.3
    elif direction == 'up':
        goal_pose.pose.position.z += 0.3

    return goal_pose

def listener():
    centroid_list = rospy.wait_for_message("/pcl_centroids", MarkerArray)
    name_list = rospy.wait_for_message("/pcl_names", MarkerArray)

    trans_base = tf_buffer.lookup_transform("map", "base_footprint",  rospy.Time(0), rospy.Duration(2.0))
    trans_map = tf_buffer.lookup_transform("base_footprint", "map",  rospy.Time(0), rospy.Duration(2.0))
    R_m2b, T_m2b = get_R_and_T(trans_base)
    R_b2m, T_b2m = get_R_and_T(trans_map)

    with open(DUMP_DIR+'planning.txt', 'r') as f:
        plan = f.read()
        
    moveit_commander.roscpp_initialize(sys.argv) 
    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    arm_torso_group = moveit_commander.MoveGroupCommander("arm_torso")
    arm_group = moveit_commander.MoveGroupCommander("arm")
    gripper = moveit_commander.MoveGroupCommander("gripper")

    objects_dict = {}

    plan = plan.split('\n')
    if SPEECH:
        say_phrase("I have found a plan")
    for name, marker in zip(name_list.markers, centroid_list.markers):
        objects_dict[name.text] = copy.deepcopy(marker)

    for key, value in objects_dict.items():
        object_array = np.array([value.pose.position.x, value.pose.position.y, value.pose.position.z])
        object_array = np.dot(np.transpose(R_m2b), object_array-T_m2b)
        value.pose.position.x = object_array[0]
        value.pose.position.y = object_array[1]
        value.pose.position.z = object_array[2]
        objects_dict[key] = value

    back_init()

    for step in plan:
        if SPEECH:
            say_phrase(step)
        command = step.split(' ')[0]
        labels = extract_labels_per_step(step)
        
        if command == 'GRAB':
            for label in labels:
                if label in objects_dict.keys():
                    grab(arm_torso_group, gripper, objects_dict[label])
                    break
        
        elif command == 'DROP':
            goal = None
            for label in labels:
                if label in objects_dict.keys():
                    goal = objects_dict[label]
                    break

            for direction in ['right', 'left', 'on']:
                if direction in labels:
                    goal = apply_direction_to_goal(goal, direction)
                    break

            drop(arm_torso_group, gripper, goal)
        
        elif command == 'PUSH':
            for label in labels:
                if label in objects_dict.keys():
                    push(arm_torso_group, gripper, objects_dict[label])
                    break
        
        elif command == 'PULL':
            for label in labels:
                if label in objects_dict.keys():
                    pull(arm_torso_group, gripper, objects_dict[label])
                    break
        
        elif command == 'NAVIGATE':
            goal = None
            for label in labels:
                if label in objects_dict.keys():
                    goal = objects_dict[label]
                    break
            object_array = np.array([goal.pose.position.x, goal.pose.position.y, goal.pose.position.z])
            object_array = np.dot(np.transpose(R_b2m), object_array-T_b2m)
            goal.pose.position.x = object_array[0]
            goal.pose.position.y = object_array[1]
            goal.pose.position.z = object_array[2]
            navigate(arm_torso_group, gripper, goal)
        
        back_init()



if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener()

