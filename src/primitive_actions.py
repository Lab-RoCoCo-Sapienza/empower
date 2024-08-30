
import rospy
import open3d as o3d
import os
import numpy as np
import copy
import math
import time
from math import pi
import rospy
import sys
from geometry_msgs.msg import Pose, Quaternion
from geometry_msgs.msg import Twist
import moveit_commander
import moveit_msgs.msg
import tf2_ros
import tf2_py as tf2
import pickle
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

move_base_publisher = rospy.Publisher('/mobile_base_controller/cmd_vel')

# GRABBING_QUATERNION =  Quaternion(-0.528,0.412,0.473,0.571)
GRABBING_QUATERNION = Quaternion(-0.742818371853,-0.0634568007104,0.0418472405416,0.665163821431)
GRIPPER_OFFSET = 0.15

def reach_pose(group,pose):
    group.set_pose_target(pose)
    group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

def close_grippers(gripper):
    gripper.go([0.01, 0.02], wait=True)
    gripper.stop()
    gripper.clear_pose_targets()

def reach_waypoints(group,waypoints):
    (plan, fraction) = group.compute_cartesian_path(
                           waypoints,   # waypoints to follow
                           0.01,        # eef_step
                           0.0)         # jump_threshold
    group.execute(plan, wait=True)
    group.stop()
    group.clear_pose_targets()

def open_grippers(gripper):
    gripper.go([0.04, 0.04], wait=True)
    gripper.stop()
    gripper.clear_pose_targets()

def back_init():
    arm_torso = "arm_torso"
    gripper = "gripper"
    arm_group = moveit_commander.MoveGroupCommander(arm_torso)
    gripper_group = moveit_commander.MoveGroupCommander(gripper)

    init = list(np.array([11, -77, -11, 111, -90, 78, 0])*np.pi/180)
    new_wp1 = list(np.array([42, 16, -109, 105, -60, -56, -108])*np.pi/180)
    new_wp1 = [0.35]+new_wp1

    arm_group.go(new_wp1, wait=True)
    arm_group.stop()

    open_grip = [0.04, 0.04]
    gripper_group.go(open_grip, wait=True)
    gripper_group.stop()

def home():
    moveit_commander.roscpp_initialize(sys.argv) 

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "arm_torso"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Move arm to the initial position
    init_pos = [0, 0, 0, 0, 0, 0, 0, 0]
    init_pos[0] = 0.35 #torso height
    init_pos[1:] = [0.2, -1.34, -0.2, 1.94, -1.57, 1.37, 0.0] # arms

    move_group.go(init_pos, wait=True)
    move_group.stop() 
    move_group.clear_pose_targets()                         

    gripper = moveit_commander.MoveGroupCommander("gripper")
    gripper.go([0.001, 0.002], wait=True)
    gripper.stop()
    gripper.clear_pose_targets()

def grab(group, gripper, goal_pose):
    init_pose = group.get_current_pose().pose
    
    wp1 = copy.deepcopy(init_pose)
    wp1.position.z = goal_pose.position.z
    wp1.position.y = goal_pose.position.y

    wp2 = copy.deepcopy(wp1)

    wp2.position.x = goal_pose.position.x

    wp1.orientation = GRABBING_QUATERNION
    wp2.orientation = GRABBING_QUATERNION

    waypoints = [wp1]
    reversed_waypoints = [init_pose]
    
    reach_waypoints(group, waypoints)
    close_grippers(gripper)
    reach_waypoints(group, reversed_waypoints) 

def drop(group, gripper, goal_pose):
    init_pose = group.get_current_pose().pose
    
    wp1 = copy.deepcopy(init_pose)
    wp1.position.z = goal_pose.position.z
    wp1.position.y = goal_pose.position.y
    wp1.position.x = goal_pose.position.x

    wp1.orientation = GRABBING_QUATERNION

    waypoints = [wp1]
    reversed_waypoints = [init_pose]
    
    reach_waypoints(group, waypoints)
    open_grippers(gripper)
    reach_waypoints(group, reversed_waypoints)

def push(group, gripper, goal_pose):
    init_pose = group.get_current_pose().pose
    current_position = init_pose.position
    current_orientation = init_pose.orientation
    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "base_link"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position = current_position
    goal.target_pose.pose.positon.x += 2
    goal.target_pose.pose.orientation = current_orientation

    client.send_goal(goal)
    wait = client.wait_for_result()
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    pass

def navigate(group, gripper, goal_pose):
    init_pose = group.get_current_pose().pose
    current_orientation = init_pose.orientation
    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = goal_pose.position.x-0.3
    goal.target_pose.pose.position.y = goal_pose.position.y-0.3
    goal.target_pose.pose.position.z = goal_pose.position.z
    goal.target_pose.pose.orientation = current_orientation

    client.send_goal(goal)
    wait = client.wait_for_result()
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")

def pull(group, gripper, goal_pose):
    init_pose = group.get_current_pose().pose
    
    wp1 = copy.deepcopy(init_pose)
    wp1.position.z = goal_pose.position.z
    wp1.position.y = goal_pose.position.y

    wp2 = copy.deepcopy(wp1)

    wp2.position.x = goal_pose.position.x

    wp1.orientation = GRABBING_QUATERNION
    wp2.orientation = GRABBING_QUATERNION

    waypoints = [wp1]
    
    reach_waypoints(group, waypoints)
    close_grippers(gripper)

    current_orientation = init_pose.orientation
    current_position = init_pose.position
    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "base_link"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position = current_position
    goal.target_pose.pose.positon.x -= 0.4
    goal.target_pose.pose.orientation = current_orientation

    client.send_goal(goal)
    wait = client.wait_for_result()
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")