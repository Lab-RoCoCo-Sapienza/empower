#! /usr/bin/env python

# Standard Library Imports
import io
import os
import time
import json
import re
from scipy.io.wavfile import write
import networkx as nx
import matplotlib.pyplot as plt

# ROS and ROS messages
import rospy
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal

# OpenAI and Audio Processing Libraries
from openai import OpenAI
import librosa
import soundfile as sf

# Computer Vision and Point Cloud Processing
import cv2
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import numpy as np

# Custom Paths and Agents
from agents import Agent
from paths import PROMPT_DIR, OUTPUT_DIR

visual_prompt = PROMPT_DIR+"visual_agent_prompt.txt"
conversational_prompt_path = PROMPT_DIR+"conversational_prompt.txt"
planner_prompt_path = PROMPT_DIR+"planner_prompt.txt"

client = OpenAI()
rospy.init_node('audio_process', anonymous=True)

agents = Agent()

tasks = {
    "task_1" : "Pour the water into the cup",
    "task_2" : "Throw away the objects",
    "task_3" : "Put the ball under the table"
}

task_id = "task_2"

#Set log stuff
save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), task_id , time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)
point_cloud_path = os.path.join(save_folder,"point_cloud.pcd")
img_path = os.path.join(save_folder, "image.png")
text_path = os.path.join(save_folder, "logs.txt")

audio_path = os.path.join(save_folder, "whisper_audio.wav")
log_file = open(text_path, "w")

# Save RGB image
bridge = CvBridge()
msg_img = rospy.wait_for_message("/xtion/rgb/image_rect_color", Image)
img = bridge.imgmsg_to_cv2(msg_img, "bgr8")
cv2.imwrite(img_path, img)

#TODO: move in utils.py
def depth_image_to_point_cloud(depth_image, camera_intrinsics):
    height, width = depth_image.shape
    points = []

    v, u =  np.indices((height, width))

    x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
    z = depth_image

    points = np.dstack((x, y, z)).reshape(-1, 3)

    return points

def capture_pcd():    
    msg_img_g = rospy.wait_for_message("/xtion/depth/image_raw", Image)
    camera_info = rospy.wait_for_message("/xtion/depth/camera_info", CameraInfo)
    proj_matrix = camera_info.K  
    fx = proj_matrix[0]
    fy = proj_matrix[4]
    cx = proj_matrix[2]
    cy = proj_matrix[5]

    camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    img_g = bridge.imgmsg_to_cv2(msg_img_g)
    depth_image = np.asarray(img_g)
    point_cloud = depth_image_to_point_cloud(depth_image, camera_intrinsics)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    pcd.transform(np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))
    o3d.io.write_point_cloud(point_cloud_path, pcd)

def create_graph(json_structure, save_path):
    G = nx.Graph()
    for obj in json_structure['Objects']:
        label = obj['Label']
        relations = obj['Relations']            
        G.add_node(label)
        
        for relation in relations:
            relation_attribute, label2 = relation.split(";")[0], relation.split(";")[1]
            G.add_edge(label, label2, relation=relation_attribute)

    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Aumenta k per maggiore spazio tra i nodi
    
    nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue", font_size=5, font_weight='bold', width=2)
    
    edge_labels = nx.get_edge_attributes(G, 'relation')
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, font_color='blue')

    # Salvataggio del grafico
    plt.savefig(save_path)
    plt.close()


capture_pcd()



def say_phrase(data):
    client = SimpleActionClient('/tts', TtsAction)
    client.wait_for_server()
    goal = TtsGoal()
    goal.rawtext.text = data
    goal.rawtext.lang_id = "en_GB"
    client.send_goal_and_wait(goal)


def listen_for(seconds):
    try:
        seconds = seconds // 3
        rospy.loginfo(f"Received audio message")
        whisper_audio = []
        for _ in range(seconds):
            data = rospy.wait_for_message("/data_topic", Float32MultiArray)
            audio_array = np.array(data.data)
            amplitude_audio = np.abs(audio_array)
            amplitude_audio = np.mean(amplitude_audio)
            rospy.loginfo(f"Amplitude: {amplitude_audio}")
            if np.max(amplitude_audio) < 0.001:
                rospy.loginfo("Audio troppo silenzioso, ignorato.")
            else:
                audio = librosa.resample(audio_array, orig_sr=44100, target_sr=16000).astype(np.float32)
                whisper_audio = np.concatenate((whisper_audio, np.array(audio, dtype=np.float32)))
                
        sf.write(audio_path,whisper_audio, 16000)


        whisper_audio = open(audio_path, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=whisper_audio,
            language='en')
        os.system("rm " + audio_path)
        return transcription.text
    except Exception as e:
        rospy.logerr(f"Error in callback: {e}")

def scene_description(visual_prompt ,img):
    scene = agents.objects_description(visual_prompt, img)
    json_structure = json.loads(scene)    
    with open(os.path.join(save_folder,time.strftime("%Y%m%d-%H%M%S")+ ".json"), 'w') as outfile:
        json.dump(json_structure, outfile,indent=4)
    create_graph(json_structure, os.path.join(save_folder,time.strftime("%Y%m%d-%H%M%S")+ ".png"))
    log_file.write(scene + "\n")
    return json_structure



if __name__ == '__main__':    

    visual_prompt = open(visual_prompt,"r").read()
    json_scene = scene_description(visual_prompt, img)

    conversational_prompt = open(conversational_prompt_path,"r").read()
    conversational_prompt = conversational_prompt.replace("<SCENE_DESCRIPTION>", json_scene["Scene description"])

    planner_prompt = open(planner_prompt_path,"r").read()
    planner_prompt = planner_prompt.replace("<SCENE_DESCRIPTION>", json_scene["Scene description"])

    conversation = "ORIGINAL TASK: " + tasks[task_id] + "\n CONVERSATION:" 

    llm_answer = agents.llm_call(conversational_prompt,conversation)
    log_file.write(llm_answer + "\n")

    while "<QUESTION>" in llm_answer or "<ANSWER>" in llm_answer or not "PLAN_INFO" in llm_answer:
        question = re.search(r"(?<=<ANSWER>)(.*)(?=</ANSWER>)",llm_answer) if re.search(r"(?<=<QUESTION>)(.*)(?=</QUESTION>)",llm_answer) is None \
                                                                        else re.search(r"(?<=<QUESTION>)(.*)(?=</QUESTION>)",llm_answer)
        """
        say_phrase(question[0])
        time.sleep(3)
        human_answer = listen_for(6)
        """
        print(question[0])
        human_answer = input("write answer: ")

        conversation += "\nYOU:" + llm_answer + "\n HUMAN:" + human_answer

        llm_answer = agents.llm_call(conversational_prompt, conversation)

        log_file.write(question[0] + "\n")
        log_file.write(human_answer + "\n")
        log_file.write(conversation + "\n")
        log_file.write(llm_answer + "\n")

    print(llm_answer)
    additional_info = re.search(r"(?<=<PLAN_INFO>)(.*)(?=</PLAN_INFO>)",llm_answer)
    log_file.write(additional_info[0])

    planner_prompt = open(planner_prompt_path,"r").read()
    planner_prompt = planner_prompt.replace("<SCENE_DESCRIPTION>", json_scene["Scene description"])
    planner_prompt = planner_prompt.replace("<PLAN_INFO>", additional_info[0])

    
    visual_prompt += "\nAfter an exploration we discovered also additional information : " + additional_info[0] +"You have to add these information in spatial information"
    json_scene = scene_description(visual_prompt, img)

    plan = agents.llm_call(planner_prompt, tasks[task_id])

    print(plan)
    log_file.write(plan + "\n")
