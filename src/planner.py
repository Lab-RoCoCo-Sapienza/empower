#! /usr/bin/env python

from agents import Agent
import rospy
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image, CameraInfo

import numpy as np
import os
import librosa
from openai import OpenAI
import soundfile as sf
from openai import OpenAI
from paths import PROMPT_DIR, OUTPUT_DIR
import os
from cv_bridge import CvBridge
import re
from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal

visual_prompt = PROMPT_DIR+"visual_agent_prompt.txt"
plan_prompt = PROMPT_DIR+"planner_hri_agent_prompt.txt"

client = OpenAI()
script_path = os.path.dirname(os.path.realpath(__file__))
rospy.init_node('audio_process', anonymous=True)


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
        for i in range(seconds):
            data = rospy.wait_for_message("/data_topic", Float32MultiArray)
            audio_array = np.array(data.data)
            amplitude_audio = np.abs(audio_array)
            amplitude_audio = np.mean(amplitude_audio)
            rospy.loginfo(f"Amplitude: {amplitude_audio}")
            if np.max(amplitude_audio) < 0.001:
                rospy.loginfo("Audio troppo silenzioso, ignorato.")
            audio = librosa.resample(audio_array, orig_sr=44100, target_sr=16000).astype(np.float32)
            whisper_audio = np.concatenate((whisper_audio, np.array(audio, dtype=np.float32)))
        audio_path = os.path.join(script_path, "whisper_audio.wav")
        sf.write(audio_path, whisper_audio, 16000)

        whisper_audio = open(audio_path, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=whisper_audio,
            language='en')
        print(transcription.text)
        return transcription.text
    except Exception as e:
        rospy.logerr(f"Error in callback: {e}")

bridge = CvBridge()

msg_img = rospy.wait_for_message("/xtion/rgb/image_rect_color", Image)
text_publisher = rospy.Publisher("/text_topic",String, queue_size = 10)
img = bridge.imgmsg_to_cv2(msg_img, "bgr8")

if __name__ == '__main__':
    scene = Agent().objects_description(img)
    print(scene)
    prompt = open(plan_prompt,"r").read()
    prompt = prompt.replace("<SCENE_DESCRIPTION>", scene["Scene description"])
    conversation = "ORIGINAL TASK: throw away the objects + \n CONVERSATION:" 
    plan = Agent().llm_call(prompt,conversation)
    while "<HUMAN>" in plan or "<HELPER>" in plan or "<ANSWER>" in plan:
        question = re.search(r"(?<=<HELPER>)(.*)(?=</HELPER>)",plan)
        llm_answer = re.search(r"(?<=<ANSWER>)(.*)(?=</ANSWER>)",plan)
        print(llm_answer)
        print(question)
        if question != None:
                say_phrase(question[0])

                human_answer = listen_for(6)
                print(human_answer)

                conversation += "\nYOU:" + question[0] + "\n HUMAN:" + human_answer
                plan = Agent().llm_call(prompt, conversation)
        elif llm_answer != None:
                say_phrase(llm_answer[0])

                human_answer = listen_for(6)
                print(human_answer)
                conversation += "\nYOU:" + llm_answer[0] + "\n HUMAN:" + human_answer
                plan = Agent().llm_call(prompt, conversation)
    print(plan)
    text_publisher.publish(plan)
