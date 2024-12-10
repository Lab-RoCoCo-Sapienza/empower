#! /usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import os
import librosa
from openai import OpenAI
import soundfile as sf

script_path = os.path.dirname(os.path.realpath(__file__))

client = OpenAI()


def callback(data):
    """Callback per gestire i dati audio ricevuti."""
    try:
        rospy.loginfo(f"Received audio message")
        whisper_audio = []
        for i in range(3):

            audio_array = np.array(data.data)
            amplitude_audio = np.abs(audio_array)
            amplitude_audio = np.mean(amplitude_audio)  
            rospy.loginfo(f"amplitude: {amplitude_audio}")
            if np.max(amplitude_audio) < 0.001:
                rospy.loginfo("Audio troppo silenzioso, ignorato.")
                return            
            audio = librosa.resample(audio_array, orig_sr=44100, target_sr=16000).astype(np.float32)
            whisper_audio = np.concatenate((whisper_audio, np.array(audio, dtype=np.float32)))
        print(whisper_audio)
       
        sf.write(os.path.join(script_path,"whisper_audio.wav"), whisper_audio, 16000)

        whisper_audio = open(os.path.join(script_path,"whisper_audio.wav"), "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=whisper_audio,
            language='en')
        print(transcription.text)       



    except Exception as e:
        rospy.logerr(f"Error in callback: {e}")



def consumer():
    """Nodo ROS per il consumer."""
    rospy.init_node('consumer', anonymous=True)
    rospy.Subscriber('data_topic', Float32MultiArray, callback)
    rospy.spin()  # Mantiene il nodo attivo


if __name__ == '__main__':
    try:
        consumer()
    except rospy.ROSInterruptException:
        pass


'''

scene = Agent().objects_description(img)
print(scene)
prompt = open(plan_prompt,"r").read()
prompt = prompt.replace("<SCENE_DESCRIPTION>", scene["Scene description"])
conversation = "ORIGINAL TASK: throw away the objects + \n CONVERSATION:" 
plan = Agent().llm_call(prompt,conversation)
question = re.search(r"(?<=<HELPER>)(.*)(?=</HELPER>)",plan)
llm_answer = re.search(r"(?<=<ANSWER>)(.*)(?=</ANSWER>)",plan)
print(llm_answer)
print(question)
if question != None:
        text_publisher.publish(question[0])
        conversation += "\nYOU:" + question[0] + "\n HUMAN:" + "human_answer"
        plan = Agent().llm_call(prompt, conversation)
elif llm_answer != None:
        text_publisher.publish(llm_answer[0])
        conversation += "\nYOU:" + llm_answer[0] + "\n HUMAN:" + "human_answer"
        plan = Agent().llm_call(prompt, conversation)

text_publisher.publish(plan)

'''