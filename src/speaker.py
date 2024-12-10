#! /usr/bin/env python

from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from std_msgs.msg import String
import rospy


def say_phrase(data):
    client = SimpleActionClient('/tts', TtsAction)
    client.wait_for_server()
    goal = TtsGoal()
    goal.rawtext.text = data.data
    goal.rawtext.lang_id = "en_GB"
    client.send_goal_and_wait(goal)


def listener():
    rospy.init_node('speaker', anonymous=True)
    rospy.Subscriber('/text_topic',String, say_phrase)
    rospy.spin()

if __name__ == "__main__":
    listener()