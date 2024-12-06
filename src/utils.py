from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal

def say_phrase(phrase):
    client = SimpleActionClient('/tts', TtsAction)
    client.wait_for_server()
    goal = TtsGoal()
    goal.rawtext.text = phrase
    goal.rawtext.lang_id = "en_GB"
    client.send_goal_and_wait(goal)