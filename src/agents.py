from openai import OpenAI
from paths import PROMPT_DIR
import base64
import json

client = OpenAI()
p = PROMPT_DIR+"visual_agent_prompt.txt"

class Agent:
    def __init__(self,image,task_description):
        self.encoded_image = image
        self.task_description = task_description
    
    def vlm_call(self, prompt):
        agent = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                    "text":f"{prompt}"}, #TODO
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encoded_image}",
                },
                },
            ],
            }
        ],
        temperature=0.1,
        )
        response = (agent.choices[0].message.content)
        return response
    
image_path = "/home/semanticnuc/Pictures/Screenshots/rgb.jpg"
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")
p = open(p,"r").read()
ag = Agent(encoded_image, "")
json_answer = (ag.vlm_call(p))
json_answer = json_answer.replace("```json", "").replace("```","")
#print(json_answer)
json_stucture = json.loads(json_answer)

objects = [[object_name["Label"], object_name["Relations"]] for object_name in json_stucture["Objects"][0:len(json_stucture["Objects"])]]
print(objects)