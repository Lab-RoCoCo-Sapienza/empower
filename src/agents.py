from openai import OpenAI
from paths import PROMPT_DIR, OUTPUT_DIR
import base64
import cv2
import os

visual_prompt = PROMPT_DIR+"visual_agent_prompt.txt"
plan_prompt = PROMPT_DIR+"planner_hri_agent_prompt.txt"

client = OpenAI()


class Agent:
    
    def llm_call(self,prompt, task_description):
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {
                "role": "user",
                "content": task_description
            }
        ]
        )

        return (completion.choices[0].message.content)

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
    
    def image_to_buffer(self,image):
        if os.path.isfile(image):
            with open(image, "rb") as f:
                self.encoded_image = base64.b64encode(f.read()).decode("utf-8")
        else:
            _, buffer = cv2.imencode('.png', image)
            self.encoded_image = base64.b64encode(buffer).decode("utf-8")

    
    def objects_description(self,prompt, image):
        self.image_to_buffer(image)
        json_answer = self.vlm_call(prompt)
        json_answer = json_answer.replace("```json", "").replace("```","")
        return json_answer
    


