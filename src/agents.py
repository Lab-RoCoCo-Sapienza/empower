from openai import OpenAI
from paths import PROMPT_DIR, OUTPUT_DIR
import base64
#import networkx as nx
#import matplotlib.pyplot as plt
import json
import cv2
import os
import re

visual_prompt = PROMPT_DIR+"visual_agent_prompt.txt"
plan_prompt = PROMPT_DIR+"planner_hri_agent_prompt.txt"

client = OpenAI()
class Agent:
    
    def llm_call(self,prompt, task_description):
        completion = client.chat.completions.create(
        model="gpt-4o",
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

    
    def objects_description(self,image):
        prompt = open(visual_prompt,"r").read()
        self.image_to_buffer(image)
        json_answer = self.vlm_call(prompt)
        json_answer = json_answer.replace("```json", "").replace("```","")
        #print(json_answer)
        json_stucture = json.loads(json_answer)
        #self.create_graph(json_stucture)
        
        with open(OUTPUT_DIR + 'json_data.txt', 'w') as outfile:
            json.dump(json_stucture, outfile,indent=4)
        
        return json_stucture
    '''
    def create_graph(self, json_structure):
        G = nx.Graph()
        for obj in json_structure['Objects']:
            label = obj['Label']
            relations = obj['Relations']            
            G.add_node(label)
            
            # Aggiungi archi per le relazioni
            for relation in relations:
                relation_attribute, label2 = relation.split(";")[0], relation.split(";")[1]
                G.add_edge(label, label2, relation=relation_attribute)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.show()
    '''
