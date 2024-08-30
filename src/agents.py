from openai import OpenAI
client = OpenAI()


class Agents:
    def __init__(self,image,task_description):
        self.encoded_image = image
        self.task_description = task_description
    
    def single_agent(self):
            agent = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                       "text":"You are a mobile robot with a base allow you to move around the environment.\n\
                            You have a robotic arm with a gripper that allows you to pick up and place one object at time.\n\
                            You are also very capable in describing a scene provided an image in input.\n\
                            From the image, you need to produce in the output a set of relations in the in the form of a triple (subject, relation, object). \n\
                            Write just the triples that are essential to solve the following task: " + self.task_description + "\n\
                            Use specific relation to describe the position of the objects in the scene. Do not use 'next to' but you must \
                            use 'right to', 'left to', 'behind to', 'beside to', 'on'\n\
                            For example, if in a scene there is a door, a table in front of the door and a book on the table \
                            with a pen right to it, your answer should be: \
                            1) (table, in front of, door) \n\
                            2) (book, on, table) \n\
                            3) (pen, on, table) \n\
                            4) (pen, right to, book). \n\
                            For the same task given in input you, should plan a sequence of actions to solve the task.\n\
                            Use univocal name given in the relations of the environment to specify the object.\n\
                            Work as a Markovian agent, so you can only see the last action and the current state of the environment.\n\
                            After each step, update the state of the environment to elaborate the next step execuable in the updated enviroment.\
                            You must use only the following actions for the plan and nothing else: \n\
                            NAVIGATE : for the movement in the scene towards a point far from you, for example 'NAVIGATE to the table'\n\
                            GRAB : for the action of picking up an object and specifying which object to grab, for example 'GRAB bottle'\n\
                            DROP : for the action of placing an object ,specifying where with respect to another object, for example 'DROP bottle left to mug' or 'DROP mug right to bottle' or 'DROP pen into bag'\n\
                            PULL : for the action of pulling an object with the gripper,\n\
                            PUSH: for the action of pushing an object on the ground with the base to free its trajectory if necessary.\n\
                            Write only the actions for the plan and nothing else\n\
                            The output should be in this format:\n\
                            '***RELATIONS***\n\
                            set of relations obtained\n\
                            ***PLAN***\n\
                            set of steps to achieve the task\n'"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encoded_image}",
                    },
                    },
                ],
                }
            ],
            max_tokens=500,
            temperature=0,
            )
            response = (agent.choices[0].message.content)
            return response

    def multi_agent_vision_planning(self):
    
        def enviroment_agent():
            agent = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                        "text":"You are an assistent which is able to accurately describe the content of an image. \n\
                        In particular, you are able to capture the main objects present \
                        in the image and to provide all the relations that exist between them. \n\
                        These relations are described in the form of a triple (subject, relation, object) \
                        and when you answer you are only expected to answer with triples and nothing else. \n\
                        Write just the triples that are essential to solve the following task: " + self.task_description + "\n\
                        Use specific relation to describe the position of the objects in the scene. Do not use 'next to' but you must \
                        use 'right to', 'left to', 'behind to', 'beside to', 'on'\n\
                        For example, if in a scene there is a door, a table in front of the door and a book on the table \
                        with a pen right to it, your answer should be: \
                        1) (table, in front of, door) \n\
                        2) (book, on, table) \n\
                        3) (pen, on, table) \n\
                        4) (pen, right to, book."},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encoded_image}",
                    },
                    },
                ],
                }
            ],
            max_tokens=300,
            temperature=0,
            )

            response = (agent.choices[0].message.content)
            return response


        def description_agent():
            agent = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                        "text":"You are an assistent which is able to accurately describe the content of an image. \n\
                            In particular, you are able to accurately describe the content of the image to make one understand \
                            all the details of the image without seeing it. \n\
                            Use the name in the following relations : " + enviroment_info +" and nothing else without adding adjectives for the objects.\n\
                            You should describe how the scene is made with high level description and precise instruction to solve \
                            the following task : " + self.task_description+". Try to minimize the steps required and find the best plan to solve\n\
                            If the task contains ambiguity in the solution of the task, for example many objects of the same type,\
                            specify the position of the object in the image or in relation to other objects.\n"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encoded_image}",
                    },
                    },
                ],
                }
            ],
            max_tokens=500,
            temperature=0,
            )

            response = (agent.choices[0].message.content)
            return response

        
        enviroment_info = enviroment_agent()
        description_agent_info = description_agent()

        agent = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
        {"role": "system", "content": 
            "You are a mobile robot with a base allow you to move around the environment.\n\
            You have a robotic arm with a gripper that allows you to pick up and place one object at time.\n\
            You know the detailed description of the scene and preliminar instructions that can help you in the definition of the plan \
            with the following information : " + description_agent_info +"\n\
            Remember that you have to use this information just as an help.\n\
            For the task given in input you should plan a sequence of actions to solve the task.\n\
            Use univocal name given in the relations of the environment to specify the object.\n\
            Work as a Markovian agent, so you can only see the last action and the current state of the environment.\n\
            After each step, update the state of the environment to elaborate the next step execuable in the updated enviroment."
            },
        {"role": "user", "content": 
            "The task is " + self.task_description+"\n\
                You must use only the following actions for the plan and nothing else: \n\
                            NAVIGATE : for the movement in the scene towards a point far from you, for example 'NAVIGATE to the table'\n\
                            GRAB : for the action of picking up an object and specifying which object to grab, for example 'GRAB bottle'\n\
                            DROP : for the action of placing an object ,specifying where with respect to another object, for example 'DROP bottle left to mug' or 'DROP mug right to bottle' or 'DROP pen into bag'\n\
                            PULL : for the action of pulling an object with the gripper,\n\
                            PUSH: for the action of pushing an object on the ground with the base to free its trajectory if necessary.\n\
                            Write only the actions for the plan and nothing else"},
        ],
        temperature=0,
        max_tokens=600,
        )
        return enviroment_info,description_agent_info, agent.choices[0].message.content
