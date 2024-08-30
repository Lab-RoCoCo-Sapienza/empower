
import pickle
import cv2
import numpy as np
# import time
import base64
import math
import os
from agents import Agents

class Detection:

    def __init__(self):
        task_description_order = "move the objects on the table to have the objects ordered by height from the highest to lowest"
        task_description_exit = "exit the room"
        task_description_diff = "throw away the objects in the corresponding recycling bin"
        task_description_shelf = "move the objects in the shelf in order to have for each level of the shelf only the objects made of the same material"
        task_description_shelf2 = "move the objects in the shelf in order to have exactly two objects for level"
        task_description_jacket = "give me the green jacket from the clothing rack"

        self.task_dict = {
            "order_by_height": task_description_order,
            "exit": task_description_exit,
            "shelf_number": task_description_shelf2,
            "shelf_material": task_description_shelf,
            "recycle": task_description_diff,
            "jacket": task_description_jacket
        }
    
    def run_experiment(self):
        use_case = self.loader_instance.use_case
        image_path = self.loader_instance.SCAN_DIR+"scan.jpg"
        with open(image_path, "rb") as im_file:
            encoded_image = base64.b64encode(im_file.read()).decode("utf-8")

        agents = Agents(encoded_image, self.task_dict[use_case])
        # self.single_agent_info = agents.single_agent() 

        environment_agent_info, description_agent_info, planning_agent_info = agents.multi_agent_vision_planning()

        self.results_multi = {
            "environment_agent_info": environment_agent_info,
            "description_agent_info": description_agent_info,
            "planning_agent_info": planning_agent_info,
        }

        with open(self.loader_instance.DUMP_DIR+"planning.pkl",'wb') as f:
            pickle.dump(self.results_multi, f, protocol=2)
        
        with open(self.loader_instance.DUMP_DIR+"planning.txt",'w') as f:
            f.write(self.results_multi["planning_agent_info"])


    def set_loader(self,loader_instance):
        self.loader_instance = loader_instance
        self.run_experiment()
        self.run_image(image_path=self.loader_instance.SCAN_DIR+"scan.jpg")
        
    def split_word(self,words):
        splitted_word = []
        words = words.lower()
        doc = self.loader_instance.nlp(words)
        for token in doc:
            if token.pos_ == "AUX" or (token.pos_ == "NOUN" and token.dep_ in ["dobj","ROOT","nsubj"]) or token.pos_ == "VERB":
                splitted_word.append(token.text)
        return splitted_word

    def compare_two_words(self,list1, list2):
        word1 = list1.copy()
        word2 = list2.copy()
        min_1 = len(word1)
        min_2 = len(word2)
        sim_word = []
        if min_1 > min_2:
            for index in range(min_1):
                found = False
                sim = 0
                min_2 = len(word2)
                for index_2 in range(min_2):
                    sim = (self.loader_instance.wv.similarity(word1[index], word2[index_2]))
                    if sim > 0.708:
                        found = True
                        sim_word.append(sim)
                        word2.pop(index_2)
                        break
                if not found:
                    sim_word.append(sim)
        else:
            for index in range(min_2):
                found = False
                sim = 0
                min_1 = len(word1)
                for index_2 in range(min_1):
                    sim = (self.loader_instance.wv.similarity(word2[index], word1[index_2]))
                    if sim > 0.708:
                        found = True
                        sim_word.append(sim)
                        word1.pop(index_2)
                        break
                if not found:
                    sim_word.append(sim)
        if sim_word == []:
            return None
        sim_word =  np.mean(sim_word)
        return sim_word

    def is_in_list(self,word,list):
        for obj in list:
            if self.compare_two_words(word, obj) != None and self.compare_two_words(word, obj) > 0.708:
                return True
        return False

    def list_to_yoloworld(self,list):
        list_objects = ""
        for object in list:
            if object != []:
                list_objects += (" ".join(object))
                list_objects += ","
        return list_objects[:-1]

    def get_classes(self,object_relations):
        relation_list = []

        for relation in object_relations:

            relation = relation.replace("(", "")
            relation_object_first = relation.split(")")[1].split(",")[0]
            relation_object_second = relation.split(")")[1].split(",")[2]
            word_1 = self.split_word(relation_object_first)
            word_2 = self.split_word(relation_object_second)
            if not self.is_in_list(word_1,relation_list):
                relation_list.append(word_1)
            if not self.is_in_list(word_2,relation_list):
                relation_list.append(word_2)

        return self.list_to_yoloworld(relation_list)

    def show_mask(self,mask, random_color = True):
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image.numpy() * 255
        mask_image = mask_image.astype(np.uint8)
        return mask_image

    def find_bb_relation(self,relation_object):
        index_ = []
        for index, detection in self.dict_detections.items():
            if detection['label'].lower() in relation_object.lower():
                index_.append(index)
        return index_

    def compare_two_list_of_objects(self,position_in_image_first,position_in_image_second,relation,object_first,object_second):
        if position_in_image_first == {} or position_in_image_second == {}:
            return
    
        if "on" in relation:
            min_distance_x = 30000
            for key_first in position_in_image_first.keys():
                for key_second in position_in_image_second.keys():
                    dis_x = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x'])
                    dis_y = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y'])
                    distance = math.sqrt((dis_x*dis_x + dis_y*dis_y))
                    if min_distance_x > dis_y:
                        index_first = key_first
                        index_second = key_second
                        min_distance_x = distance

            self.data_reordered[index_first]['label'] = object_first
            self.data_reordered[index_second]['label'] = object_second
       
        if "left" in relation:
            min_distance_x = 30000
            min_distance_y = 10000
            for key_first in position_in_image_first.keys():
                for key_second in position_in_image_second.keys():
                    min_distance_x_bb = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x'])//2
                    min_distance_y_bb = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y'])//2
                    if min_distance_x_bb < min_distance_x and min_distance_x_bb != 0 and min_distance_y > min_distance_y_bb :
                        index_first = key_first
                        index_second = key_second
                        min_distance_x = min_distance_x_bb
                        min_distance_y = min_distance_y_bb

            self.data_reordered[index_first]['label'] = object_first
            self.data_reordered[index_second]['label'] = object_second
            self.dict_detections.pop(index_first)
            self.dict_detections.pop(index_second)

        if "right" in relation:
            min_distance_x = 30000
            min_distance_y = 10000
            for key_first in position_in_image_first.keys():
                for key_second in position_in_image_second.keys():
                    min_distance_x_bb = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x'])//2
                    min_distance_y_bb = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y'])//2
                    if min_distance_x_bb < min_distance_x and min_distance_x_bb != 0 and min_distance_y > min_distance_y_bb and min_distance_y_bb != 0:
                        
                        index_first = key_first
                        index_second = key_second
                        min_distance_x = min_distance_x_bb
                        min_distance_y = min_distance_y_bb
            self.data_reordered[index_first]['label'] = object_first
            self.data_reordered[index_second]['label'] = object_second
            self.dict_detections.pop(index_first)
            self.dict_detections.pop(index_second)

        

    def obtain_bb_grounded(self,index_first,index_second,relation,object_first,object_second):
            detection_data = self.dict_detections
            position_in_image_first = {}
            position_in_image_second = {}
            for i in range(len(index_first)):
                if index_first[i] not in position_in_image_first.keys():
                    position_in_image_first[index_first[i]] = {'x':None,'y':None}
                position_in_image_first[index_first[i]]['x'] = (detection_data[index_first[i]]['bbox'][0] + detection_data[index_first[i]]['bbox'][2]) //2
                position_in_image_first[index_first[i]]['y'] = (detection_data[index_first[i]]['bbox'][1] + detection_data[index_first[i]]['bbox'][3]) //2
            for i in range(len(index_second)):
                if index_second[i] not in position_in_image_second.keys():
                    position_in_image_second[index_second[i]] = {'x':None,'y':None}
                position_in_image_second[index_second[i]]['x'] = (detection_data[index_second[i]]['bbox'][0] + detection_data[index_second[i]]['bbox'][2]) //2
                position_in_image_second[index_second[i]]['y'] = (detection_data[index_second[i]]['bbox'][1] + detection_data[index_second[i]]['bbox'][3]) //2
            self.compare_two_list_of_objects(position_in_image_first,position_in_image_second,relation,object_first,object_second)


    def run_image(self,image_path):
        index_detection = 0
        print("Running detection")
        object_relations = self.results_multi['environment_agent_info'].split('\n')

        # print("Time")
        # start = time.time()
        labels = self.get_classes(object_relations)

        # print("end vocabulary : " + str(time.time() -start))

        self.loader_instance.yolow_model.set_class_name(labels)

        image = cv2.imread(image_path)
        masked_image = image.copy()
        image_with_bbox = image.copy()
        image_yolow = image.copy()
        self.dict_detections = {}
        overlay_ = masked_image 

        #YOLOW inference
        # start = time.time()
        bboxs, scores, labels_idx = self.loader_instance.yolow_model(image_path)
        # print("detection: " + str(time.time()- start))
        # start = time.time()
        for i, (bbox,score,cls_id) in enumerate(zip(bboxs[0], scores, labels_idx[0])):
            x1,y1,x2,y2 = bbox

            if score > 0.1:

                label = self.loader_instance.yolow_model.get_class_name(cls_id)
                masks, _ = self.loader_instance.vit_sam_model(masked_image, bbox)
               
                if index_detection not in self.dict_detections.keys():
                    self.dict_detections[index_detection] = {'bbox':None,'label':None}
                    self.dict_detections[index_detection]['bbox'] = bbox
                    self.dict_detections[index_detection]['label'] = label
                    cv2.rectangle(image_yolow, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                    cv2.putText(image_yolow, f"{label}: {score:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
                    
                # Convert binary mask to 3-channel image
                for mask in masks:
                    binary_mask = self.show_mask(mask)
                    overlay = masked_image
                    self.dict_detections[index_detection]['mask'] = binary_mask
                    overlay = cv2.addWeighted(overlay, 1, binary_mask, 0.5, 0)
                    cv2.imwrite(self.loader_instance.DUMP_DIR+f"rgb_{index_detection}.jpg", overlay)
                    overlay_ = cv2.addWeighted(overlay_, 1, binary_mask, 0.5, 0)
                index_detection += 1
        
        # print("mask : " + str(i) + " : " + str(time.time() -start))
        cv2.imwrite(self.loader_instance.DUMP_DIR+"mask.jpg", overlay_)
        cv2.imwrite(self.loader_instance.DUMP_DIR+"yolo.jpg", image_yolow)

        self.data_reordered = self.dict_detections.copy()
        # start = time.time()
        for relation in object_relations:
                relation = relation.replace("(", "")

                relation_object_first = relation.split(")")[1].split(",")[0]
                relation_object_first = relation_object_first[1:]
                relation_type = relation.split(")")[1].split(",")[1]
                relation_type = relation_type[1:]
                relation_object_second = relation.split(")")[1].split(",")[2]
                relation_object_second = relation_object_second[1:]
                index_bounding_box_first = self.find_bb_relation(relation_object_first)
                index_bounding_box_second = self.find_bb_relation(relation_object_second)
                if index_bounding_box_first != [] or index_bounding_box_second != []:
                    self.obtain_bb_grounded(index_bounding_box_first,index_bounding_box_second,relation_type,relation_object_first,relation_object_second)

        # print("grounfing : " + str(time.time() -start))
        for i, value in self.data_reordered.items():
            cv2.rectangle(image_with_bbox, (int(value['bbox'][0]), int(value['bbox'][1])), (int(value['bbox'][2]), int(value['bbox'][3])), (255, 0, 0), 1 )
            cv2.putText(image_with_bbox, value['label'], (int(value['bbox'][0]+10), int(value['bbox'][1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        with open(self.loader_instance.DUMP_DIR+"detection.pkl", 'wb') as f:
            pickle.dump(self.data_reordered, f, protocol=2)
        
        cv2.imwrite(self.loader_instance.DUMP_DIR+"scan_with_bb.jpg", image_with_bbox)
        os.system(f'rm -rf {self.loader_instance.YOLOW_PATH}/logs/')