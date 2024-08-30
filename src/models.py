

import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from torchvision.ops import nms
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientvit.export_encoder import SamResize
from efficientvit.inference import SamDecoder, SamEncoder

class VitSam():

    def __init__(self, encoder_model, decoder_model):
        self.decoder = SamDecoder(decoder_model)
        self.encoder = SamEncoder(encoder_model)


    def __call__(self, img, bboxes):
        raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_image_size = raw_img.shape[:2]
        img = self._preprocess(raw_img, img_size=512)
        img_embeddings = self.encoder(img)
        boxes = np.array(bboxes, dtype=np.float32)
        masks, _, _ = self.decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            boxes=boxes,
        )

        return masks, boxes

    def _preprocess(self, x, img_size=512):
        pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
        pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

        x = torch.tensor(x)
        resize_transform = SamResize(img_size)
        x = resize_transform(x).float() / 255
        x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

        h, w = x.shape[-2:]
        th, tw = img_size, img_size
        assert th >= h and tw >= w
        x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy()

        return x
    
    

class YOLOW():

    def __init__(self,YOLOW_PATH):
        cfg = Config.fromfile(
            YOLOW_PATH + "yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        )
        cfg.work_dir = YOLOW_PATH
        cfg.load_from = YOLOW_PATH + "yolow.pth"
        cfg.__setattr__("log_level","WARNING")
        self.runner = Runner.from_cfg(cfg)
        self.runner.call_hook("before_run")
        self.runner.load_or_resume()
        pipeline = cfg.test_dataloader.dataset.pipeline
        self.runner.pipeline = Compose(pipeline)
        self.runner.model.eval()

    def set_class_name(self,objects):
        self.class_names = (objects)
        self.objects = objects.split(",")

    def get_class_name(self, id):
        return self.objects[id]

    def __call__(self,input_image,max_num_boxes=100,score_thr=0.05,nms_thr=0.5):

        texts = [[t.strip()] for t in self.class_names.split(",")] + [[" "]]
        data_info = self.runner.pipeline(dict(img_id=0, img_path=input_image,
                                        texts=texts))

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]
            self.runner.model.class_names = texts
            pred_instances = output.pred_instances

        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]
        output.pred_instances = pred_instances

        pred_instances = pred_instances.cpu().numpy()

        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores'] 

        return xyxy, confidence, class_id
