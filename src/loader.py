import os
import time
from models import YOLOW, VitSam
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

from paths import *

class Loader:
    _instance = None

    def __new__(cls, *args):
        if not cls._instance:
            cls._instance = super(Loader, cls).__new__(cls)
            cls._instance.initialize(*args)
        return cls._instance

    def initialize(self, *args):
        use_case = args[0]
        self._use_case = use_case

        self._CONFIG = CONFIG_DIR
        self._IMAGES = IMAGES_DIR
        self._OUTPUT = OUTPUT_DIR

        self._YOLOW_PATH = self._CONFIG + "yolow/"
        self._ENCODER_PATH = self._CONFIG + "efficientvitsam/l2_encoder.onnx"
        self._DECODER_PATH = self._CONFIG + "efficientvitsam/l2_decoder.onnx"

        self._SCAN_DIR = self._IMAGES + use_case + '/'
        os.makedirs(self._SCAN_DIR, exist_ok=True)
        self._DUMP_DIR = self._OUTPUT + use_case + '/'
        os.makedirs(self._DUMP_DIR, exist_ok=True)
        self._MASKED_SCANS_DIR = self

        self._yolow_model = YOLOW(self._YOLOW_PATH)
        self._vit_sam_model = VitSam(self._ENCODER_PATH, self._DECODER_PATH) 

        self._nlp = spacy.load("en_core_web_sm") 
        self._wv = api.load('word2vec-google-news-300') 

    @property
    def nlp(self):
        return self._nlp
    
    @nlp.setter
    def nlp(self, value):
        self._nlp = value

    @property
    def wv(self):
        return self._wv
    
    @wv.setter
    def wv(self, value):
        self._wv = value

    @property
    def yolow_model(self):
        return self._yolow_model
    
    @yolow_model.setter
    def yolow_model(self, value):
        self._yolow_model = value

    @property
    def vit_sam_model(self):
        return self._vit_sam_model
    
    @vit_sam_model.setter
    def vit_sam_model(self, value):
        self._vit_sam_model = value
        
    @property
    def CONFIG(self):
        return self._CONFIG
    
    @CONFIG.setter
    def CONFIG(self, value):
        self._CONFIG = value
    
    @property
    def IMAGES(self):
        return self._IMAGES
    
    @IMAGES.setter
    def IMAGES(self, value):
        self._IMAGES = value
    
    @property
    def YOLOW_PATH(self):
        return self._YOLOW_PATH
    
    @YOLOW_PATH.setter
    def YOLOW_PATH(self, value):
        self._YOLOW_PATH = value

    @property
    def ENCODER_PATH(self):
        return self._ENCODER_PATH
    
    @ENCODER_PATH.setter
    def ENCODER_PATH(self, value):
        self._ENCODER_PATH = value
    
    @property
    def DECODER_PATH(self):
        return self._DECODER_PATH
    
    @DECODER_PATH.setter
    def DECODER_PATH(self, value):
        self._DECODER_PATH = value
    
    @property
    def SCAN_DIR(self):
        return self._SCAN_DIR
    
    @SCAN_DIR.setter
    def SCAN_DIR(self, value):
        self._SCAN_DIR = value
    
    @property
    def DUMP_DIR(self):
        return self._DUMP_DIR
    
    @DUMP_DIR.setter
    def DUMP_DIR(self, value):
        self._DUMP_DIR = value
    
    @property
    def OUTPUT(self):
        return self._OUTPUT
    
    @OUTPUT.setter
    def OUTPUT(self, value):
        self._OUTPUT = value
    
    @property
    def use_case(self):
        return self._use_case
    
    @use_case.setter
    def use_case(self, value):
        self._use_case = value