from typing import Dict

import torch

from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


class EvaluateModel():
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def _predict(self, original_image):
        return self.model([original_image])[0]
    
    def predict(self, original_images):
        predictions = []
        for original_image in original_images:
            predictions.append(self._predict(original_image))
        
        return predictions
    
    def __call__(self, original_images, require_grad=False):
        if require_grad:
            with torch.enable_grad():
                return self.predict(original_images)
        else:
            with torch.no_grad(): 
                return self.predict(original_images)
    
    def eval(self):
        # Override eval to ensure model is in eval mode and no dropout/batchnorm updates occur.
        self.model.eval()
    
    def to(self, device):
        self.model.to(device)
        return self
    

class BNCalibration():
    def __init__(self, cfg, state_dict: Dict[str, torch.Tensor]=None):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.train()
        if len(cfg.DATASETS.TRAIN):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        checkpointer = DetectionCheckpointer(self.model)
        if state_dict is not None:
            checkpointer._load_model({"model": state_dict})
        else:
            checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def _predict(self, original_image):
        return self.model(original_image)
    
    def __call__(self, original_images):
        return self._predict(original_images)
    
    def eval(self):
        # Override eval to ensure model is in eval mode and no dropout/batchnorm updates occur.
        self.model.eval()

    def train(self):
        # Override train to ensure model is in train mode and batchnorm updates occur.
        self.model.train()
    
    def to(self, device):
        self.model.to(device)
        return self

