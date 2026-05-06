import torch
import detectron2.data.transforms as T

from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import read_image


class InferenceWrapper(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model.eval()

    def predict(self, original_image):
        return self.model([original_image])[0]
        
    def __call__(self, original_images):
        with torch.no_grad(): 
            predictions = []
            for original_image in original_images:
                predictions.append(self.predict(original_image))
        return predictions
    
    def eval(self):
        # Override eval to ensure model is in eval mode and no dropout/batchnorm updates occur.
        self.model.eval()