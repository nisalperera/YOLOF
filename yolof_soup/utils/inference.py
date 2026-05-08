import torch

from detectron2.engine import DefaultPredictor


class InferenceWrapper(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model.eval()

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

    def zero_grad(self):
        self.model.zero_grad()
    
    def named_parameters(self):
        return self.model.named_parameters()
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=False):
        self.model.load_state_dict(state_dict, strict=strict)