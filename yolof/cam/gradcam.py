import cv2

import numpy as np

import torch

class PrintLogger:
    def info(self, message):
        print(f"INFO: {message}")

    def warn(self, message):
        print(f"WARN: {message}")

    def error(self, message):
        print(f"ERROR: {message}")

    def debug(self, message):
        print(f"DEBUG: {message}")

class GradCAM():
    """
    Class to implement the GradCam function with it's necessary Pytorch hooks.

    Attributes
    ----------
    model : detectron2 GeneralizedRCNN Model
        A model using the detectron2 API for inferencing
    layer_name : str
        name of the convolutional layer to perform GradCAM with
    """

    def __init__(self, model, target_layer_name, logger=None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradient = None
        self.model.eval()
        self.hooks = {}

        self.logger = logger if logger is not None else PrintLogger()
        self._register_forward_hook()
        self._register_backward_hook()

    def _get_activations_hook(self, module, input, output):
        self.activations = output.cpu().data.numpy()

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0].cpu().data.numpy()

    def _register_forward_hook(self):
        named_modules = dict(self.model.named_modules())

        if self.target_layer_name in named_modules:
            module = named_modules[self.target_layer_name]
            self.hooks["forward"] =  module.register_forward_hook(self._get_activations_hook)
        else:
            self.logger.error(f"Layer {self.target_layer_name} not found in Model!")

        # self.logger.info(f"Activations hooks registered to layer {self.target_layer_name}")

    def _register_backward_hook(self):
        named_modules = dict(self.model.named_modules())

        if self.target_layer_name in named_modules:
            module = named_modules[self.target_layer_name]
            self.hooks["backward"] = module.register_full_backward_hook(self._get_grads_hook)
        else:
            self.logger.error(f"Layer {self.target_layer_name} not found in Model!")

        # self.logger.info(f"Gradients hooks registered to layer {self.target_layer_name}")

    def _release_grads(self):
        for name, handle in self.hooks.items():
            if name != "forward":
                handle.remove()

        self.gradient = None
        self.hooks.pop("backward")
    
    def _postprocess_cam(self, raw_cam, img_width, img_height):
        cam_orig = np.sum(raw_cam, axis=0)  # [H,W]
        cam_orig = np.maximum(cam_orig, 0)  # ReLU
        cam_orig -= np.min(cam_orig)
        
        if np.count_nonzero(cam_orig) == 0:
            return cv2.resize(cam_orig, (img_width, img_height))
        
        cam_orig /= np.max(cam_orig)
        cam = cv2.resize(cam_orig, (img_width, img_height))
        return cam
    
    
    def _forward_pass(self, inputs):
        torch.set_grad_enabled(True)
        with torch.enable_grad():
            output = self.model([inputs])[0]
            return output
    
    def _backward_pass(self, outputs, target_instance):
        if target_instance == None:
          target_instance =  np.argmax(outputs.scores.cpu().data.numpy(), axis=-1)
        
        if len(outputs) <= target_instance:
            self.logger.warn(f"Only {len(outputs)} objects found but you request object number {target_instance}")
            return 
        else:
            score = outputs.scores[target_instance]
            score.retain_grad()
            score.backward(retain_graph=True)
            # score.backward()


    def get_cam(self, output, target_instance):
        self._backward_pass(output, target_instance=target_instance)
        if len(output):
            if self.gradient is not None:
                gradient = self.gradient[0]  # [C,H,W]
                activations = self.activations[0]  # [C,H,W]
                weight = np.mean(gradient, axis=(1, 2))  # [C]

                cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
                cam = self._postprocess_cam(cam, output.image_size[1], output.image_size[0])

                return cam
            else:
                return np.array([])
        else:
            return np.array([])


    def __enter__(self):
        return self
    

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass


    def __call__(self, inputs):
        """
        Calls the GradCAM instance

        Parameters
        ----------
        inputs : dict
            The input in the standard detectron2 model input format
            https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format

        target_instance : int, optional
            The target category index. If `None` the highest scoring class will be selected

        Returns
        -------
        cam : np.array()
          Gradient weighted class activation map
        output : list
          list of Instance objects representing the detectron2 model output
        """
        
        self.model.zero_grad()
        return self._forward_pass(inputs)


class GradCamPlusPlus(GradCAM):
    """
    Subclass to implement the GradCam++ function with it's necessary PyTorch hooks.
    ...

    Attributes
    ----------
    model : detectron2 GeneralizedRCNN Model
        A model using the detectron2 API for inferencing
    target_layer_name : str
        name of the convolutional layer to perform GradCAM++ with

    """
    def __init__(self, model, target_layer_name):
        super().__init__(model, target_layer_name)

    def get_cam(self, output, target_instance):
        self._backward_pass(output, target_instance=target_instance)
        if len(output):
            if self.gradient is not None:
                gradient = self.gradient[0]  # [C,H,W]
                activations = self.activations[0]  # [C,H,W]

                #from https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py
                grads_power_2 = gradient**2
                grads_power_3 = grads_power_2 * gradient
                # Equation 19 in https://arxiv.org/abs/1710.11063
                sum_activations = np.sum(activations, axis=(1, 2))
                eps = 0.000001
                aij = grads_power_2 / (2 * grads_power_2 +
                                    sum_activations[:, None, None] * grads_power_3 + eps)
                # Now bring back the ReLU from eq.7 in the paper,
                # And zero out aijs where the activations are 0
                aij = np.where(gradient != 0, aij, 0)

                weights = np.maximum(gradient, 0) * aij
                weight = np.sum(weights, axis=(1, 2))

                cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
                cam = self._postprocess_cam(cam, output.image_size[1], output.image_size[0])

                return cam
            else:
                return np.array([])
        else:
            return np.array([])
