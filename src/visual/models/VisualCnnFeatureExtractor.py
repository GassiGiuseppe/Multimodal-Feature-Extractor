from sklearn.decomposition import PCA
import tensorflow as tf
import torchvision.models as models
import torch
import numpy as np
import os

from torchvision.datasets import imagenet
from src.utils.model_map import tensorflow_models_for_extraction, torch_models_for_extraction


# convolution neural network
# the file is heavily cut, the complete file is in the 'old' directory
class VisualCnnFeatureExtractor:
    def __init__(self, gpu='-1'):
        self._framework_list = None
        self.model = None
        self._output_layer = None
        self._model_name = None
        self._gpu = gpu

        self._device = torch.device("cuda:" + str(self._gpu) if self._gpu != '-1' else "cpu")

        # self.imagenet = imagenet
        # self.pca = PCA(n_components=pca_dim)
        # self.reshape = reshape

    def set_model(self, model_name):
        self._model_name = model_name
        if self._model_name in tensorflow_models_for_extraction and 'tensorflow' in self._framework_list:
            self.model = tensorflow_models_for_extraction[self._model_name]()
        elif self._model_name in torch_models_for_extraction and 'torch' in self._framework_list:
            self.model = torch_models_for_extraction[self._model_name](pretrained=True)
            self.model.to(self._device)
            self.model.eval()
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def set_output_layer(self, output_layer):
        self._output_layer = output_layer

    def set_framework(self, framework_list):
        self._framework_list = framework_list

    def extract_feature(self, image):
        if self._model_name in tensorflow_models_for_extraction and 'tensorflow' in self._framework_list:
            input_model = self.model.input
            output_layer = self.model.get_layer(self._output_layer).output
            output = tf.keras.Model(input_model, output_layer)(image, training=False)
            # update the framework list
            self._framework_list = ['tensorflow']

        else:

            s1 = torch.nn.Sequential(*list(self.model.children())[:-1])
            s2 = torch.nn.Flatten()
            if isinstance(list(self.model.children())[-1], torch.nn.Linear):
                # HERE HERE HERE there is a issue
                s3 = list(self.model.children())[-1]
            else:
                # -1 instead of
                s3 = torch.nn.Sequential(*list(list(self.model.children())[-1][1:-self._output_layer]))
            feature_model = torch.nn.Sequential(s1, s2, s3)
            feature_model.eval()
            output = np.squeeze(feature_model(
                image[None, ...].to(self._device)
            ).data.cpu().numpy())
            # update the framework list
            self._framework_list = ['torch']

        return output
