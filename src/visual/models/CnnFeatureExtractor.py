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
class CnnFeatureExtractor:
    def __init__(self, gpu='-1'):
        self.framework_list = None
        self.model = None
        self.output_layer = None
        self.model_name = None
        self.gpu = gpu

        self.device = torch.device("cuda:" + str(self.gpu) if self.gpu != '-1' else "cpu")

        # self.imagenet = imagenet
        # self.pca = PCA(n_components=pca_dim)
        # self.reshape = reshape

    def set_model(self, model_name):
        self.model_name = model_name
        if self.model_name in tensorflow_models_for_extraction and 'tensorflow' in self.framework_list:
            self.model = tensorflow_models_for_extraction[self.model_name]()
        elif self.model_name in torch_models_for_extraction and 'torch' in self.framework_list:
            self.model = torch_models_for_extraction[self.model_name](pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def set_output_layer(self, output_layer):
        self.output_layer = output_layer

    def set_framework(self, framework_list):
        self.framework_list = framework_list

    def extract_feature(self, image):
        if self.model_name in tensorflow_models_for_extraction and 'tensorflow' in self.framework_list:
            input_model = self.model.input
            output_layer = self.model.get_layer(self.output_layer).output
            output = tf.keras.Model(input_model, output_layer)(image, training=False)

        else:
            # torch models??? the layer is linked nowhere
            s1 = torch.nn.Sequential(*list(self.model.children())[:-1])
            s2 = torch.nn.Flatten()
            s3 = torch.nn.Sequential(*list(list(self.model.children())[-1][1:-self.output_layer]))
            feature_model = torch.nn.Sequential(s1, s2, s3)
            feature_model.eval()
            output = np.squeeze(feature_model(
                image[None, ...].to(self.device)
            ).data.cpu().numpy())
        return output
