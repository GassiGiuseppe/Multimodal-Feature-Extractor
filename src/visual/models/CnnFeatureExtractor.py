from sklearn.decomposition import PCA
import tensorflow as tf
import torchvision.models as models
import torch
import numpy as np
import os

from torchvision.datasets import imagenet
from src.utils.ModelsMap import TensorflowModelsForExtraction


# convolution neural network
# the file is heavily cut, the complete file is in the 'old' directory
class CnnFeatureExtractor:
    def __init__(self, model_name, output_layer, reshape=False, pca_dim=None, gpu='-1'):
        self.model_name = model_name
        self.gpu = gpu
        self.output_layer = output_layer

        self.device = torch.device("cuda:" + str(self.gpu) if self.gpu != '-1' else "cpu")
        self.imagenet = imagenet
        self.pca = PCA(n_components=pca_dim)
        self.reshape = reshape

        #if self.model_name == 'ResNet50':
         #   self.model = tf.keras.applications.ResNet50()
        #elif self.model_name == 'VGG19':
         #   self.model = tf.keras.applications.VGG19()
        #elif self.model_name == 'ResNet152':
         #   self.model = tf.keras.applications.ResNet152()
        if self.model_name in TensorflowModelsForExtraction.__members__:
            command = TensorflowModelsForExtraction[self.model_name].value
            self.model = command
        elif self.model_name == 'AlexNet':
            self.model = models.alexnet(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def extract_feature(self, image):
        input_model = self.model.input
        output_layer = self.model.get_layer(self.output_layer).output
        output = tf.keras.Model(input_model, output_layer)(image, training=False)
        return output
