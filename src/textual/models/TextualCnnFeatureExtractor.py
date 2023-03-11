from sklearn.decomposition import PCA
import tensorflow as tf
import torchvision.models as models
import torch
import numpy as np
import os
from transformers import pipeline

from torchvision.datasets import imagenet
from src.utils.model_map import tensorflow_models_for_extraction, torch_models_for_extraction


# convolution neural network
# the file is heavily cut, the complete file is in the 'old' directory
class TextualCnnFeatureExtractor:
    def __init__(self, gpu='-1'):
        self.tokenizer = None
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
        '''

        # old....

        self._model_name = model_name
        if self._model_name in tensorflow_models_for_extraction and 'tensorflow' in self._framework_list:
            self.model = tensorflow_models_for_extraction[self._model_name]()
        elif self._model_name in torch_models_for_extraction and 'torch' in self._framework_list:
            self.model = torch_models_for_extraction[self._model_name](pretrained=True)
            self.model.to(self._device)
            self.model.eval()
        else:
            raise NotImplemented('This feature extractor has not been added yet!')
        '''
        sentiment_pipeline = pipeline(model=model_name)

        # this is as the professor has written
        model = list(sentiment_pipeline.model.children())[-3] # HERE LAYER???
        # this is as hugme suggest:
        # model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        model.eval()
        model.to(self._device)
        self.model = model
        self.tokenizer = sentiment_pipeline.tokenizer
        # or
        # AutoTokenizer.from_pretrained("bert-base-cased")

    def set_output_layer(self, output_layer):
        self._output_layer = output_layer

    def set_framework(self, framework_list):
        self._framework_list = framework_list

    def extract_feature(self, sample_input):
        output = self.tokenizer.encode_plus(sample_input, return_tensors="pt").to(self._device)
        return self.model(**output.to(self._device)).pooler_output.detach().cpu().numpy()
