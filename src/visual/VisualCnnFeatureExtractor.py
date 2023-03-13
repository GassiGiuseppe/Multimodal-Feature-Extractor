import tensorflow as tf
import torch
import numpy as np

from src.internal.utils.model_map import tensorflow_models_for_extraction, torch_models_for_extraction
from src.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather


class VisualCnnFeatureExtractor(CnnFeatureExtractorFather):
    def __init__(self, gpu='-1'):
        super().__init__(gpu)

    def set_model(self, model_name):
        """
        Args:
            model_name: is the name of the model to use.
                        NOTE: the model name have to be in the model map in utils
        Returns: nothing but it initializes the protected model attribute, later used for extraction
        """
        self._model_name = model_name
        if self._model_name in tensorflow_models_for_extraction and 'tensorflow' in self._framework_list:
            self._model = tensorflow_models_for_extraction[self._model_name]()
        elif self._model_name in torch_models_for_extraction and 'torch' in self._framework_list:
            self._model = torch_models_for_extraction[self._model_name](pretrained=True)
            self._model.to(self._device)
            self._model.eval()
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def extract_feature(self, image):
        if self._model_name in tensorflow_models_for_extraction and 'tensorflow' in self._framework_list:
            input_model = self._model.input
            output_layer = self._model.get_layer(self._output_layer).output
            output = tf.keras.Model(input_model, output_layer)(image, training=False)
            # update the framework list
            self._framework_list = ['tensorflow']

        else:

            s1 = torch.nn.Sequential(*list(self._model.children())[:-1])
            s2 = torch.nn.Flatten()
            if isinstance(list(self._model.children())[-1], torch.nn.Linear):

                s3 = list(self._model.children())[-1]
            else:

                s3 = torch.nn.Sequential(*list(list(self._model.children())[-1][1:-self._output_layer]))
            feature_model = torch.nn.Sequential(s1, s2, s3)
            feature_model.eval()
            output = np.squeeze(feature_model(
                image[None, ...].to(self._device)
            ).data.cpu().numpy())
            # update the framework list
            self._framework_list = ['torch']

        return output
