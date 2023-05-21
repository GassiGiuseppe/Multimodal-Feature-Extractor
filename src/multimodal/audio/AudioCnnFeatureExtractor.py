import torch
from transformers import pipeline
import torchaudio
from src.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather
import numpy
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, AutoModel


class AudioCnnFeatureExtractor(CnnFeatureExtractorFather):
    def __init__(self, gpu='-1'):
        """

        Args:
            gpu:
        """
        self._model_to_initialize = None
        self._tokenizer = None
        super().__init__(gpu)

    def set_model(self, model_name):
        """
        Args:
            model_name: is the name of the model to use.
        Returns: nothing but it initializes the protected model, later used for extraction
        """
        self._model_name = model_name
        if 'torch' in self._framework_list or 'torchaudio' in self._framework_list:
            model_to_initialize = getattr(torchaudio.pipelines, model_name)
            self._model = model_to_initialize.get_model()
            # self._model.to(self._gpu)
        elif 'transformers' in self._framework_list:
            self._model = Wav2Vec2Model.from_pretrained(self._model_name)

    def extract_feature(self, sample_input):
        """

        Args:
            sample_input:

        Returns:

        """
        audio = sample_input[0]
        sample_rate = sample_input[1]
        if 'torch' in self._framework_list or 'torchaudio' in self._framework_list:
            # extraction
            # num_layer is the number of layers to go trought
            features, _ = self._model.extract_features(audio, num_layers=self._output_layer)
            feature = features[-1]
            # return the N-Dimensional Tensor as a numpy array
            return feature.detach().numpy()
        elif 'transformers' in self._framework_list:
            # feature extraction
            outputs = self._model(audio, output_hidden_states=True)
            # layer extraction
            layer_output = outputs.hidden_states[self._output_layer]
            return layer_output.detach().numpy()
