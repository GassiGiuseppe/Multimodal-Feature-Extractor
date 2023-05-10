import torch
from transformers import pipeline
import torchaudio
from src.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather
import numpy
from transformers import Wav2Vec2Model


class AudioCnnFeatureExtractor(CnnFeatureExtractorFather):
    def __init__(self, gpu='-1'):
        self._model_to_initialize = None
        self._tokenizer = None
        super().__init__(gpu)

    def set_model(self, model_name):
        """
        Args:
            model_name: is the name of the model to use.
                        NOTE: in this case we are using transformers so the model name have to be in its list
        Returns: nothing but it initializes the protected model and tokenizer attributes, later used for extraction
        """
        # sentiment_pipeline = pipeline(model=model_name)
        # model = list(sentiment_pipeline.model.children())[-3]
        # model.eval()
        # model.to(self._device)
        # print(list(torchaudio.transforms.__dict__))
        # torchaudio.transforms.
        if 'torch' in self._framework_list or 'torchaudio' in self._framework_list:
            self._model_to_initialize = getattr(torchaudio.pipelines, model_name)
        elif 'transformers' in self._framework_list:
            self._model = Wav2Vec2Model.from_pretrained(model_name)

    def extract_feature(self, sample_input):
        if 'torch' in self._framework_list or 'torchaudio' in self._framework_list:
            audio = sample_input[0]
            sample_rate = sample_input[1]
            self._model = self._model_to_initialize.get_model()

            # extraction
            features, _ = self._model.extract_features(audio, num_layers=self._output_layer)

            # return the N-Dimensional Tensor as a numpy array
            return numpy.array(features)
        if 'transformers' in self._framework_list:
            # uses less computation since does not calculate gradients
            with torch.no_grad():
                #
                output = self._model(sample_input, output_hidden_states=True)

                #
                layer_output = output.hidden_states[self._output_layer]
                return layer_output.numpy()
