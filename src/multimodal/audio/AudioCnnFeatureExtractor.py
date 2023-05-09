import torch
from transformers import pipeline
import torchaudio
from src.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather
import numpy

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
        self._model_to_initialize = getattr(torchaudio.pipelines, model_name)

    def extract_feature(self, sample_input):
        if 'torch' in self._framework_list or 'torchaudio' in self._framework_list:
            audio = sample_input[0]
            sample_rate = sample_input[1]
            self._model = self._model_to_initialize.get_model()
            features = self._model.extract_feature(audio)
            # audio = torch.from_numpy(audio)
            # spectral_transform = torchaudio.transforms.Spectrogram(n_fft=2048, win_length=2048, hop_length=1024)
            # spectral_features = spectral_transform(audio)
            #audio = torch.nn.functional.pad(audio, (0, 78163 - audio.shape[-1]), mode='constant', value=0)

            # normalized feature
            log_spectrogram = torchaudio.transforms.AmplitudeToDB()(features)

            # convert to numpy
            log_spectrogram_np = log_spectrogram.numpy()

            # layer output
            # layer_output = features[self._output_layer]


            return log_spectrogram_np
