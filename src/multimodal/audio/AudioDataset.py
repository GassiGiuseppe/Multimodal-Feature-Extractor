import os
import re
import numpy
import torchaudio
from src.internal.father_classes.DatasetFather import DatasetFather
from src.internal.utils.TextualFileManager import TextualFileManager
import soundfile
from transformers import Wav2Vec2Processor


class AudioDataset(DatasetFather):

    def __init__(self, input_directory_path, output_directory_path):
        super().__init__(input_directory_path, output_directory_path, model_name=None)
        self._model_for_preprocessing = None

    def __getitem__(self, index):
        audio_path = os.path.join(self._input_directory_path, self._filenames[index])

        # right now both torchaudio and transformers do the same thing, but it is preferable to keep them work
        # separately for future improvement

        if 'torch' in self._framework_list or 'torchaudio' in self._framework_list:
            audio, sample_rate = torchaudio.load(audio_path)
            return self._pre_processing([audio, sample_rate])
        elif 'transformers' in self._framework_list:
            audio, sample_rate = torchaudio.load(audio_path)
            return self._pre_processing([audio, sample_rate])

    def set_model(self, model):
        self._model_name = model
        if 'transformers' in self._framework_list:
            self._model_for_preprocessing = Wav2Vec2Processor.from_pretrained(self._model_name)

    def _pre_processing(self, pre_process_input):
        audio = pre_process_input[0]
        rate = pre_process_input[1]
        if 'torch' in self._framework_list or 'torchaudio' in self._framework_list:
            bundle = getattr(torchaudio.pipelines, self._model_name)
            waveform = torchaudio.functional.resample(audio, rate, bundle.sample_rate)
            return [waveform, bundle.sample_rate]
        elif 'transformers' in self._framework_list:
            '''
            new_sample_rate = self._model_for_preprocessing.feature_extractor.sampling_rate
            resampler = torchaudio.transforms.Resample(rate, new_sample_rate)
            waveform = resampler(audio[0])
            waveform = waveform.unsqueeze(0)
            return self._model_for_preprocessing(waveform, sampling_rate=new_sample_rate,
                                                 return_tensors="pt").input_values
            '''
            return pre_process_input
