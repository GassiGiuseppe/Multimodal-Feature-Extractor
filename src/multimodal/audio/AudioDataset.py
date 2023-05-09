import os
import re
import numpy
import torchaudio
from src.internal.father_classes.DatasetFather import DatasetFather
from src.internal.utils.TextualFileManager import TextualFileManager
import soundfile


# the following function is not called right now. but it will be needed in the future


class AudioDataset(DatasetFather):

    def __init__(self, input_directory_path, output_directory_path):
        super().__init__(input_directory_path, output_directory_path, model_name=None)

    def __getitem__(self, index):

        if 'torch' in self._framework_list or 'torchaudio' in self._framework_list:
            audio_path = os.path.join(self._input_directory_path, self._filenames[index])
            audio, sample_rate = torchaudio.load(audio_path)
            # torchaudio.backend.
            # audio, sample_rate = soundfile.read(audio_path)
            return [audio, sample_rate]
