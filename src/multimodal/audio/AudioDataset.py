import os
import re
import numpy
from src.internal.father_classes.DatasetFather import DatasetFather
from src.internal.utils.TextualFileManager import TextualFileManager


class TextualDataset(DatasetFather):

    def __init__(self, input_directory_path, output_directory_path, model_name):
        super().__init__(input_directory_path, output_directory_path, model_name)

    def __getitem__(self, idx):
        image_path = os.path.join(self._input_directory_path, self._filenames[idx])
        sample = Image.open(image_path)

        norm_sample = self._pre_processing(sample)

        if 'tensorflow' in self._framework_list:
            # np for tensorflow
            return np.expand_dims(norm_sample, axis=0)
        else:
            # torch
            return norm_sample

    def _pre_processing(self, sample):
        print('im preprocessing yay')
