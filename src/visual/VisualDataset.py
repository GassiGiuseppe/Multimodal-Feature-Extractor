from abc import ABC

from PIL import Image
from src.dataset.DatasetFather import DatasetFather
import tensorflow as tf
from torchvision import transforms
import numpy as np
import os
from src.utils.ModelsMap import tensorflow_models_for_normalization


class VisualDataset(DatasetFather, ABC):

    def __init__(self, input_directory_path, output_directory_path, model_name='VGG19', reshape=(224, 224)):
        super().__init__(input_directory_path, output_directory_path, model_name)
        self.framework_list = None
        self.__reshape = reshape


    def __getitem__(self, idx):
        image_path = os.path.join(self._input_directory_path, self._filenames[idx])
        sample = Image.open(image_path)

        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        norm_sample = self._pre_processing(sample)

        if self._model_name in tensorflow_models_for_normalization and 'tensorflow' in self.framework_list:
            # np for tensorflow
            return np.expand_dims(norm_sample, axis=0)
        else:
            # np for torch
            return norm_sample

    def _pre_processing(self, sample):
        # resize
        if self.__reshape:
            res_sample = sample.resize(self.__reshape, resample=Image.BICUBIC)
        else:
            res_sample = sample

        if self._model_name in tensorflow_models_for_normalization and 'tensorflow' in self.framework_list:
            command = tensorflow_models_for_normalization[self._model_name]
            norm_sample = command.preprocess_input(np.array(res_sample))
        else:
            # if the model is a torch model, the normalization is the same for everyone
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])
            norm_sample = transform(res_sample)

        return norm_sample

    def set_reshape(self, reshape):
        self.__reshape = reshape

    def set_framework(self, framework_list):
        self.framework_list = framework_list
