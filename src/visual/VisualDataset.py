from abc import ABC

from PIL import Image
from src.dataset.DatasetFather import DatasetFather
import tensorflow as tf
from torchvision import transforms
import numpy as np
import os
from src.utils.ModelsMap import TensorflowModelsForNormalization


class VisualDataset(DatasetFather, ABC):

    def __init__(self, input_directory_path, output_directory_path, model_name='VGG19', resize=(224, 224),
                 normalize=True):
        super().__init__(input_directory_path, output_directory_path, model_name)
        self.__resize = resize
        self.__normalize = normalize
        # does it need override?

    def __getitem__(self, idx):
        image_path = os.path.join(self._input_directory_path, self._filenames[idx])
        sample = Image.open(image_path)

        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        norm_sample = self._pre_processing(sample)

        if self._model_name == 'AlexNet':
            # ive changed it sample->norm_sample
            return norm_sample, np.array(norm_sample), self._filenames[idx]
        else:
            # ive changed it sample->norm_sample
            return np.expand_dims(norm_sample, axis=0)

    def _pre_processing(self, sample):
        # resize
        if self.__resize:
            res_sample = sample.resize(self.__resize, resample=Image.BICUBIC)
        else:
            res_sample = sample

        # normalize
        if self.__normalize:

            # if self._model_name == 'ResNet50':
            #   norm_sample = tf.keras.applications.resnet.preprocess_input(np.array(res_sample))
            # elif self._model_name == 'VGG19':
            #   norm_sample = tf.keras.applications.vgg19.preprocess_input(np.array(res_sample))
            # elif self._model_name == 'ResNet152':
            #   norm_sample = tf.keras.applications.resnet.preprocess_input(np.array(res_sample))
            if self._model_name in TensorflowModelsForNormalization.__members__:
                command = TensorflowModelsForNormalization[self._model_name].value
                # norm_sample = exec(command)
                norm_sample = command.preprocess_input(np.array(res_sample))
            elif self._model_name == 'AlexNet':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                                ])
                norm_sample = transform(res_sample)
            else:
                raise NotImplemented('This feature extractor has not been added yet!')
        else:
            return res_sample

        return norm_sample
