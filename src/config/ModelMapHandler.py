import os
from src.config.YamlFileManager import YamlFileManager
import tensorflow as tf
import torchvision.models as models


class ModelMapHandler:

    def __init__(self, yaml_map_file):
        # both absolute and relative path are fine
        # self._yaml_manager = YamlFileManager(yaml_map_file)
        # self._data_dict = self._yaml_manager.get_raw_dict()
        # print(self._data_dict)

    def load_model_from_string(self, model_name):






