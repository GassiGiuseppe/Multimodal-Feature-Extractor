from abc import abstractmethod
from src.utils.human_sorting import human_sort
import os


class DatasetFather:
    def __init__(self, input_directory_path, output_directory_path):
        self.__input_directory_path = input_directory_path
        self.__output_directory_path = output_directory_path

        # the input path must already exist since is where are located the input file
        if not os.path.exists(self.__input_directory_path):
            raise FileExistsError('input folder does not exists')
        if not os.path.exists(self.__output_directory_path):
            os.makedirs(self.__output_directory_path)

        self._filenames = os.listdir(self.__input_directory_path)
        self._filenames = human_sort(self._filenames)

        self.__num_samples = len(self._filenames)

    def __len__(self):
        return self.__num_samples

    def __getfilenames__(self):
        return self._filenames

    @abstractmethod
    def __getitem__(self, idx):
        pass


