from abc import ABC

from src.dataset.DatasetFather import DatasetFather


class VisualDataset(DatasetFather):

    def __init__(self, input_directory_path, output_directory_path):
        super().__init__(input_directory_path, output_directory_path)
        # does it need override?

    def __getitem__(self, index):
        element_name = self._filenames[index]
        print(element_name)
        # this method will be a copy of the same method in src/visual/Dataset
