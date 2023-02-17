

from src.dataset.DatasetFather import DatasetFather


class TextualDataset(DatasetFather):

    def __init__(self, input_directory_path, output_directory_path):
        super().__init__(input_directory_path, output_directory_path)
        # does it need override?

    def __getitem__(self, index):
        print(index)

    #WARNING: HAVE NOT TO BE HERE, it is here momentarily
    def textual_preprocessing(self, text_to_process):
        print('nothing')