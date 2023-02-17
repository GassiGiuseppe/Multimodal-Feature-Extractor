import os

from config.Config import Config
from visual.VisualDataset import VisualDataset


def foo():
    config = Config()

    # set gpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get_gpu()

    if config.has_config('items', 'textual'):
        working_paths = config.paths_for_extraction('items', 'textual')
        for model in config.get_models_list('items', 'textual'):
            print('here do')
    if config.has_config('items', 'visual'):
        working_paths = config.paths_for_extraction('items', 'visual')
        visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])
        for model in config.get_models_list('items', 'visual'):
            print('here do')
        # HERE
        # the following code is used during test
        for index in range(visual_dataset.__len__()):
            visual_dataset.__getitem__(index)

        print(working_paths)
    if config.has_config('interactions', 'textual'):
        working_paths = config.paths_for_extraction('interactions', 'textual')
        for model in config.get_models_list('interactions', 'textual'):
            print('here do')
    if config.has_config('interactions', 'visual'):
        working_paths = config.paths_for_extraction('interactions', 'visual')
        for model in config.get_models_list('interactions', 'visual'):
            print('here do')


if __name__ == '__main__':
    foo()
