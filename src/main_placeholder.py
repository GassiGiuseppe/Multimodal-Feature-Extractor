import os

from config.Config import Config
from visual.VisualDataset import VisualDataset
from visual.models.CnnFeatureExtractor import CnnFeatureExtractor


def foo():
    config = Config()

    # set gpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get_gpu()

    # NOTE FOR FUTURE DEVELOP: if these ifs are all the same, create a unified fun

    if config.has_config('items', 'visual'):
        working_paths = config.paths_for_extraction('items', 'visual')
        models = config.get_models_list('items', 'visual')
        for model in models.keys():
            print(model)
            visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'], model_name=model)
            model_layer = models[model]['output_layers']
            cnn_feature_extractor = CnnFeatureExtractor(model, model_layer)
            for index in range(visual_dataset.__len__()):
                dataset_output = visual_dataset.__getitem__(index)
                print(dataset_output)
                extractor_output = cnn_feature_extractor.extract_feature(dataset_output)
                print(extractor_output)
        print(working_paths)

    # the following code will be customized and then replaced
    if config.has_config('items', 'textual'):
        working_paths = config.paths_for_extraction('items', 'textual')
        for model in config.get_models_list('items', 'textual'):
            print('here do')
    if config.has_config('interactions', 'textual'):
        working_paths = config.paths_for_extraction('interactions', 'textual')
        for model in config.get_models_list('interactions', 'textual'):
            print('here do')
    if config.has_config('interactions', 'visual'):
        working_paths = config.paths_for_extraction('interactions', 'visual')
        for model in config.get_models_list('interactions', 'visual'):
            print('here do')


def execute_extraction(origin_of_elaboration, type_of_extractions, config):
    # if there is the correct configuration
    # es items/interactions
    if config.has_config(origin_of_elaboration, type_of_extractions):
        # input/output path
        working_paths = config.paths_for_extraction(origin_of_elaboration, type_of_extractions)
        models = config.get_models_list(origin_of_elaboration, type_of_extractions)
        for model in models.keys():
            print(model)
            visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'], model_name=model)
            model_layer = models[model]['output_layers']
            cnn_feature_extractor = CnnFeatureExtractor(model, model_layer)
            for index in range(visual_dataset.__len__()):
                dataset_output = visual_dataset.__getitem__(index)
                print(dataset_output)
                extractor_output = cnn_feature_extractor.extract_feature(dataset_output)
                print(extractor_output)
        print(working_paths)


if __name__ == '__main__':
    foo()
