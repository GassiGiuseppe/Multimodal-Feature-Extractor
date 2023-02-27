import os

from config.Config import Config
from visual.VisualDataset import VisualDataset
from visual.models.CnnFeatureExtractor import CnnFeatureExtractor


def foo():
    config = Config()

    # set gpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get_gpu()

    if config.has_config('items', 'visual'):
        # get paths and models
        working_paths = config.paths_for_extraction('items', 'visual')
        models = config.get_models_list('items', 'visual')
        # generate dataset and extractor
        visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])
        cnn_feature_extractor = CnnFeatureExtractor(config.get_gpu())
        for model in models.keys():
            # set framework
            cnn_feature_extractor.set_framework(models[model]['framework'])
            visual_dataset.set_framework(models[model]['framework'])
            # set model
            cnn_feature_extractor.set_model(model)
            visual_dataset.set_model(model)
            # set reshape
            visual_dataset.set_reshape(models[model]['reshape'])
            for model_layer in models[model]['output_layers']:
                # set output layer
                cnn_feature_extractor.set_output_layer(model_layer)
                for index in range(visual_dataset.__len__()):
                    # for evey image do the extraction
                    # retrieve the image from dataset
                    adjusted_item = visual_dataset.__getitem__(index)
                    # do the extraction
                    extractor_output = cnn_feature_extractor.extract_feature(adjusted_item)
                    # create the npy file with the extraction output
                    visual_dataset.create_output_file(index, extractor_output, model_layer)

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


if __name__ == '__main__':
    foo()
