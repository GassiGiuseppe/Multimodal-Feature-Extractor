import os
import logging
from config.Config import Config
from visual.VisualDataset import VisualDataset
from visual.models.CnnFeatureExtractor import CnnFeatureExtractor


def foo():
    config = Config()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # set gpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get_gpu()

    do_item_visual_extractions(config)
    do_interaction_visual_extractions(config)

    # the following code will be customized and then replaced
    if config.has_config('items', 'textual'):
        working_paths = config.paths_for_extraction('items', 'textual')
        for model in config.get_models_list('items', 'textual'):
            print('here do')
    if config.has_config('interactions', 'textual'):
        working_paths = config.paths_for_extraction('interactions', 'textual')
        for model in config.get_models_list('interactions', 'textual'):
            print('here do')


def do_item_visual_extractions(config):
    if config.has_config('items', 'visual'):

        logging.info(' Config for visual extractions from items detected, the extraction is going to start ...')

        # get paths and models
        working_paths = config.paths_for_extraction('items', 'visual')
        models = config.get_models_list('items', 'visual')
        # generate dataset and extractor
        visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])
        cnn_feature_extractor = CnnFeatureExtractor(config.get_gpu())

        logging.info(' Working environment created')
        logging.info(' Number of models to use: %s', str(models.__len__()))

        for model in models:

            logging.info(' Now using model: %s', str(model['name']))

            # set framework
            cnn_feature_extractor.set_framework(model['framework'])
            visual_dataset.set_framework(model['framework'])
            # set model
            cnn_feature_extractor.set_model(model['name'])
            visual_dataset.set_model(model['name'])
            # set reshape
            visual_dataset.set_reshape(model['reshape'])
            for model_layer in model['output_layers']:

                logging.info(' Now using layer: - %s', str(model_layer))

                # set output layer
                cnn_feature_extractor.set_output_layer(model_layer)

                logging.info(' Num images :       %s', visual_dataset.__len__())
                for index in range(visual_dataset.__len__()):
                    logging.info(' Now elaborating:   %s', visual_dataset.get_image_name(index))
                    # for evey image do the extraction
                    # retrieve the image from dataset
                    adjusted_item = visual_dataset.__getitem__(index)
                    # do the extraction
                    extractor_output = cnn_feature_extractor.extract_feature(adjusted_item)
                    # create the npy file with the extraction output
                    visual_dataset.create_output_file(index, extractor_output, model_layer)


def do_interaction_visual_extractions(config):
    if config.has_config('interactions', 'visual'):

        logging.info(' Config for visual extractions from interactions detected, the extraction is going to start ...')

        # get paths and models
        working_paths = config.paths_for_extraction('interactions', 'visual')
        models = config.get_models_list('interactions', 'visual')
        # generate dataset and extractor
        visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])
        cnn_feature_extractor = CnnFeatureExtractor(config.get_gpu())

        logging.info(' Working environment created')
        logging.info(' Number of models to use: %s', str(models.__len__()))

        for model in models:

            logging.info(' Now using model: %s', str(model['name']))

            # set framework
            cnn_feature_extractor.set_framework(model['framework'])
            visual_dataset.set_framework(model['framework'])
            # set model
            cnn_feature_extractor.set_model(model['name'])
            visual_dataset.set_model(model['name'])
            # set reshape
            visual_dataset.set_reshape(model['reshape'])
            for model_layer in model['output_layers']:

                logging.info(' Now using layer: - %s', str(model_layer))

                # set output layer
                cnn_feature_extractor.set_output_layer(model_layer)

                logging.info(' Num images :       %s', visual_dataset.__len__())
                for index in range(visual_dataset.__len__()):
                    logging.info(' Now elaborating:   %s', visual_dataset.get_image_name(index))
                    # for evey image do the extraction
                    # retrieve the image from dataset
                    adjusted_item = visual_dataset.__getitem__(index)
                    # do the extraction
                    extractor_output = cnn_feature_extractor.extract_feature(adjusted_item)
                    # create the npy file with the extraction output
                    visual_dataset.create_output_file(index, extractor_output, model_layer)


if __name__ == '__main__':
    foo()
