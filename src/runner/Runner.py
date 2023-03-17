import os
import logging
from tqdm import tqdm

from src.config.Config import Config
from src.visual.VisualDataset import VisualDataset
from src.textual.TextualDataset import TextualDataset
from src.visual.VisualCnnFeatureExtractor import VisualCnnFeatureExtractor
from src.textual.TextualCnnFeatureExtractor import TextualCnnFeatureExtractor


def _execute_extraction_from_models_list(models, extractor, dataset, modality_type):
    for model in models:
        logging.info(' Now using model: %s', str(model['name']))

        # set framework
        extractor.set_framework(model['framework'])
        dataset.set_framework(model['framework'])
        # set model
        extractor.set_model(model['name'])
        dataset.set_model(model['name'])
        # set reshape
        if modality_type == 'visual':
            dataset.set_reshape(model['reshape'])
        elif modality_type == 'textual':
            dataset.set_clean_flag(model['clear_text'])
        # execute extractions
        for model_layer in model['output_layers']:

            logging.info(' Now using layer: - %s', str(model_layer))

            # set output layer
            extractor.set_output_layer(model_layer)

            with tqdm(total=dataset.__len__()) as t:
                # for evey image do the extraction
                for index in range(dataset.__len__()):
                    # retrieve the item (preprocessed) from dataset
                    preprocessed_item = dataset.__getitem__(index)
                    # do the extraction
                    extractor_output = extractor.extract_feature(preprocessed_item)
                    # create the npy file with the extraction output
                    dataset.create_output_file(index, extractor_output, model_layer)
                    # update the progress bar
                    t.update()


class MultimodalFeatureExtractor:

    def __init__(self, config_file_path=r'./config/config.yml'):
        self._config = Config(config_file_path)
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        # set gpu to use
        os.environ['CUDA_VISIBLE_DEVICES'] = self._config.get_gpu()

    def execute_extraction(self):
        self.do_item_visual_extractions()
        self.do_interaction_visual_extractions()
        self.do_item_textual_extractions()
        self.do_interaction_textual_extractions()

    def do_item_visual_extractions(self):
        if self._config.has_config('items', 'visual'):
            logging.info(' Config for visual extractions from items detected, the extraction is going to start ...')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'visual')
            models = self._config.get_models_list('items', 'visual')
            # generate dataset and extractor
            visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])
            cnn_feature_extractor = VisualCnnFeatureExtractor(self._config.get_gpu())

            logging.info(' Working environment created')
            logging.info(' Number of models to use: %s', str(models.__len__()))
            _execute_extraction_from_models_list(models, cnn_feature_extractor, visual_dataset, 'visual')

    def do_item_textual_extractions(self):
        if self._config.has_config('items', 'textual'):
            logging.info(' Config for textual extractions from items detected, the extraction is going to start ...')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'textual')
            models = self._config.get_models_list('items', 'textual')
            # generate dataset and extractor
            textual_dataset = TextualDataset(working_paths['input_path'], working_paths['output_path'])
            cnn_feature_extractor = TextualCnnFeatureExtractor(self._config.get_gpu())

            logging.info(' Working environment created')
            logging.info(' Number of models to use: %s', str(models.__len__()))

            _execute_extraction_from_models_list(models, cnn_feature_extractor, textual_dataset, 'textual')

    def do_interaction_visual_extractions(self):
        if self._config.has_config('interactions', 'visual'):
            logging.info(
                ' Config for visual extractions from interactions detected, the extraction is going to start ...')

            # get paths and models
            working_paths = self._config.paths_for_extraction('interactions', 'visual')
            models = self._config.get_models_list('interactions', 'visual')
            # generate dataset and extractor
            visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])
            cnn_feature_extractor = VisualCnnFeatureExtractor(self._config.get_gpu())

            logging.info(' Working environment created')
            logging.info(' Number of models to use: %s', str(models.__len__()))
            _execute_extraction_from_models_list(models, cnn_feature_extractor, visual_dataset, 'visual')

    def do_interaction_textual_extractions(self):
        if self._config.has_config('interactions', 'textual'):
            logging.info(' Config for textual extractions from items detected, the extraction is going to start ...')

            # get paths and models
            working_paths = self._config.paths_for_extraction('interactions', 'textual')
            models = self._config.get_models_list('interactions', 'textual')
            # generate dataset and extractor
            textual_dataset = TextualDataset(working_paths['input_path'], working_paths['output_path'])
            cnn_feature_extractor = TextualCnnFeatureExtractor(self._config.get_gpu())

            logging.info(' Working environment created')
            logging.info(' Number of models to use: %s', str(models.__len__()))

            _execute_extraction_from_models_list(models, cnn_feature_extractor, textual_dataset, 'textual')



