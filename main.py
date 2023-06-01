from src.runner.Runner import MultimodalFeatureExtractor


config_ls_of_dict = {'dataset': './data/prova',
                     'gpu list': -1, 'model map': 'modelMap.yml',
                     'visual':
                         {'items':
                              {'input': 'images',
                               'output': 'output',
                               'model': [{'name': 'ResNet50', 'output_layers': ['avg_pool', 'conv5_block3_out'], 'reshape': [224, 224], 'framework': ['tensorflow']},
                                         {'name': 'AlexNet', 'output_layers': 5, 'reshape': [224, 224]},
                                         {'name': 'ResNet50', 'output_layers': 3, 'reshape': [224, 224], 'framework': ['torch']}
                                         ]
                               }
                          }
                     }
# extractor_obj = MultimodalFeatureExtractor(config_file_path='./config/config.yml')
extractor_obj = MultimodalFeatureExtractor(command_as_ls_of_dict=config_ls_of_dict)
extractor_obj.execute_extractions()
