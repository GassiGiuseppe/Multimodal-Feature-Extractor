from src.runner.Runner import MultimodalFeatureExtractor

config_ls_of_dict = {'dataset': 'D:\\dati\\uni\\tesi\\datasetFolder',
                     'gpu list': -1, 'model map': 'modelMap.yml',
                     'visual':
                         {'items':
                              {'input': 'inputVisual',
                               'output': 'outputVisual',
                               'model': [{'name': 'ResNet50', 'output_layers': ['avg_pool', 'conv5_block3_out'],
                                          'reshape': [224, 224], 'framework': ['tensorflow']},
                                         {'name': 'AlexNet', 'output_layers': 5, 'reshape': [224, 224]},
                                         {'name': 'ResNet50', 'output_layers': 3, 'reshape': [224, 224],
                                          'framework': ['torch']}
                                         ]
                               },
                          'interactions':
                              {'input': None,
                               'output': None,
                               'model': []
                               }
                          },
                     'textual':
                         {'items':
                              {'input': 'inputTextual\\esempio.csv',
                               'output': 'outputTextual',
                               'model': [
                                   {'name': 'nlptown/bert-base-multilingual-uncased-sentiment', 'output_layers': 3,
                                    'task': 'sentiment-analysis', 'clear_text': True}]
                               },
                          'interactions':
                              {'input': None,
                               'output': None,
                               'model': []
                               }
                          },
                     'audio':
                         {'interactions':
                              {'input': 'inputAudio',
                               'output': 'outputAudio',
                               'model': [{'name': 'HUBERT_BASE', 'output_layers': 3, 'framework': 'torch'},
                                         {'name': 'facebook/wav2vec2-base-960h', 'output_layers': 3,
                                          'framework': 'transformers'}
                                         ]
                               }
                          }
                     }


# extractor_obj = MultimodalFeatureExtractor(config_file_path='./config/config.yml')
extractor_obj = MultimodalFeatureExtractor(command_as_ls_of_dict=config_ls_of_dict)
extractor_obj.execute_extractions()


def update_dict(dict_to_modify, keys_as_string, value):
    ls_of_keys = keys_as_string.split('.')
    first_key = ls_of_keys.pop(0)
    new_value = sub_of_update_dict(ls_of_keys, value, dict_to_modify[first_key])
    dict_to_modify.update({first_key: new_value})
    return dict_to_modify


def sub_of_update_dict(ls_of_keys, last_value, sub_dict):
    if len(ls_of_keys) == 1:
        sub_dict.update({ls_of_keys.pop(0): last_value})
        return sub_dict
    else:
        key_to_use = ls_of_keys.pop(0)
        sub_dict.update({key_to_use: sub_of_update_dict(ls_of_keys, last_value, sub_dict[key_to_use])})
        return sub_dict


# update_dict(config_ls_of_dict, 'visual.items.input', 'ciaone')
# print('ciao')