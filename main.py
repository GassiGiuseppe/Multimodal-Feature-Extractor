from src.runner.Runner import MultimodalFeatureExtractor


config_ls_of_dict = [{'dataset': 'D:\\dati\\uni\\tesi\\datasetFolder'},
                     {'gpu list': -1}, {'model map': 'modelMap.yml'},
                     {'visual': [{'items': [{'input': 'inputVisual'},
                                            {'output': 'outputVisual'},
                                            {'model': [{'name': 'ResNet50', 'output_layers': ['avg_pool', 'conv5_block3_out'], 'reshape': [224, 224], 'framework': ['tensorflow']},
                                                       {'name': 'AlexNet', 'output_layers': 5, 'reshape': [224, 224]},
                                                       {'name': 'ResNet50', 'output_layers': 3, 'reshape': [224, 224], 'framework': ['torch']}
                                                       ]
                                             }]
                                  },
                                 {'interactions': [{'input': None},
                                                   {'output': None},
                                                   {'model': []}
                                                   ]}
                                 ]
                      },
                     {'textual': [{'items': [{'input': 'inputTextual\\esempio.csv'},
                                             {'output': 'outputTextual'},
                                             {'model': [{'name': 'nlptown/bert-base-multilingual-uncased-sentiment', 'output_layers': 3, 'task': 'sentiment-analysis', 'clear_text': True}]
                                              }]
                                   },
                                  {'interactions': [{'input': None},
                                                    {'output': None},
                                                    {'model': []}
                                                    ]}
                                  ]
                      },
                     {'audio': [{'interactions': [{'input': 'inputAudio'},
                                                  {'output': 'outputAudio'},
                                                  {'model': [{'name': 'HUBERT_BASE', 'output_layers': 3, 'framework': 'torch'},
                                                             {'name': 'facebook/wav2vec2-base-960h', 'output_layers': 3, 'framework': 'transformers'}
                                                             ]}
                                                  ]
                                 }]
                      }]

# extractor_obj = MultimodalFeatureExtractor(config_file_path='./config/config.yml')
extractor_obj = MultimodalFeatureExtractor(command_as_ls_of_dict=config_ls_of_dict)
extractor_obj.execute_extractions()
