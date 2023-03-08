import yaml
import os


class Config:

    def __init__(self, config_file_path=r'../config/config.yml'):
        # both absolute and relative path are fine
        self._data_dict = None
        self.__find_yaml_file_path(config_file_path)
        self.__load_config_from_file()

    def __find_yaml_file_path(self, old_path):
        """
        if old_path links to a directory the method search a 'yaml' file in the directory. Otherwise, if it poinst to a
        file, all is fine. Else the method try to correct the path in a working one, if it fails raise an error
        Args:
            old_path: the path given from the user, here starts the search for the file

        Returns:
            it returns nothing but set the __config_file_path that points directly to the yaml file

        """
        # the path can be:
        # - a path only to the directory
        # - a complete path to a yml/yaml, in this case must be verified that the extension is correct
        if os.path.isdir(old_path):
            # search through the directory a file with the correct extension
            dir_list = os.listdir(old_path)
            for file in dir_list:
                # the extensions can be both .yml or .yaml
                if file[-4:] == '.yml' or file[-5:] == '.yaml':
                    self.__config_file_path = os.path.join(old_path, file)
                    return
        elif os.path.exists(old_path):
            # the path points directly to the file, all is fine
            self.__config_file_path = old_path
        else:
            # in this case an error has occurred, thanks to the 2 possible extension
            # maybe the user wrote .yml but the correct extension is .yaml or the opposite
            if os.path.exists(old_path[-3:] + 'yaml'):
                self.__config_file_path = old_path[-3:] + 'yaml'
            elif os.path.exists(old_path[-4:] + 'yml'):
                self.__config_file_path = old_path[-4:] + 'yml'
            else:
                # it is impossible to find the config file
                raise FileNotFoundError('the path given is wrong: ' + old_path)

    def __load_config_from_file(self):
        """
            it simply loads the data contained in the file and call the method __clean_dict on it
        """
        # there is no need here to raise an exception if the file is not found
        # since the os raises it autonomously
        with open(self.__config_file_path, 'r') as file:
            data = yaml.safe_load(file)
        self._data_dict = self.__clean_dict(data)

    def __clean_dict(self, data):
        """
        It crosses in every element of the dict in search of a list of dict to transfrom in a big dict:
        if there is a dict, it crosses every value (recalling this method).
        If there is a list, it crosses every item (recalling this method). then if the items are dicts the list
        is swapped with a big dict
        Args:
            data: it's the data contained in the yaml file as a dict

        Returns:
            data: it returns data cleaned, every list of dict is transformed in a single dict

        """
        # using yaml there is a problem:
        # it has no strict rules, so you can have [[{}]] [[]] {[]} {{}} ecc
        # this recursive method transform everything as {...{}...} or {...[]...}
        temp_dict = {}
        if isinstance(data, dict):
            for key in data.keys():
                # the model dict follow a particular configuration that is necessary not to change
                if key != 'model':
                    value = self.__clean_dict(data[key])
                    data.update({key: value})
        if isinstance(data, list):
            for element in data:
                element = self.__clean_dict(element)
                # the following code follow a statement that is always true using yaml:
                # if in the list one element is a dict, so are all the others elements
                if isinstance(element, dict):
                    temp_dict.update(element)
        if bool(temp_dict):
            data = temp_dict
        return data

    def get_gpu(self):
        """

        Returns: the gpu list as a string

        """
        # if there is not a gpu config then "-1" (use cpu only)
        # otherwise return the config
        if 'gpu list' in self._data_dict:
            gpu_list = self._data_dict['gpu list']
            if isinstance(gpu_list, str):
                # es '1' or '1,2'
                return gpu_list
            elif isinstance(gpu_list, int):
                # es 1 -> '1'
                return str(gpu_list)
            elif isinstance(gpu_list, list):
                # es [1,3] -> '1,3'
                return ','.join(str(x) for x in gpu_list)
            else:
                raise SyntaxError('the gpu list is written in a incorrect way')
        else:
            return '-1'

    def has_config(self, origin_of_elaboration, type_of_extractions):
        """
        Search the config in the data dicts then check that this config have values in it
        Args:
            origin_of_elaboration: 'items' or 'interactions'
            type_of_extractions: 'textual' or 'visual'

        Returns: Bool True/False if contains the configuration

        """
        # example of origin_of_elaboration: 'items', 'interactions'
        # example of type_of_extractions: 'textual', 'visual'
        if origin_of_elaboration in self._data_dict and type_of_extractions in self._data_dict[origin_of_elaboration]:
            local_dict = self._data_dict[origin_of_elaboration][type_of_extractions]
            # check if local dict has input/output/model
            if 'input' in local_dict and 'output' in local_dict and 'model' in local_dict:
                # in this case it's all right but must be checked that the values are not empty
                input_value = local_dict['input']
                output_value = local_dict['output']
                model_value = local_dict['model']
                if input_value is not None and output_value is not None and model_value is not None:
                    return True
        return False

    def paths_for_extraction(self, origin_of_elaboration, type_of_extraction):
        """

        Args:
            origin_of_elaboration: 'items' or 'interactions'
            type_of_extraction: 'textual' or 'visual'

        Returns: a dict as { 'input_path': input path, 'output_path': output_path }

        """
        # {'input_path': ///, 'output_path': ///}
        relative_input_path = self._data_dict[origin_of_elaboration][type_of_extraction]['input']
        relative_output_path = self._data_dict[origin_of_elaboration][type_of_extraction]['output']

        return {
            'input_path': os.path.join(self._data_dict['dataset'], relative_input_path),
            'output_path': os.path.join(self._data_dict['dataset'], relative_output_path)}

    def get_models_list(self, origin_of_elaboration, type_of_extractions):
        """

        Args:
            origin_of_elaboration: 'items' or 'interactions'
            type_of_extractions: 'textual' or 'visual'

        Returns: a dict of the models, every model is a dict with 'output_layers': the layers of extraction,
        'reshape': height, width as pixel to reshape, 'framework': framework to work with tensorflow or torch

        """

        models = self._data_dict[origin_of_elaboration][type_of_extractions]['model']

        for model in models:

            # output_layers has to be a list
            if not isinstance(model['output_layers'], list):
                model.update({'output_layers': [model['output_layers']]})

            # Framework elaboration
            # - if INPUT FRAMEWORK is ['tensorflow', 'torch'] then two different model dicts will be added to the list,
            #   each one identical to the other except for the fact that it contains only one of the 2 type of framework
            # - if OUTPUT FRAMEWORK is ['tensorflow', 'torch'] then outside of this method it means that
            #   the framework in which operate is not known but only one of them will be executed
            if 'framework' in model.keys():
                framework_value = model['framework']

                if framework_value == ['tensorflow', 'torch']:

                    first_model = model
                    first_model.update({'framework': ['tensorflow']})

                    second_model = model
                    second_model.update({'framework': ['torch']})

                    # layers

                    first_model_layers = []
                    second_model_layers = []
                    for layer in model['output_layers']:
                        if isinstance(layer, int):
                            second_model_layers.append(layer)
                        else:
                            first_model_layers.append(layer)

                    first_model.update({'output_layers': first_model_layers})
                    second_model.update({'output_layers': second_model_layers})

                    # models_list.append(second_model)
                    # models_list.append(first_model)

                    # this setting does not work properly because the two framework uses calls different layers
                    raise ValueError(' unfortunately calling both framework simultaneity doesnt work')
                # framework value must be a list
                elif framework_value == 'tensorflow' or framework_value == 'torch':
                    model.update({'framework': [framework_value]})

                elif framework_value != ['tensorflow'] and framework_value != ['torch']:

                    raise ValueError('the framework tag in the yaml file is not written correctly')
            else:
                # the framework is not set, it is not know in which one operate, so both are set as plausible
                model.update({'framework': ['tensorflow', 'torch']})

        return models

    def get_models_list_from_dict(self, origin_of_elaboration, type_of_extractions):
        """

        Args:
            origin_of_elaboration: 'items' or 'interactions'
            type_of_extractions: 'textual' or 'visual'

        Returns: a dict of the models, every model is a dict with 'output_layers': the layers of extraction,
        'reshape': height, width as pixel to reshape, 'framework': framework to work with tensorflow or torch

        """

        # new plan: now each model is a element of a list
        models = self._data_dict[origin_of_elaboration][type_of_extractions]['model']

        # transform model from dict of dicts to list of dicts
        models_list = []
        for model_name, model_dict in models.items():
            # it is now created a temp single model dict that in the end will be appended to the list

            # add to the single dict the tag model name
            model_dict.update({'name': model_name})

            # output_layers has to be a list
            if not isinstance(model_dict['output_layers'], list):
                # then it may be a str or an int, transform in a list and go on
                model_dict.update({'output_layers': [model_dict['output_layers']]})

            # Framework elaboration
            # - if INPUT FRAMEWORK is ['tensorflow', 'torch'] then two different model dicts will be added to the list,
            #   each one identical to the other except for the fact that it contains only one of the 2 type of framework
            # - if OUTPUT FRAMEWORK is ['tensorflow', 'torch'] then outside of this method it means that
            #   the framework in which operate is not known but only one of them will be executed
            if 'framework' in model_dict.keys():
                framework_value = model_dict['framework']

                if framework_value == ['tensorflow', 'torch']:

                    first_model = model_dict
                    first_model.update({'framework': ['tensorflow']})
                    # models_list.append(first_model)

                    second_model = model_dict
                    second_model.update({'framework': ['torch']})
                    # models_list.append(second_model)

                    # this setting does not work properly because the two framework uses calls different layers
                    raise ValueError(' unfortunately calling both framework simultaneity doesnt work')

                elif framework_value == ['tensorflow'] or framework_value == ['torch']:
                    models_list.append(model_dict)

                # framework value must be a list
                elif framework_value == 'tensorflow' or framework_value == 'torch':
                    model_dict.update({'framework': [framework_value]})
                    models_list.append(model_dict)
                else:
                    raise ValueError('the framework tag in the yaml file is not written correctly')
            else:
                # the framework is not set, it is not know in which one operate, so both are set as plausible
                model_dict.update({'framework': ['tensorflow', 'torch']})
                models_list.append(model_dict)

        return models_list

    def get_dict(self):
        return self._data_dict
