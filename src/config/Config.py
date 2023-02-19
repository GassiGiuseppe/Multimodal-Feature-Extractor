import yaml
import os


class Config:

    def __init__(self, config_file_path=r'../config/config.yml'):
        # both absolute and relative path are fine
        self.__data_dict = None
        self.__find_yaml_file_path(config_file_path)
        self.__load_config_from_file()

    def __find_yaml_file_path(self, old_path):
        # the path can be:
        # a path only to the directory
        # a complete path to a yml/yaml, in this case must be verified that the extension is correct
        if os.path.isdir(old_path):
            dir_list = os.listdir(old_path)
            for file in dir_list:
                if file[-4:] == '.yml' or file[-5:] == '.yaml':
                    self.__config_file_path = os.path.join(old_path, file)
                    return
        elif os.path.exists(old_path):
            # the path points directly to the file, all is fine
            self.__config_file_path = old_path
        else:
            # in this case an error has occurred, thanks to the 2 possible extension
            if os.path.exists(old_path[-3:] + 'yaml'):
                self.__config_file_path = old_path[-3:] + 'yaml'
            elif os.path.exists(old_path[-4:] + 'yml'):
                self.__config_file_path = old_path[-4:] + 'yml'
            else:
                # it is impossible to find the config file
                raise FileNotFoundError('the path given is wrong: ' + old_path)

    def __load_config_from_file(self):
        # there is no need here to raise an exception if the file is not found
        # since the os raises it autonomously
        with open(self.__config_file_path, 'r') as file:
            data = yaml.safe_load(file)
        self.__data_dict = self.__clean_dict(data)

    def __clean_dict(self, data):
        # using yaml there is a problem:
        # it has no strict rules, so you can have [[{}]] [[]] {[]} {{}} ecc
        # this recursive method transform everything as {...{}...} or {...[]...}
        temp_dict = {}
        if isinstance(data, dict):
            for key in data.keys():
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
        # if there is not a gpu config then "-1" (use cpu only)
        # otherwise return the config
        if 'gpu list' in self.__data_dict:
            gpu_list = self.__data_dict['gpu list']
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
        # example of origin_of_elaboration: 'items', 'interactions'
        # example of type_of_extractions: 'textual', 'visual'
        if origin_of_elaboration in self.__data_dict:
            if type_of_extractions in self.__data_dict[origin_of_elaboration]['input'] and \
                    type_of_extractions in self.__data_dict[origin_of_elaboration]['output']:
                # in this case it's all right but must be checked that the values are not empty
                input_value = self.__data_dict[origin_of_elaboration]['input'][type_of_extractions]
                output_value = self.__data_dict[origin_of_elaboration]['output'][type_of_extractions]
                if input_value is not None and output_value is not None:

                    return True
        return False

    def paths_for_extraction(self, origin_of_elaboration, type_of_extraction):
        # {'input_path': ///, 'output_path': ///}
        relative_input_path = self.__data_dict[origin_of_elaboration]['input'][type_of_extraction]
        relative_output_path = self.__data_dict[origin_of_elaboration]['output'][type_of_extraction]

        return {
            'input_path': os.path.join(self.__data_dict['dataset'], relative_input_path),
            'output_path': os.path.join(self.__data_dict['dataset'], relative_output_path)}

    def get_models_list(self, origin_of_elaboration, type_of_extractions):
        # example of origin_of_elaboration: 'items', 'interactions'
        # example of type_of_extractions: 'textual', 'visual'
        models = self.__data_dict[origin_of_elaboration]['model'][type_of_extractions]
        if isinstance(models, str):
            return [models]
        else:
            return models

    def get_dict(self):
        return self.__data_dict
