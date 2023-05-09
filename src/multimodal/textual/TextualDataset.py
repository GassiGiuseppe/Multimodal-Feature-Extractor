import os
import re
import numpy
from src.internal.father_classes.DatasetFather import DatasetFather
from src.internal.utils.TextualFileManager import TextualFileManager


# the following function is not called right now. but it will be needed in the future
def complex_spit_of_list_of_string(sample, splitter):
    sample_list = []
    for el in sample:
        temp = el.split(splitter)
        for sentence in temp[:-1]:
            sentence = sentence + splitter
            sample_list.append(sentence)
        # now append the last that was excluded in the for each
        sample_list.append(temp[-1])
    return sample_list


class TextualDataset(DatasetFather):

    def __init__(self, input_directory_path, output_directory_path):
        super().__init__(input_directory_path, output_directory_path, model_name=None)
        self._text_to_be_cleaned = True
        self._textual_file_manager = TextualFileManager()
        # if num_sample is 1, it means it have to be the num of sample in the single file
        # in this case the textual file manager have to behave accordingly
        if self._num_samples == 1:
            self._prepare_environment_for_single_file_extractions()

    def _prepare_environment_for_single_file_extractions(self):
        if self._filenames[0] == '':
            file_path = self._input_directory_path
        else:
            file_path = os.path.join(self._input_directory_path, self._filenames[0])
        self._textual_file_manager.set_file_path(file_path)
        self._num_samples = self._textual_file_manager.initiate_element_list_and_get_len()

    def __getitem__(self, index):
        """
        Args:
            index: is the index in the filenames list from which extract the name of te file to elaborate
        Returns: a String which contains the data of the file. It may be processed and cleaned
        """
        '''
        if self._filenames[index] == '':
            file_path = self._input_directory_path
        else:
            file_path = os.path.join(self._input_directory_path, self._filenames[index])
        self._textual_file_manager.set_file_path(file_path)
        element_list = self._textual_file_manager.get_element_list()
        preprocessed_list = []
        for el in element_list:
            preprocessed_list.append(self._pre_processing(el))
        return preprocessed_list
        '''
        return self._textual_file_manager.get_item_from_id(index)

    def _pre_processing(self, sample):
        if self._text_to_be_cleaned:
            sample = re.sub(r"[^A-Za-z0-9',.!;?()]", " ", sample)

            sample = re.sub(r"\.", " . ", sample)
            sample = re.sub(r"!+", " ! ", sample)
            sample = re.sub(r",", " , ", sample)
            sample = re.sub(r";", " ; ", sample)
            sample = re.sub(r"\\", " \\ ", sample)
            sample = re.sub(r"!", " ! ", sample)
            sample = re.sub(r"\(", " ( ", sample)
            sample = re.sub(r"\)", " ) ", sample)
            sample = re.sub(r"\?", " ? ", sample)

            sample = re.sub(r"\s{2,}", " ", sample)
            sample = re.sub(r"(\.|\s){7,}", " ... ", sample)
            sample = re.sub(r"(?<= )(\w \. )+(\w \.)", lambda x: x.group().replace(" ", ""), sample)
            # sample = re.sub(r"(\.|\s){4,}", " ... ", sample)

            sample = re.sub(r"\'s", " \'s", sample)
            sample = re.sub(r"\'ve", " \'ve", sample)
            sample = re.sub(r"n\'t", " n\'t", sample)
            sample = re.sub(r"\'re", " \'re", sample)
            sample = re.sub(r"\'d", " \'d", sample)
            sample = re.sub(r"\'m", " \'m", sample)
            sample = re.sub(r"\'ll", " \'ll", sample)

            # sample = re.sub(r"[^A-Za-z0-9']", " ", sample)
            sample = re.sub(
                r"(?!(('(?=s\b))|('(?=ve\b))|('(?=re\b))|('(?=d\b))|('(?=ll\b))|('(?=m\b))|((?<=n\b)'(?=t\b))))'",
                " ", sample)

            # Glove style
            # sample = re.sub(' [0-9]{5,} ', ' ##### ', sample)
            # sample = re.sub(' [0-9]{4} ', ' #### ', sample)
            # sample = re.sub(' [0-9]{3} ', ' ### ', sample)
            # sample = re.sub(' [0-9]{2} ', ' ## ', sample)
            sample = re.sub(' 0 ', ' zero ', sample)
            sample = re.sub(' 1 ', ' one ', sample)
            sample = re.sub(' 2 ', ' two ', sample)
            sample = re.sub(' 3 ', ' three ', sample)
            sample = re.sub(' 4 ', ' four ', sample)
            sample = re.sub(' 5 ', ' five ', sample)
            sample = re.sub(' 6 ', ' six ', sample)
            sample = re.sub(' 7 ', ' seven ', sample)
            sample = re.sub(' 8 ', ' eight ', sample)
            sample = re.sub(' 9 ', ' nine ', sample)

            sample = re.sub(r"\s{2,}", " ", sample)
            sample.strip().lower()

        return sample

    def set_clean_flag(self, text_to_be_cleaned):
        """
        Args:
            text_to_be_cleaned: flag True/False if the text will be preprocessed and cleaned

        Returns: nothing
        """
        self._text_to_be_cleaned = text_to_be_cleaned

    def set_type_of_extraction(self, type_of_extraction):
        self._textual_file_manager.set_type_of_extraction(type_of_extraction)

    def create_output_file(self, index, extracted_data, model_layer):
        '''
        if isinstance(extracted_data, list):
            # index is to indicate the element from the list of all the elelment in the input folder
            # idx indicate the element in the same file (csv)
            for idx, el in enumerate(extracted_data):
                # generate file name
                input_file_name = self._filenames[index].split('.')[0] + self._textual_file_manager.build_path_from_id(
                    idx)
                output_file_name = input_file_name + '.npy'

                # generate output path
                framework = self._framework_list[0]
                output_path = os.path.join(self._output_directory_path, framework)
                output_path = os.path.join(output_path, self._model_name)
                output_path = os.path.join(output_path, str(model_layer))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # create file
                path = os.path.join(output_path, output_file_name)
                numpy.save(path, el)
        else:
            super().create_output_file(index, extracted_data, model_layer)
        '''
        # generate file name
        input_file_name = self._filenames[0].split('.')[0] + self._textual_file_manager.build_path_from_id(
            index)
        output_file_name = input_file_name + '.npy'

        # generate output path
        framework = self._framework_list[0]
        output_path = os.path.join(self._output_directory_path, framework)
        output_path = os.path.join(output_path, self._model_name)
        output_path = os.path.join(output_path, str(model_layer))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # create file
        path = os.path.join(output_path, output_file_name)
        numpy.save(path, extracted_data)
