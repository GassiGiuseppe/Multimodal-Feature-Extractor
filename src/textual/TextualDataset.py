import os
import re

from src.internal.father_classes.DatasetFather import DatasetFather


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

    def __getitem__(self, index):
        """
        Args:
            index: is the index in the filenames list from which extract the name of te file to elaborate
        Returns: a String which contains the data of the file. It may be processed and cleaned
        """
        image_path = os.path.join(self._input_directory_path, self._filenames[index])
        with open(image_path, 'r') as f:
            sample = f.read()
        return self._pre_processing(sample)

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
