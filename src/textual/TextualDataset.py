import os
import re

from src.dataset.DatasetFather import DatasetFather


def complex_spit_of_list_of_string(sample, splitter):
    sample_list = []
    for el in sample:
        temp = el.split(splitter)
        for sentance in temp[:-1]:
            sentance = sentance + splitter
            sample_list.append(sentance)
        # now append the last that was excluded in the for each
        sample_list.append(temp[-1])
    return sample_list


class TextualDataset(DatasetFather):

    def __init__(self, input_directory_path, output_directory_path):
        super().__init__(input_directory_path, output_directory_path, model_name=None)
        self._text_to_be_cleaned = True


    def __getitem__(self, index):
        image_path = os.path.join(self._input_directory_path, self._filenames[index])
        with open(image_path, 'r') as f:
            sample = f.read()
        return self._pre_processing(sample)

    def set_clean_text_flag(self, text_to_be_cleaned):
        self._text_to_be_cleaned = text_to_be_cleaned

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

        # now using the Tokenizer theory a list of string is needed, a string for each sentence
        # split for [. , ! \n]
        sample_list = []
        #sample = sample.split('\n')
        #sample = complex_spit_of_list_of_string(sample, '.')
        #sample = complex_spit_of_list_of_string(sample, '!')
        #sample = complex_spit_of_list_of_string(sample, '?')
        for element in sample:
            if element is None:
                print('problema')

        return sample


