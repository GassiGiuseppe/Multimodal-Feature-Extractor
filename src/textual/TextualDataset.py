import re

from src.dataset.DatasetFather import DatasetFather


class TextualDataset(DatasetFather):

    def __init__(self, input_directory_path, output_directory_path, model_name):
        super().__init__(input_directory_path, output_directory_path, model_name)
        #here do preprocessing

        # does it need override?

    def __getitem__(self, index):
        print(index)

    def _pre_processing(self, sample):
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
        return sample.strip().lower()
