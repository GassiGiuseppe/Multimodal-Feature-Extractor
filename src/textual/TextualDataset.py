import re

from src.dataset.DatasetFather import DatasetFather


class TextualDataset(DatasetFather):

    def __init__(self, input_directory_path, output_directory_path, model_name):
        super().__init__(input_directory_path, output_directory_path, model_name)
        #here do preprocessing

        # does it need override?

    def __getitem__(self, index):
        print(index)

    def _pre_processing(self, string):
        string = re.sub(r"[^A-Za-z0-9',.!;?()]", " ", string)

        string = re.sub(r"\.", " . ", string)
        string = re.sub(r"!+", " ! ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"\\", " \\ ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)

        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"(\.|\s){7,}", " ... ", string)
        string = re.sub(r"(?<= )(\w \. )+(\w \.)", lambda x: x.group().replace(" ", ""), string)
        # string = re.sub(r"(\.|\s){4,}", " ... ", string)

        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'m", " \'m", string)
        string = re.sub(r"\'ll", " \'ll", string)

        # string = re.sub(r"[^A-Za-z0-9']", " ", string)
        string = re.sub(
            r"(?!(('(?=s\b))|('(?=ve\b))|('(?=re\b))|('(?=d\b))|('(?=ll\b))|('(?=m\b))|((?<=n\b)'(?=t\b))))'",
            " ", string)

        # Glove style
        # string = re.sub(' [0-9]{5,} ', ' ##### ', string)
        # string = re.sub(' [0-9]{4} ', ' #### ', string)
        # string = re.sub(' [0-9]{3} ', ' ### ', string)
        # string = re.sub(' [0-9]{2} ', ' ## ', string)
        string = re.sub(' 0 ', ' zero ', string)
        string = re.sub(' 1 ', ' one ', string)
        string = re.sub(' 2 ', ' two ', string)
        string = re.sub(' 3 ', ' three ', string)
        string = re.sub(' 4 ', ' four ', string)
        string = re.sub(' 5 ', ' five ', string)
        string = re.sub(' 6 ', ' six ', string)
        string = re.sub(' 7 ', ' seven ', string)
        string = re.sub(' 8 ', ' eight ', string)
        string = re.sub(' 9 ', ' nine ', string)

        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
