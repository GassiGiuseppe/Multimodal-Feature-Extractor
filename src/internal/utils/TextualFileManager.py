import pandas
import csv


class TextualFileManager:
    def __init__(self, ):
        self._internal_list = None
        self._type_of_extraction = None
        self._file_path = None
        return

    def set_type_of_extraction(self, type_of_extraction):
        # interaction/item
        self._type_of_extraction = type_of_extraction

    def set_file_path(self, file_path):
        self._file_path = file_path

    def build_path_from_id(self, id):
        if self._type_of_extraction == 'interactions':
            user = self._file_path[id]['user']
            return user+'_'+str(id)
        elif self._type_of_extraction == 'items':
            return str(id)

    def initiate_element_list_and_get_len(self):
        internal_list = []
        # element_list = []
        with open(self._file_path, newline='') as csvfile:
            file_dict = csv.DictReader(csvfile, delimiter='\t')
            for row in file_dict:
                internal_list.append(row)
                # if self._type_of_extraction == 'interactions':
                #     element_list.append(row['comment'])
                # elif self._type_of_extraction == 'items':
                #     element_list.append(row['description'])
        self._internal_list = internal_list
        return len(internal_list)

    def get_item_from_id(self, idx):
        row = self._internal_list[idx]
        if self._type_of_extraction == 'interactions':
            return row['comment']
        elif self._type_of_extraction == 'items':
            return row['description']

