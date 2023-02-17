import re

# this code is inspired by
# https://nedbatchelder.com/blog/200712/human_sorting.html


def _if_int(text):
    return int(text) if text.isdigit() else text


def _natural_keys(text):
    return [_if_int(c) for c in re.split(r'(\d+)', text)]


def human_sort(unsorted_list):
    unsorted_list.sort(key=_natural_keys)
    return unsorted_list
