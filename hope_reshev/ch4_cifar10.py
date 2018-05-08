import tensorflow as tf


class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
