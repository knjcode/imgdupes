#!/usr/bin/env python
# coding: utf-8

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
# handler.setLevel(DEBUG)
# logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

from pathlib import Path
import joblib

class HashCache:
    def __init__(self, load_path=None):
        if load_path and Path(load_path).exists():
            self.cache = joblib.load(load_path)
            logger.debug("Load hash cache: {}".format(load_path))
        else:
            self.cache = {}

    def __len__(self):
        return len(self.cache)

    def get(self, filepath):
        return self.cache.get(filepath, False)

    def set(self, filepath, hsh):
        self.cache[filepath] = hsh

    def load(self, load_path):
        if load_path and Path(load_path).exists():
            self.cache = joblib.load(load_path)
            logger.debug("Load hash cache: {}".format(load_path))
        else:
            self.cache = {}

    def dump(self, dump_path):
        joblib.dump(self.cache, dump_path, protocol=2, compress=True)
        logger.debug("Dump hash cache: {}".format(dump_path))
