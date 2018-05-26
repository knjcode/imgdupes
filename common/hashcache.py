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
from PIL import Image
from tqdm import tqdm

import imagehash
import joblib
import sys

class HashCache:
    def __init__(self, image_filenames, hash_method, load_path=None):
        self.image_filenames = image_filenames
        self.hashfunc = self.gen_hashfunc(hash_method)
        self.cache = []
        if load_path and Path(load_path).exists():
            self.cache = joblib.load(load_path)
            if len(image_filenames) == len(self.cache):
                logger.debug("Load hash cache: {}".format(load_path))
            else:
                self.cache = []
        else:
            self.cache = []


    def __len__(self):
        return len(self.cache)


    def hshs(self):
        return self.cache


    def get(self, index):
        return self.cache[index]


    def gen_hash(self, img):
        try:
            with Image.open(img) as i:
                hsh = self.hashfunc(i)
        except:
            hsh = imagehash.hex_to_hash('0x0000000000000000')
        return hsh


    def chunk(self, seq, chunk_size):
        for i in range(0, len(seq), chunk_size):
            yield seq[i:i+chunk_size]


    def make_hash_list(self):
        logger.warn("Calculating image hashes...")
        for image in tqdm(self.image_filenames):
            self.cache.append(self.gen_hash(image))


    def gen_hashfunc(self, hash_method):
        if hash_method == 'ahash':
            hashfunc = imagehash.average_hash
        elif hash_method == 'phash':
            hashfunc = imagehash.phash
        elif hash_method == 'dhash':
            hashfunc = imagehash.dhash
        elif hash_method == 'whash-haar':
            hashfunc = imagehash.whash
        elif hash_method == 'whash-db4':
            hashfunc = lambda img: imagehash.whash(img, mode='db4')
        return hashfunc


    def load(self, load_path):
        if load_path and Path(load_path).exists():
            self.cache = joblib.load(load_path)
            if len(self.image_filenames) == len(self.cache):
                logger.debug("Load hash cache: {}".format(load_path))
            else:
                self.cache = []
                self.make_hash_list()
        else:
            self.cache = []
            self.make_hash_list()


    def dump(self, dump_path):
        joblib.dump(self.cache, dump_path, protocol=2, compress=True)
        logger.debug("Dump hash cache: {}".format(dump_path))
