#!/usr/bin/env python
# coding: utf-8

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
# handler.setLevel(DEBUG)
# logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


from pathos.multiprocessing import ProcessPool
from multiprocessing import Pool, cpu_count
from pathlib import Path
from PIL import Image
from termcolor import colored, cprint
from tqdm import tqdm

import imagehash
import joblib
import sys

from common.spinner import Spinner


class NgtHashCache:
    def __init__(self, image_filenames, hash_method, num_hash_proc, load_path=None):
        self.image_filenames = image_filenames
        self.hashfunc = self.gen_hashfunc(hash_method)
        self.num_hash_proc = num_hash_proc
        self.cache = []


    def __len__(self):
        return len(self.cache)


    def hshs(self):
        return self.cache


    def get(self, index):
        return self.cache[index]


    def set(self, index, hsh):
        self.cache[index] = hsh


    def gen_hash(self, img):
        try:
            with Image.open(img) as i:
                hsh = self.hashfunc(i)
                hsh = [ 1 if b else 0 for b in hsh.hash.reshape((64))]
        except:
            hsh = [2] * 64
        return hsh


    def make_hash_list(self):
        if self.num_hash_proc is None:
            self.num_hash_proc = cpu_count() - 1
        try:
            from ngt import base as _ngt
        except:
            logger.error(colored("Error: Unable to load NGT. Please install NGT and python binding first.", 'red'))
            sys.exit(1)
        try:
            spinner = Spinner(prefix="Calculating image hashes for NGT...")
            spinner.start()
            with ProcessPool(self.num_hash_proc) as pool:
                self.cache = pool.map(self.gen_hash, self.image_filenames)
            spinner.stop()
        except KeyboardInterrupt:
            spinner.stop()
            sys.exit(1)


    def gen_hashfunc(self, hash_method):
        if hash_method == 'ahash':
            hashfunc = imagehash.average_hash
        elif hash_method == 'phash':
            hashfunc = imagehash.phash
        elif hash_method == 'dhash':
            hashfunc = imagehash.dhash
        elif hash_method == 'whash':
            hashfunc = imagehash.whash
        return hashfunc


    def load(self, load_path, use_cache):
        if load_path and Path(load_path).exists() and use_cache:
            self.cache = joblib.load(load_path)
            if len(self.image_filenames) == len(self.cache):
                logger.debug("Load hash cache: {}".format(load_path))
            else:
                self.cache = [[2] * 64 for i in range(len(self.image_filenames))]
                self.make_hash_list()
        else:
            self.cache = [[2] * 64 for i in range(len(self.image_filenames))]
            self.make_hash_list()


    def dump(self, dump_path, use_cache):
        if use_cache:
            joblib.dump(self.cache, dump_path, protocol=2, compress=True)
            logger.debug("Dump hash cache: {}".format(dump_path))
