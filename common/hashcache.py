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
from multiprocessing import cpu_count
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import imagehash
import joblib
import sys

from common.spinner import Spinner


class HashCache:
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


    def make_hash_list(self, process=None):
        if self.num_hash_proc is None:
            self.num_hash_proc = cpu_count() - 1
        try:
            spinner = Spinner(prefix="Calculating image hashes...")
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


    def check_mtime(self, path):
        return Path(path).stat().st_mtime


    def check_latest_dir_mtime(self, path):
        return max([p.stat().st_mtime for p in Path(path).glob('**')])


    def load(self, load_path, use_cache, target_dir):
        if load_path and Path(load_path).exists() and use_cache:
            cache_mtime = self.check_mtime(load_path)
            target_mtime = self.check_latest_dir_mtime(target_dir)
            if cache_mtime > target_mtime:
                logger.debug("Load hash cache: {}".format(load_path))
                self.cache = joblib.load(load_path)
                return True
            else:
                self.cache = []
                self.make_hash_list()
                return False
        else:
            self.cache = []
            self.make_hash_list()
            return False


    def dump(self, dump_path, use_cache):
        if use_cache:
            joblib.dump(self.cache, dump_path, protocol=2, compress=True)
            logger.debug("Dump hash cache: {}".format(dump_path))
            return True
        else:
            return False
