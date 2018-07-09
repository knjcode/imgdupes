#!/usr/bin/env python
# coding: utf-8

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
# handler.setLevel(DEBUG)
# logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


from multiprocessing import cpu_count
from pathlib import Path
from PIL import Image
from termcolor import colored, cprint

import imagehash
import joblib
import sys
import six

from common.spinner import Spinner


class KnnHashCache:
    def __init__(self, args, image_filenames, hash_method, hash_size, num_proc, load_path=None):
        self.args = args
        self.image_filenames = image_filenames
        self.hashfunc = self.gen_hashfunc(hash_method)
        self.hash_size = hash_size
        self.hash_bits = hash_size ** 2
        self.num_proc = num_proc
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
                hsh = self.hashfunc(i, hash_size=self.hash_size)
                hsh = [ 1 if b else 0 for b in hsh.hash.reshape((self.hash_bits))]
        except:
            hsh = [2] * self.hash_bits
        return hsh


    def package_check(self):
        if self.args.ngt:
            try:
                from ngt import base as _ngt
            except:
                logger.error(colored("Error: Unable to load NGT. Please install NGT and python binding first.", 'red'))
                sys.exit(1)
        elif self.args.hnsw:
            try:
                import hnswlib as _hnswlib
            except:
                logger.error(colored("Error: Unable to load hnsw. Please install hnsw python binding first.", 'red'))
                sys.exit(1)
        elif self.args.faiss_flat:
            try:
                import faiss as _faiss
            except:
                logger.error(colored("Error: Unable to load faiss. Please install faiss python binding first.", 'red'))
                sys.exit(1)
        else:
            pass


    def make_hash_list(self):
        if self.num_proc is None:
            self.num_proc = cpu_count() - 1

        self.package_check()

        try:
            spinner = Spinner(
                prefix="Calculating image hashes (hash-bits={} num-proc={})...".format(self.hash_bits, self.num_proc),
                disable=self.args.quiet)
            spinner.start()
            if six.PY2:
                from pathos.multiprocessing import ProcessPool as Pool
            elif six.PY3:
                from multiprocessing import Pool
            pool = Pool(self.num_proc)
            self.cache = pool.map(self.gen_hash, self.image_filenames)
            spinner.stop()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
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
                spinner = Spinner(prefix="Loading hash cache...", disable=self.args.quiet)
                spinner.start()
                self.cache = joblib.load(load_path)
                spinner.stop()
                return True
            else:
                self.cache = [[2] * self.hash_bits for i in range(len(self.image_filenames))]
                self.make_hash_list()
                return False
        else:
            self.cache = [[2] * self.hash_bits for i in range(len(self.image_filenames))]
            self.make_hash_list()
            return False


    def dump(self, dump_path, use_cache):
        if use_cache:
            joblib.dump(self.cache, dump_path, protocol=2, compress=True)
            logger.debug("Dump hash cache: {}".format(dump_path))
            return True
        else:
            return False
