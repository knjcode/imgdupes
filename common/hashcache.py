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
from tqdm import tqdm

import imagehash
import joblib
import sys
import six
import numpy
import scipy

from common.spinner import Spinner


class HashCache:
    def __init__(self, image_filenames, hash_method, hash_size, num_proc, load_path=None):
        self.image_filenames = image_filenames
        self.hashfunc = self.gen_hashfunc(hash_method)
        self.hash_size = hash_size
        self.num_proc = num_proc
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
                hsh = self.hashfunc(i, hash_size=self.hash_size)
        except:
            hsh = imagehash.hex_to_hash('0x0000000000000000')
        return hsh


    def chunk(self, seq, chunk_size):
        for i in range(0, len(seq), chunk_size):
            yield seq[i:i+chunk_size]


    def make_hash_list(self):
        if self.num_proc is None:
            self.num_proc = cpu_count() - 1
        try:
            spinner = Spinner(prefix="Calculating image hashes (hash-bits={} num-proc={})...".format(self.hash_size ** 2, self.num_proc))
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


    def phash_org(self, image, hash_size=8, highfreq_factor=4):
        if hash_size < 2:
                raise ValueError("Hash size must be greater than or equal to 2")

        import scipy.fftpack
        img_size = hash_size * highfreq_factor
        image = image.convert("L").resize((img_size, img_size), Image.ANTIALIAS)
        pixels = numpy.asarray(image)
        dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
        # using only the 8x8 DCT low-frequency values and excluding the first term since the DC coefficient
        # can be significantly different from the other valuesand will throw off the average.
        dctlowfreq = dct[1:hash_size+1, 1:hash_size+1]
        med = numpy.median(dctlowfreq)
        diff = dctlowfreq > med
        return imagehash.ImageHash(diff)


    def gen_hashfunc(self, hash_method):
        if hash_method == 'ahash':
            hashfunc = imagehash.average_hash
        elif hash_method == 'phash':
            hashfunc = imagehash.phash
        elif hash_method == 'dhash':
            hashfunc = imagehash.dhash
        elif hash_method == 'whash':
            hashfunc = imagehash.whash
        elif hash_method == 'phash_org':
            hashfunc = self.phash_org
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
                spinner = Spinner(prefix="Loading hash cache...")
                spinner.start()
                self.cache = joblib.load(load_path)
                spinner.stop()
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
