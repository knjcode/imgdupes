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
from PIL import Image, ImageFile
from tqdm import tqdm

import imagehash
import joblib
import sys
import six
import numpy
import scipy

from common.spinner import Spinner

ImageFile.LOAD_TRUNCATED_IMAGES = True


class HashCache:
    def __init__(self, args, image_filenames, hash_method, hash_size, num_proc, load_path=None):
        self.args = args
        self.image_filenames = image_filenames
        self.hashfunc = self.gen_hashfunc(hash_method)
        self.hash_size = hash_size
        self.hash_bits = hash_size ** 2
        self.num_proc = num_proc
        self.hash_dict = {}


    def __len__(self):
        return len(self.cache)


    def hshs(self):
        return list(self.hash_dict.values())


    def filenames(self):
        return list(self.hash_dict.keys())


    def gen_hash(self, img):
        try:
            with Image.open(img) as i:
                hsh = self.hashfunc(i, hash_size=self.hash_size)
                hsh = numpy.array([ 1 if b else 0 for b in hsh.hash.reshape((self.hash_bits))])
        except:
            hsh = numpy.array([2] * self.hash_bits)
        return hsh


    def make_hash_list(self):
        if self.num_proc is None:
            self.num_proc = cpu_count() - 1

        try:
            spinner = Spinner(prefix="Calculating image hashes (hash-bits={} num-proc={})...".format(self.hash_bits, self.num_proc))
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


    def update_hash_dict(self):
        if self.num_proc is None:
            self.num_proc = cpu_count() - 1

        # check current hash_dict
        current_files = set(self.image_filenames)
        cache_files = self.hash_dict.keys()
        lost_set = cache_files - current_files
        target_files = list(current_files - cache_files)

        if len(lost_set) + len(target_files) > 0:
            try:
                if len(self.hash_dict) == 0:
                    spinner = Spinner(prefix="Calculating image hashes (hash-bits={} num-proc={})...".format(self.hash_bits, self.num_proc))
                else:
                    spinner = Spinner(prefix="Updating image hashes (hash-bits={} num-proc={})...".format(self.hash_bits, self.num_proc))
                spinner.start()

                # del lost_set from hash_dict
                for f in lost_set:
                    del self.hash_dict[f]

                if six.PY2:
                    from pathos.multiprocessing import ProcessPool as Pool
                elif six.PY3:
                    from multiprocessing import Pool
                pool = Pool(self.num_proc)
                hashes = pool.map(self.gen_hash, target_files)
                for filename, hash_value in zip(target_files, hashes):
                    self.hash_dict[filename] = hash_value
                spinner.stop()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                spinner.stop()
                sys.exit(1)
            return True
        else:
            return False


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


    def load_hash_dict(self, load_path, use_cache, target_dir):
        if load_path and Path(load_path).exists() and use_cache:
            logger.debug("Load hash cache: {}".format(load_path))
            spinner = Spinner(prefix="Loading hash cache...")
            spinner.start()
            self.hash_dict = joblib.load(load_path)
            spinner.stop()
            is_update = self.update_hash_dict()
            return not is_update
        else:
            self.hash_dict = {}
            self.update_hash_dict()
            return False


    def dump_hash_dict(self, dump_path, use_cache):
        if use_cache:
            joblib.dump(self.hash_dict, dump_path, protocol=2, compress=True)
            logger.debug("Dump hash cache: {}".format(dump_path))
            return True
        else:
            return False
