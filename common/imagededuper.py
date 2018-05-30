#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function)

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
# handler.setLevel(DEBUG)
# logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

from builtins import input
from datetime import datetime
from multiprocessing import cpu_count
from operator import itemgetter
from pathlib import Path
from PIL import Image
from termcolor import colored, cprint
from tqdm import tqdm

import imagehash
import os
import re
import six
import sys

from common.imgcatutil import imgcat_for_iTerm2, create_tile_img
from common.hashcache import HashCache
from common.ngthashcache import NgtHashCache


class ImageDeduper:
    def __init__(self, args, image_filenames):
        self.target_dir = args.target_dir
        self.recursive = args.recursive
        self.image_filenames = image_filenames
        self.hash_method = args.hash_method
        self.hamming_distance = args.hamming_distance
        self.cache = args.cache
        self.ngt = args.ngt
        self.cleaned_target_dir = self.get_valid_filename(args.target_dir)
        if args.ngt:
            self.hashcache = NgtHashCache(self.image_filenames, self.hash_method, args.num_hash_proc)
        else:
            self.hashcache = HashCache(self.image_filenames, self.hash_method, args.num_hash_proc)
        self.group = {}
        self.num_duplecate_set = 0


    def get_valid_filename(self, path):
        path = str(path).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', path)


    def get_hashcache_dump_name(self):
        if self.ngt:
            return "hash_cache_ngt_{}_{}.pkl".format(self.cleaned_target_dir, self.hash_method)
        else:
            return "hash_cache_std_{}_{}.pkl".format(self.cleaned_target_dir, self.hash_method)


    def get_duplicate_log_name(self):
        if self.ngt:
            return "dup_ngt_{}_{}_{}.log".format(self.cleaned_target_dir, self.hash_method, self.hamming_distance)
        else:
            return "dup_std_{}_{}_{}.log".format(self.cleaned_target_dir, self.hash_method, self.hamming_distance)


    def get_delete_log_name(self):
        if self.ngt:
            return "del_ngt_{}_{}_{}.log".format(self.cleaned_target_dir, self.hash_method, self.hamming_distance)
        else:
            return "del_std_{}_{}_{}.log".format(self.cleaned_target_dir, self.hash_method, self.hamming_distance)


    def get_ngt_index_path(self):
        return "ngt_{}_{}_{}.ngt_index".format(self.cleaned_target_dir, self.hash_method, self.hamming_distance)


    def load_hashcache(self):
        return self.hashcache.load(self.get_hashcache_dump_name(), self.cache, self.target_dir)


    def dump_hashcache(self):
        return self.hashcache.dump(self.get_hashcache_dump_name(), self.cache)


    def preserve_file_question(self, file_num):
        preserve_all = {"all": True, "a": True}
        delete_all = {"none":True, "no": True, "n": True}
        file_num_set = set([i for i in range(1,file_num+1)])
        prompt = "preserve files [1 - {}, all, none]: ".format(file_num)
        error_prompt = "Please respond with comma-separated file numbers or 'all' or 'n'.\n"

        # return list of delete files index
        while True:
            sys.stdout.write(prompt)
            choice = input().lower()
            logger.debug("choice: {}".format(choice))
            if choice in preserve_all:
                return []
            elif choice in delete_all:
                return [i for i in range(1,file_num+1)]
            else:
                try:
                    input_num_set = set([int(i) for i in choice.split(',')])
                    logger.debug("input_num_set: {}".format(input_num_set))
                    delete_set = file_num_set - input_num_set
                    valid_set = input_num_set - file_num_set
                    if len(delete_set) >= 0 and len(valid_set) == 0:
                        return list(delete_set)
                    elif len(valid_set) != 0:
                        logger.debug("wrong file number: {}".format(valid_set))
                        sys.stdout.write(error_prompt)
                    else:
                        sys.stdout.write(error_prompt)
                except:
                    sys.stdout.write(error_prompt)


    def dedupe(self, args):
        if not self.load_hashcache():
            self.dump_hashcache()

        if not self.ngt:
            logger.warning("Searching similar images")
            hshs = self.hashcache.hshs()
            check_list = [0] * len(hshs)
            current_group_num = 1
            for i in tqdm(range(len(hshs))):
                new_group_found = False
                hshi = self.hashcache.get(i)
                for j in range(i+1, len(hshs)):
                    hshj = self.hashcache.get(j)
                    if (hshi - hshj) <= self.hamming_distance:
                        if check_list[j] == 0 and check_list[i] == 0:
                            # new group
                            new_group_found = True
                            check_list[i] = current_group_num
                            check_list[j] = current_group_num
                            self.group[current_group_num] = [self.image_filenames[i]]
                            self.group[current_group_num].extend([self.image_filenames[j]])
                        elif check_list[j] == 0 and check_list[i] != 0:
                            # exists group
                            exists_group_num = check_list[i]
                            check_list[j] = exists_group_num
                            self.group[exists_group_num].extend([self.image_filenames[j]])
                        elif check_list[j] != 0 and check_list[i] == 0:
                            # exists group
                            exists_group_num = check_list[j]
                            check_list[i] = exists_group_num
                            self.group[exists_group_num].extend([self.image_filenames[i]])
                        else: # check_list[j] != 0 and check_list[i] != 0
                            pass

                if new_group_found:
                    current_group_num += 1


        else:
            # NGT
            try:
                from ngt import base as ngt
            except:
                logger.error(colored("Error: Unable to load NGT. Please install NGT and python binding first.", 'red'))
                sys.exit(1)
            index_path = self.get_ngt_index_path()
            # check num_ngt_proc
            if args.num_ngt_proc is None:
                num_ngt_proc = cpu_count() -1
            else:
                num_ngt_proc = args.num_ngt_proc
            logger.warning("NGT: Creating NGT index")
            ngt_index = ngt.Index.create(index_path.encode(), 64, object_type="Integer", distance_type="Hamming")
            ngt_index.insert(self.hashcache.hshs(), num_ngt_proc)
            logger.warning("NGT: Building index")
            ngt_index.build_index(num_ngt_proc)
            logger.warning("NGT: Indexing complete")

            # NGT Approximate neighbor search
            logger.warning("NGT: Approximate neighbor searching")
            hshs = self.hashcache.hshs()
            check_list = [0] * len(hshs)
            current_group_num = 1
            for i in tqdm(range(len(hshs))):
                new_group_found = False
                if check_list[i] != 0:
                    # already grouped image
                    continue
                for res in ngt_index.search(hshs[i], k=args.ngt_k, epsilon=args.ngt_epsilon):
                    if res.id-1 == i:
                        continue
                    else:
                        if res.distance <= self.hamming_distance:
                            if check_list[res.id-1] == 0 and check_list[i] == 0:
                                # new group
                                new_group_found = True
                                check_list[i] = current_group_num
                                check_list[res.id-1] = current_group_num
                                self.group[current_group_num] = [self.image_filenames[i]]
                                self.group[current_group_num].extend([self.image_filenames[res.id-1]])
                            elif check_list[res.id-1] == 0 and check_list[i] != 0:
                                # exists group
                                exists_group_num = check_list[i]
                                check_list[res.id-1] = exists_group_num
                                self.group[exists_group_num].extend([self.image_filenames[res.id-1]])
                            elif check_list[res.id-1] != 0 and check_list[i] == 0:
                                # exists group
                                exists_group_num = check_list[res.id-1]
                                check_list[i] = exists_group_num
                                self.group[exists_group_num].extend([self.image_filenames[i]])
                            else: # check_list[res.id-1] != 0 and check_list[i] != 0
                                pass

                if new_group_found:
                    current_group_num += 1

            # remove ngt index
            if index_path:
                os.system("rm -rf {}".format(index_path))


        # write duplicate log file
        self.num_duplecate_set = current_group_num -1
        if self.num_duplecate_set > 0 and args.log:
            now = datetime.now().strftime('%Y%m%d%H%M%S')
            duplicate_log_file = "{}_{}".format(now, self.get_duplicate_log_name())
            with open(duplicate_log_file, 'w') as f:
                for _k, img_list in six.iteritems(self.group):
                    if len(img_list) > 1:
                        f.write("\n".join(img_list) + '\n')


    def print_duplicates(self, args):
        for _k, img_list in six.iteritems(self.group):
            if len(img_list) > 1:
                print("\n".join(img_list) + "\n")


    def preserve(self, args):
        deleted_filenames = []

        current_set = 0
        for _k, img_list in six.iteritems(self.group):
            if len(img_list) > 1:
                current_set += 1

                img_size_dict = {}
                img_pixel_dict = {}
                for img in img_list:
                    img_size_dict[img] = os.path.getsize(img)
                    with Image.open(img) as current_img:
                        width, height = current_img.size
                        img_pixel_dict[img] = "{}x{}".format(width, height)
                sorted_img_size_list = sorted(img_size_dict.items(), key=itemgetter(1), reverse=True)
                sorted_img_list = [img for img, size in sorted_img_size_list]

                if args.imgcat:
                    imgcat_for_iTerm2(create_tile_img(sorted_img_list, args))

                # check different parent dir
                parent_set = set([])
                for img in sorted_img_list:
                    parent_set.add(str(Path(img).parent))
                if len(parent_set) > 1 and args.print_warning:
                    logger.warning(colored('WARNING! Similar images are stored in different subdirectories.', 'red'))
                    logger.warning(colored('\n'.join(parent_set), 'red'))

                for index, (img, size) in enumerate(sorted_img_size_list, start=1):
                    pixel = img_pixel_dict[img]
                    print("[{}] {:>8.2f} kbyte {:>9} {}".format(index, (size/1024), pixel, img))
                print("")
                print("Set {} of {}, ".format(current_set, self.num_duplecate_set), end='')
                delete_list = self.preserve_file_question(len(sorted_img_list))
                logger.debug("delete_list: {}".format(delete_list))

                print("")
                for i in range(1, len(img_list)+1):
                    if i in delete_list:
                        delete_file = sorted_img_list[i-1]
                        print("   [-] {}".format(delete_file))
                        if args.run:
                            self.delete_image(delete_file)
                        deleted_filenames.append(delete_file)
                    else:
                        preserve_file = sorted_img_list[i-1]
                        print("   [+] {}".format(preserve_file))
                print("")

        # write delete log file
        if len(deleted_filenames) >0 and  args.run and args.log:
            now = datetime.now().strftime('%Y%m%d%H%M%S')
            delete_log_file = "{}_{}".format(now, self.get_delete_log_name())
            with open(delete_log_file, 'w') as f:
                for del_file in deleted_filenames:
                    f.write("{}\n".format(del_file))

        if not args.run:
            logger.debug("dry-run")
            logger.debug("delete_candidate: {}".format(deleted_filenames))


    def delete_image(self, delete_file):
        try:
            os.remove(delete_file)
        except FileNotFoundError as e:
            logger.error(e)
            pass