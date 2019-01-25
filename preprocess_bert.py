#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import argparse
import os
from multiprocessing import Pool # multithread optimization, kf
import sys, random #kf
import torch

from fairseq.data import indexed_dataset
import pdb

from fairseq.tokenizer_bert import BertTokenizer

def generate_name(length=10, module=None): #kf
  if module is None:
    module = sys.modules[__name__]
  while True:
    name = 'id{:0{length}d}'.format(random.randint(0, 10**length-1),
      length=length)
    if not hasattr(module, name):
      return name

def global_function(func): #kf
  module = sys.modules[func.__module__]
  func.__name__ = func.__qualname__ = generate_name(module=module)
  setattr(module, func.__name__, func)
  return func

def get_parser():
    parser = argparse.ArgumentParser(
        description='Data pre-processing: Create dictionary and store data in binary format')

    parser.add_argument('--bertpref', metavar='FP', default="query,passage", help='A,B text to be bert')
    parser.add_argument('--trainpref', metavar='FP', default=None, help='train file prefix')
    parser.add_argument('--validpref', metavar='FP', default=None, help='comma separated, valid file prefixes')
    parser.add_argument('--testpref', metavar='FP', default=None, help='comma separated, test file prefixes')
    parser.add_argument('--data', metavar='DIR', default='data', help='source dir')
    parser.add_argument('--destdir', metavar='DIR', default='data-bin', help='destination dir')

    parser.add_argument('--do-lower-case',
                        default=False, action='store_true',
                        help='whether case sensitive')
    parser.add_argument('--tokenizer-dir', type=str, help="where to load bert tokenizer")
    parser.add_argument('--bert-vocab', type=str, default="vocab_file.txt", help="bert tokenizer vocab")
    parser.add_argument('--only-source', action='store_true', help='Only process the A language')

    return parser


def main(args):
    print(args)
    os.makedirs(args.destdir, exist_ok=True)
    need_B = not args.only_source


    bert_vocab_file = os.path.join(args.tokenizer_dir, args.bert_vocab)
    tokenizer = BertTokenizer.build_tokenizer(bert_vocab_file, args.do_lower_case)
    print('| build bert tokenizer and  dicts done!\n')
    
    @global_function
    def dataset_dest_path(output_prefix, extension):
        return "{}/{}.{}".format(args.destdir, output_prefix, extension)

    @global_function
    def make_binary_dataset(input_prefix, output_prefix):
        # dict = dictionary.Dictionary.load(dict_path(lang))
        # print('| [{}] Dictionary: {} types'.format(lang, len(dict) - 1))
        ds = indexed_dataset.IndexedDatasetBuilder(dataset_dest_path(output_prefix, 'bin'))
        def consumer(ids):
            ds.add_item(torch.IntTensor(ids))

        # input_file = '{}{}'.format(input_prefix, ('.' + lang) if lang is not None else '')
        input_file = "{}/{}.bert".format(args.data, input_prefix)

        res = tokenizer.binarize(input_file, consumer)
        ds.finalize(dataset_dest_path(output_prefix, 'idx'))
        
    @global_function
    def make_dataset(input_prefix, output_prefix):
        make_binary_dataset(input_prefix, output_prefix)

        
    @global_function
    def make_all(data_pref):
        for i, bertpref in enumerate(args.bertpref.split(',')):
            file_prefix = "{}-{}".format(data_pref, bertpref)
            # print(input_prefix)
            make_dataset(file_prefix, file_prefix)

    p = Pool(3)
    if args.trainpref:
        p.apply_async(make_all, (args.trainpref,))
    if args.validpref:
        p.apply_async(make_all, (args.validpref,))
    if args.testpref:
        p.apply_async(make_all, (args.testpref,))
    p.close()
    p.join()

    print('| Wrote preprocessed data to {}'.format(args.destdir))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
