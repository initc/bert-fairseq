# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from fairseq import options
from fairseq.data import (
    data_utils, IndexedDataset
)

from fairseq.data.indexed_dataset import SMPIndexedDataset

from fairseq.data.smp_dataset import SMPDataset
from fairseq.tokenizer_bert import BertTokenizer
from fairseq.tokenizer_jieba import JiebaTokenizer

from . import FairseqTask, register_task
from fairseq.data import data_utils, FairseqDataset, iterators

import pdb

@register_task('bert_concat_decoder')
class BertConcatDecoder(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--max-positions', default=500, type=int, metavar='N', help='max number of tokens in the source sequence')
        # parser.add_argument('--max-tokens', default=3000, type=int, metavar='N',
        #                     help='max number of tokens in one batches')
        parser.add_argument("--max-history", type=int, default=-1, help="Number of previous exchanges to keep in history")
        parser.add_argument('--train-batch', default=2, type=int, metavar='N' ,help='train batch in msmarco data')
        parser.add_argument('--do-lower-case',default=False, action='store_true' ,help='whether case sensitive')
        parser.add_argument('--tokenizer-dir', type=str, help="where to load bert tokenizer ")
        parser.add_argument('--bert-vocab', type=str, default="vocab_bert.txt", help="bert tokenizer vocab")
        parser.add_argument('--jieba-vocab', type=str, default="vocab_jieba.txt", help="bert tokenizer vocab")
        parser.add_argument('--only-source', action='store_true', help='Only process the A language')
        parser.add_argument('--generate-file', type=str, help="where to save model output")
        parser.add_argument('--max-tokens-generate', type=int, metavar='N', default=15,help='max tokens in generate')

    def __init__(self, args, encoder_dict=None, decoder_dict=None):
        super().__init__(args)
        self.args = args
        self.encoder_dict = encoder_dict
        self.decoder_dict = decoder_dict
        self.max_tokens = args.max_tokens
        self.max_position = args.max_positions

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        bert_vocab_file = os.path.join(args.tokenizer_dir, args.bert_vocab)
        jieba_vocab_file = os.path.join(args.tokenizer_dir, args.jieba_vocab)
        encoder_dict = BertTokenizer.build_tokenizer(bert_vocab_file, args.do_lower_case)
        decoder_dict = JiebaTokenizer(jieba_vocab_file)
        print('| build bert and jieba dicts done!\n')

        return cls(args, encoder_dict, decoder_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_path):
            filename = os.path.join(data_path, '{}'.format(split))
            if SMPIndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path):
            if SMPIndexedDataset.exists(path):
                return SMPIndexedDataset(path, fix_lua_indexing=True)
            return None

        datas = []

        data_paths = self.args.data
        # pdb.set_trace()

        for dk, data_path in enumerate(data_paths):
            if split_exists(split, data_path):
                prefix = os.path.join(data_path, '{}'.format(split))
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
            datas.append(indexed_dataset(prefix))

            print('| load  {}-{} {} examples'.format(data_path, split, len(datas[-1])))
        assert len(datas) == 1

        data = datas[0]

        self.datasets[split] = SMPDataset(
            data,
            self.encoder_dict, self.decoder_dict,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            max_history=self.args.max_history
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source."""
        return self.encoder_dict

    @property
    def target_dictionary(self):
        """Return the target."""
        return self.decoder_dict

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0,num_workers=0,epoch=0,
    ):
        
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        max_positions = self.max_position
        indices = data_utils.filter_by_size(
            indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
        )

        # create mini-batches with given size constraints
        batch_sampler = self.batch_by_size(indices, max_sentences, max_tokens, dataset.num_tokens)

        # return a reusable, sharded iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            epoch=epoch,
        )

    def batch_by_size(self, indices, size, max_tokens, num_tokens_fn):

        batch = []
        tokens = 0
        def is_batch_full(num_tokens):
            if len(batch)==0:
                return False
            if num_tokens > max_tokens:
                return True
            if len(batch)==size:
                return True
            return False
        for idx in indices:
            idx_max_token = num_tokens_fn(idx)
            tokens = max(tokens, idx_max_token)
            if is_batch_full(tokens*(len(batch)+1)):
                yield batch
                batch = []
                tokens = 0
            batch.append(idx)
        if len(batch) > 0:
            yield batch
