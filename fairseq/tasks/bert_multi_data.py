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

from fairseq.data.indexed_dataset import BertIndexedCachedDataset

from fairseq.data.bert_multi_dataset import BertMultiDataset
from fairseq.tokenizer_bert import BertTokenizer

from . import FairseqTask, register_task
from fairseq.data import data_utils, FairseqDataset, iterators

@register_task('bert_multi_data')
class BertMultiGenerator(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('--max-query-positions', default=40, type=int, metavar='N',
                            help='max number of tokens in the query sequence')
        parser.add_argument('--max-passage-positions', default=200, type=int, metavar='N',
                            help='max number of tokens in the passage sequence')
        parser.add_argument('--max-target-positions', default=200, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        # parser.add_argument('--max-tokens', default=3000, type=int, metavar='N',
        #                     help='max number of tokens in one batches')
        parser.add_argument('--passage-first',
                        default=False, action='store_true',
                        help='if set, will be passage-query into bert')
        parser.add_argument('--threshold', type=int, default=0.36)
        parser.add_argument('--train-batch', default=2, type=int, metavar='N',
                            help='train batch in msmarco data')
        parser.add_argument('--bertpref', metavar='FP', default="query,passage-1,passage-2,target", help='A,B text to be bert')
        parser.add_argument('--do-lower-case',
                            default=False, action='store_true',
                            help='whether case sensitive')
        parser.add_argument('--tokenizer-dir', type=str, help="where to load bert tokenizer")
        parser.add_argument('--bert-vocab', type=str, default="vocab_file.txt", help="bert tokenizer vocab")
        parser.add_argument('--only-source', action='store_true', help='Only process the A language')

    def __init__(self, args, bert_tokenuzer=None):
        super().__init__(args)
        args.max_source_positions = args.max_query_positions+args.max_passage_positions*2+3
        self.args = args
        self.tokenizer = bert_tokenuzer
        self.max_tokens = args.max_tokens

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        bert_vocab_file = os.path.join(args.tokenizer_dir, args.bert_vocab)
        tokenizer = BertTokenizer.build_tokenizer(bert_vocab_file, args.do_lower_case)
        print('| build bert tokenizer and  dicts done!\n')

        return cls(args, tokenizer)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, bert_pref, data_path):
            filename = os.path.join(data_path, '{}-{}'.format(split, bert_pref))
            if IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path):
            if IndexedDataset.exists(path):
                return BertIndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for bert_pref in self.args.bertpref.split(","):
                if split_exists(split, bert_pref, data_path):
                    prefix = os.path.join(data_path, '{}-{}'.format(split, bert_pref))
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
                src_datasets.append(indexed_dataset(prefix))

                print('| load  {}-{} {} {} examples'.format(data_path, split, bert_pref, len(src_datasets[-1])))
        assert len(src_datasets) >= 3


        query_dataset = src_datasets[0]
        passage_datasets = src_datasets[1:-1]
        target_dataset = src_datasets[-1]


        self.datasets[split] = BertMultiDataset(
            query_dataset, query_dataset.sizes,
            passage_datasets, [p.sizes for p in passage_datasets],
            target_dataset, target_dataset.sizes,
            self.tokenizer,
            max_a_positions=self.args.max_query_positions,
            max_b_positions=self.args.max_passage_positions,
            max_target_positions=self.args.max_target_positions
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.tokenizer

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tokenizer

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0,num_workers=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch.
                Default: ``None``
            max_sentences (int, optional): max number of sentences in each
                batch. Default: ``None``
            max_positions (optional): max sentence length supported by the
                model. Default: ``None``
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long. Default: ``False``
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N. Default: ``1``
            seed (int, optional): seed for random number generator for
                reproducibility. Default: ``1``
            num_shards (int, optional): shard the data iterator into N
                shards. Default: ``1``
            shard_id (int, optional): which shard of the data iterator to
                return. Default: ``0``

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        # indices = data_utils.filter_by_size(
        #     indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
        # )

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
