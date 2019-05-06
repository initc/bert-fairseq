# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset

def get_max_lens(item_a):


    lens = max(len(a) for a in item_a)
    return lens



def collate(
    samples, pad_idx, cls_idx, sep_idx, max_source_positions=512
):
    if len(samples) == 0:
        print("hahahaha")
        return {}
    batch_size = len(samples)
    def merge(key_a, key_t, max_a):
        
        item_a = [s[key_a][:max_source_positions] for s in samples]
        item_t = [s[key_t][:max_source_positions] for s in samples]

        assert len(item_a)==len(item_t)


        max_lens = get_max_lens(item_a)
        max_lens = max_lens + 2

        sizes = len(item_a)
        input_ids = torch.LongTensor(sizes, max_lens).fill_(pad_idx)
        token_type_ids = torch.LongTensor(sizes, max_lens).fill_(0)
        attention_mask = torch.LongTensor(sizes, max_lens).fill_(pad_idx)

        # except cls
        target_ids = torch.LongTensor(sizes, max_lens-1).fill_(pad_idx)
        input_ids[:,0] = cls_idx
        for i, (a, t) in enumerate(zip(item_a, item_t)):
            assert len(a)==len(t)
            size_ab = len(a)
            # sep_A
            input_ids[i,1:size_ab+1] = a[:size_ab]
            input_ids[i,size_ab+1] = sep_idx
            attention_mask[i,:size_ab+2]=1
            target_ids[i,0:size_ab] = t[:size_ab]


        return input_ids, token_type_ids, attention_mask, target_ids



    id = torch.LongTensor([s['id'] for s in samples])
    input_ids, token_type_ids, attention_mask, target_ids = merge('a_item', 't_item', max_source_positions)

    ntokens = sum(len(s["t_item"])+1 for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        "ntokens": ntokens,
        'net_input': {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            "attention_mask": attention_mask
        },
        "target": target_ids
    }
    return batch


class NERPairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side.
            Default: ``True``
        left_pad_target (bool, optional): pad target tensors on the left side.
            Default: ``False``
        max_source_positions (int, optional): max number of tokens in the source
            sentence. Default: ``1024``
        max_target_positions (int, optional): max number of tokens in the target
            sentence. Default: ``1024``
        shuffle (bool, optional): shuffle dataset elements before batching.
            Default: ``True``
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing.
            Default: ``True``
        remove_eos_from_source (bool, optional): if set, removes eos from end of
            source if it's present. Default: ``False``
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent. Default: ``False``
    """

    def __init__(
        self, src_data, src_sizes,
        tgt_data, tgt_sizes,
        bert_tokenizer, tgt_dict,
        max_source_positions=512,
        shuffle=True, input_feeding=True
    ):
        self.tgt_count = len(tgt_data)
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes)
        self.tokenizer = bert_tokenizer
        self.tgt_dict = tgt_dict
        self.max_source_positions = max_source_positions
        self.max_target_positions = len(self.tgt_dict)
        self.shuffle = shuffle
        self.input_feeding = input_feeding


    def __getitem__(self, index):
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        # if self.append_eos_to_target:
        #     eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
        #     if self.tgt and self.tgt[index][-1] != eos:
        #         tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        # if self.remove_eos_from_source:
        #     eos = self.src_dict.eos()
        #     if self.src[index][-1] == eos:
        #         src_item = self.src[index][:-1]

        src_item = self.src_data[index]
        tgt_item = self.tgt_data[index]
        return {
            'id': index,
            'a_item': src_item,
            't_item':tgt_item
        }

    def __len__(self):
        return len(self.tgt_data)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.tokenizer.pad(), cls_idx=self.tokenizer.cls(),
            sep_idx=self.tokenizer.sep(), max_source_positions=self.max_source_positions
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = 2
        return self.collater([
            {
                'id': i,
                'a_item': torch.Tensor(src_len//2).uniform_(2, 20).long(),
                "t_item": torch.Tensor(src_len//2).uniform_(2, 20).long(),
            }
            for i in range(bsz)
        ])


    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # target_sizes = self.target_sizes[index]
        return self.src_sizes[index]


    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        lens = []
        for inx_a in indices:
            lens.append(self.num_tokens(inx_a))
        indices = indices[np.argsort(np.array(lens), kind='mergesort')]
        return indices

    def prefetch(self, indices):
        self.src_data.prefetch(indices)
        self.tgt_data.prefetch(indices)


    @property
    def supports_prefetch(self):

        return (
            hasattr(self.src_data, 'supports_prefetch')
            and self.src_data.supports_prefetch
            and hasattr(self.tgt_data, 'supports_prefetch')
            and self.tgt_data.supports_prefetch
        )
