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

def get_max_lens(item_a, item_bs):

    def lens_b(bs):
        return sum([len(b) for b in bs])

    lens = max(len(a)+lens_b(bs) for a,bs in zip(item_a, item_bs))
    return lens



def collate(
    samples, pad_idx, cls_idx, sep_idx, max_a_positions, max_b_positions, max_target_positions
):
    if len(samples) == 0:
        print("hahahaha")
        return {}
    batch_size = len(samples)
    def merge(key_a, key_b, max_a, max_b):
        
        item_a = [s[key_a] for s in samples]
        item_bs = [s[key_b] for s in samples]

        assert len(item_a)==len(item_bs)

        b_count = len(key_b[0])

        max_lens = get_max_lens(item_a, item_bs)
        max_lens = min(max_lens, max_a+max_b*b_count) + 2 + b_count

        sizes = len(item_a)
        input_ids = torch.LongTensor(sizes, max_lens).fill_(pad_idx)
        token_type_ids = torch.LongTensor(sizes, max_lens).fill_(pad_idx)
        attention_mask = torch.LongTensor(sizes, max_lens).fill_(pad_idx)
        position_ids = torch.longTensor(sizes, max_lens).fiil_(511)
        input_ids[:,0] = cls_idx
        for i, (a, bs) in enumerate(zip(item_a, item_bs)):
            size_a = min(len(a), max_a)
            size_bs = [min(len(b), max_b) for b in bs]
            # sep_A
            input_ids[i,1:size_a+1] = a[:size_a]
            input_ids[i,size_a+1] = sep_idx
            token_type_ids[i,:size_a+2] = 0
            position_ids[i, :size_a+2] = torch.arange(size_a+2)
            # sep_BS
            p_indexs = size_a+2
            for j,b in enumerate(bs):
                tmp_indexs = p_indexs + size_bs[j]

                input_ids[i,p_indexs:tmp_indexs] = b1[:size_bs[j]]
                input_ids[i,p_indexs] = sep_idx

                tmp_indexs += 1
                token_type_ids[i,p_indexs:tmp_indexs] = j
                position_ids[i,p_indexs:tmp_indexs] = torch.arange(size_bs[j]+1)
                p_indexs = tmp_indexs
            # MASK
            attention_mask[i,:p_indexs]=1

        return input_ids, token_type_ids, attention_mask, position_ids

    def merge_(key_data, max_tokens, copy_eos_to_beginning=True):
        values = [s[key_data][:max_tokens] for s in samples]
        size = max(v.size(0)+1 for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()+1
            if copy_eos_to_beginning:
                dst[0] = sep_idx
                # dst[0] = eos_idx
                dst[1:] = src[:]
            else:
                dst[:-1] = src[:]
                dst[-1] = sep_idx
                # dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][:len(v)+1])
        return res


    id = torch.LongTensor([s['id'] for s in samples])
    input_ids, token_type_ids, attention_mask = merge('a_item', 'b_items', max_a_positions, max_b_positions)
    prev_output_tokens = merge_("t_item", max_target_positions, copy_eos_to_beginning=True)
    target = merge_("t_item", max_target_positions, copy_eos_to_beginning=False)
    ntokens = sum(len(s["t_item"])+1 for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        "ntokens": ntokens,
        'net_input': {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            "attention_mask": attention_mask,
            "prev_output_tokens":prev_output_tokens,
            "position_ids":position_ids,
        },
        "target": target
    }
    return batch


class BertMultiDataset(FairseqDataset):
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
        self, a_data, a_sizes,
        b_datas, b_sizes,
        target_dataset, target_sizes, 
        bert_tokenizer,
        max_a_positions=1024, max_b_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True
    ):
        self.b_count = len(b_datas)
        self.a_data = a_data
        self.b_datas = b_datas
        self.target_dataset = target_dataset
        self.a_sizes = np.array(a_sizes)
        self.b_sizes = [np.array(b) for b in b_sizes]
        self.target_sizes = np.array(target_sizes)
        self.tokenizer = bert_tokenizer
        self.max_a_positions = max_a_positions
        self.max_b_positions = max_b_positions
        self.max_source_positions = max_a_positions+max_b_positions+3
        self.max_target_positions = max_target_positions
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

        a_item = self.a_data[index]
        b_items = [item[index] for item in self.b_datas]
        # b2_item = self.b2_data[index]
        t_item = self.target_dataset[index]
        return {
            'id': index,
            'a_item': a_item,
            'b_items': b_items,
            't_item':t_item
        }

    def __len__(self):
        return len(self.target_dataset)

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
            sep_idx=self.tokenizer.sep(), max_a_positions=self.max_a_positions, max_b_positions=self.max_b_positions,
            max_target_positions=self.max_target_positions
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
                'b_items': [torch.Tensor(src_len//2).uniform_(2, 200).long() for _ in range(self.b_count)],
                "t_item": torch.Tensor(tgt_len).uniform_(2, 20).long(),
            }
            for i in range(bsz)
        ])


    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # target_sizes = self.target_sizes[index]
        b_size = sum([b_sizes[index] for b_sizes in self.b_sizes])
        return min(self.a_sizes[index]+b_size+self.target_sizes[index], self.max_a_positions+self.max_b_positions*self.b_count+self.max_target_positions)

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
        self.target_dataset.prefetch(indices)
        for b_data in self.b_datas:
            b_data.prefetch(indices)
        self.a_data.prefetch(indices)


    @property
    def supports_prefetch(self):
        b_surports = [hasattr(b_data, 'supports_prefetch') and b_data.supports_prefetch for b_data in self.b_datas]
        return (
            hasattr(self.a_data, 'supports_prefetch')
            and self.a_data.supports_prefetch
            and hasattr(self.target_dataset, 'supports_prefetch')
            and self.target_dataset.supports_prefetch
            and all(b_surports)
        )
