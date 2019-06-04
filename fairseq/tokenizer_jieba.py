import json
import os

import collections
from collections import defaultdict

VOCAB_FILE = "vocab.txt"

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

class JiebaTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""
    def __init__(self, vocab_file):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}' ".format(vocab_file))


        self.encode = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.encode.items()])

    def __len__(self):
        return len(self.encode)

    def convert_to_dict(self, lprobs):
        vocab = dict()
        lprobs = lprobs.tolist()
        for ids, tok in self.ids_to_tokens.items():
            vocab[tok] = lprobs[ids]
        return vocab

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.encode["[PAD]"]

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.encode["[EOS]"]

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.encode["[UNK]"]

    def cls(self):
        return self.encode["[CLS]"]

    def sep(self):
        return self.encode["[SEP]"]

    def mask(self):
        return self.encode["MASK"]

    def add_symbol(self, word):
        """Adds a word to the dictionary"""
        if word not in self.encode:
            idx = len(self.symbols)
            self.encode[word] = idx
            self.symbols.append(word)
            return idx


    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token not in self.encode:
                ids.append(self.unk())
            else:
                ids.append(self.encode[token])
        return ids


    def convert_text_to_ids(self, text):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = text.strip().split()
        return self.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, indices):
        tokens = []
        for inx in indices:
            tokens.append(self.ids_to_tokens[inx])
        return tokens