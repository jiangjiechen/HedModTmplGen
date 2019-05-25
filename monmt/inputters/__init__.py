"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from monmt.inputters.inputter import make_features, \
	load_old_vocab, get_fields, OrderedIterator, \
	build_dataset, build_vocab, old_style_vocab, \
	collect_features, save_fields_to_vocab
from monmt.inputters.dataset_base import DatasetBase
from monmt.inputters.text_dataset import TextDataset

__all__ = ['DatasetBase', 'make_features',
           'load_old_vocab', 'get_fields',
           'build_dataset', 'old_style_vocab', 'collect_features',
           'build_vocab', 'OrderedIterator',
           'TextDataset']
