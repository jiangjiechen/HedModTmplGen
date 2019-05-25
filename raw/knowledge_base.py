# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2018/4/12 13:04
@Contact    : mi0134sher@hotmail.com
@Description: 
'''

import sys

sys.path.append('..')
import time, cjjpy as cjj, random
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

random.seed(1111)


def kbinp_adjust_to_onmt(kb, opt, feat_delim=u"￨"):
	src, _, tgt, tmpl = kb.truncate(max_src_len=opt.max_src_len,
	                                max_val_words_num=opt.max_val_words_num,
	                                max_tgt_len=opt.max_tgt_len,
	                                return_id=False,
	                                need_pad=False)
	assert len(src) == len(tgt) == len(tmpl)

	def list2str(item, is_src):
		if is_src:
			return ' '.join(feat_delim.join(it) for it in item)
		else:
			return ' '.join(item)

	src = list(map(lambda x: list2str(x, True), src))
	tgt = list(map(lambda x: list2str(x, False), tgt))
	tmpl = list(map(lambda x: list2str(x, False), tmpl))

	return src, tgt, tmpl


class PosVocab:
	def __init__(self, max_num):
		self.max_num = max_num + 2
		self.pad_token = '<pad>'
		self.unk_token = '<unk>'
		self.stoi = {str(i): i for i in range(max_num)}
		self.stoi[self.pad_token] = max_num
		self.stoi[self.unk_token] = max_num + 1
		self.itos = {word: idx for idx, word in self.stoi.items()}

	def __getstate__(self):
		return dict(self.__dict__, stoi=dict(self.stoi))

	def __setstate__(self, state):
		self.__dict__.update(state)
		self.stoi = defaultdict(lambda: 0, self.stoi)

	def __len__(self):
		return self.max_num


class Dictionary:
	def __init__(self, vocab_file, vocab_size=-1, has_freq=False, prop_vocab=False):
		assert vocab_file.endswith('.vocab')

		self.vocab_file = vocab_file
		self.vocab_size = vocab_size
		self.has_freq = has_freq
		self.pad_token = "<pad>"
		self.eos_token = "<eos>"
		self.bos_token = "<bos>"
		self.unk_token = "<unk>"
		self.hed_token = "<hed>"
		self.mod_token = "<mod>"
		if prop_vocab:
			self.special_words = [self.pad_token, self.unk_token]
		else:
			self.special_words = [self.pad_token, self.eos_token, self.bos_token, self.unk_token,
			                      self.hed_token, self.mod_token]
		self.itos, self.stoi = self._build_vocab()
		self.vocab_size = min(vocab_size, len(self.itos)) if vocab_size > 0 else len(self.itos)

	def _build_vocab(self):
		'''
		map dict(word) to index
		:return: id2w, w2id
		'''
		special_num = len(self.special_words)
		with open(self.vocab_file, 'r') as f:
			dict_data = f.read().splitlines()
			self.tot_vocab_size = len(dict_data) + special_num

		set_words, leftovers = [], []
		if self.has_freq:
			voc_sz = 0
			for line in dict_data:
				splt = line.split('\t')
				if len(splt) != 2: continue
				if self.vocab_size > special_num \
						and voc_sz >= self.vocab_size - special_num:
					leftovers.append(splt[0])
				else:
					voc_sz += 1
					set_words.append(splt[0])
		else:
			anchor = self.vocab_size - special_num
			set_words = dict_data[:anchor] \
				if self.vocab_size > special_num else dict_data
			leftovers = dict_data[anchor:]
		id2w = {idx: word for idx, word in enumerate(self.special_words + set_words)}
		w2id = {word: idx for idx, word in id2w.items()}
		for w in leftovers:
			w2id[w] = w2id[self.unk_token]
		return id2w, w2id

	def w2id(self, word):
		return self.stoi.get(word, self.stoi[self.unk_token])

	def id2w(self, id):
		return self.itos.get(id, self.unk_token)

	def make_unk(self):
		# TODO: unk
		pass

	def __len__(self):
		return self.vocab_size


class InfoBox:
	def __init__(self, qid):
		self.qid = qid
		self.desc = ''
		self.name = ''
		self.box = {}
		self.feat_box = []
		self.feat_num = 0
		self.feat_w_id = False

	def add_kv(self, prop, val):
		# val: word list
		assert type(val) == list and type(prop) == str
		if prop == 'DESC':
			self.desc = val
		elif prop == 'NAME':
			self.name = ' '.join(val)
		elif prop == 'TMPL':
			self.tmpl_desc = val
		else:
			self.box[prop] = val

	def reformat(self, vocab=None, pvocab=None, max_val_words_num=None):
		# typedesc = TypeDesc(' '.join(self.desc))
		# self.tmpl_desc = typedesc.tag().split()

		for p in self.box:
			val = self.box[p]
			for i, v in enumerate(val):
				if max_val_words_num and i == max_val_words_num:
					break
				if vocab and pvocab:
					self.feat_w_id = True
					self.feat_box.append([vocab.w2id(v), pvocab.w2id(p), i])
				else:
					self.feat_box.append([v, p, str(i)])
		if vocab:
			self.tok_desc = [vocab.w2id(w) for w in self.desc]
			self.tok_tmpl_desc = [vocab.w2id(w) for w in self.tmpl_desc]

		self.feat_num = len(self.feat_box)


class KnowledgeBase:
	def __init__(self, infobox_file, vocab_file, vocab_size=-1,
	             pid_dict=None, qid_dict=None,
	             mix_style=True, max_val_num=5, role=None):
		'''
		:param infobox_file: .infobox
		:param pid_dict: segmented id.dict
		:param qid_dict: segmented id.dict
		:param mix_style: all values under same prop mixed into one
		:param max_val_num: max value number under same prop
		'''
		assert role in ['train', 'valid', 'test']
		self.role = role
		self.p_vocab = Dictionary(cjj.ChangeFileFormat(vocab_file, '_prop.vocab'), prop_vocab=True)
		self.vocab = Dictionary(vocab_file, vocab_size)
		if pid_dict and qid_dict:
			self.pid_dict = cjj.LoadIDDict(pid_dict)
			self.qid_dict = cjj.LoadIDDict(qid_dict)
			print('*** ID dict loaded. ***')
		else:
			self.pid_dict = None
			self.qid_dict = None
		self.mix = mix_style
		self.infobox = self._load_infobox(infobox_file, max_val_num)
		print('*** All together %d/%d words, %d properties. ***' % (len(self.vocab),
		                                                            self.vocab.tot_vocab_size,
		                                                            self.p_vocab.tot_vocab_size))

	def _load_infobox(self, infobox_file, max_val_num=None):
		assert infobox_file.endswith('.infobox')
		# it's kinda important the file is strictly in accordance with the format of infobox
		# that is, property only appears once per entity
		st = time.time()

		infobox = defaultdict(InfoBox)
		with open(infobox_file) as f:
			for line in f:
				line = line.strip()
				splt = line.split('\t')
				if len(splt) != 3:
					continue
				s, p, os = splt
				vals = os.split('|||')
				if max_val_num:
					random.shuffle(vals)
					vals = vals[:max_val_num]

				if self.mix:
					vals = ' '.join([self._qid_query(v) for v in vals]).split()
				else:
					# TODO: not support separate values yet, not necessary?
					raise ValueError

				prop = self._pid_query(p)

				if not infobox.get(s):
					infobox[s] = InfoBox(s)
				infobox[s].add_kv(prop, vals)

		print('*** Infobox loaded in %s. ***' % (cjj.TimeClock(time.time() - st)))
		return infobox

	def _pid_query(self, id):
		return self.pid_dict.get(id, id) if self.pid_dict else id

	def _qid_query(self, id):
		return self.qid_dict.get(id, id) if self.qid_dict else id

	def _pad_feats(self, curr_feats, pad_tok=0, max_rows=None, need_pad=True):
		"""
		curr_feats is a bsz-len list of nrows-len list of features
		returns:
		  a bsz x max_nrows x nfeats tensor & a [bsz,] length tensor
		"""
		max_rows = max(len(feats) for feats in curr_feats) if max_rows is None else max_rows
		nfeats = len(curr_feats[0][0])
		curr_feats = list(map(lambda x: x[:max_rows], curr_feats))
		src_lengths = []
		for feats in curr_feats:
			src_lengths.append(len(feats))
			if need_pad:
				if len(feats) < max_rows:
					[feats.append([pad_tok for _ in range(nfeats)])
					 for _ in range(max_rows - len(feats))]
		return curr_feats, src_lengths

	def _pad_sents(self, sents, pad_tok=0, max_len=None, need_pad=True):
		if max_len:
			sents = list(map(lambda x: x[:max_len], sents))
		else:
			max_len = max(list(map(lambda x: len(x), sents)))
		sents = list(map(lambda x: x + [pad_tok] * (max_len - len(x)), sents)) if need_pad else sents
		return sents

	def truncate(self, max_src_len, max_val_words_num, max_tgt_len, return_id=True, need_pad=True):
		'''
		:param bsz: batch size
		:param max_src_len: max src word number
		:param max_val_words_num: max word number per value
		:param max_tgt_len: max tgt word number
		:param batch_first: [b x *]
		:return: dataloader
		'''
		assert max_val_words_num > 2 and max_tgt_len > 2
		# [bsz x nfields x nfeats=3]

		# TODO: limit vocabulary size
		for ent in self.infobox:
			if return_id:
				self.infobox[ent].reformat(self.vocab, self.p_vocab, max_val_words_num)
			else:
				self.infobox[ent].reformat(max_val_words_num=max_val_words_num)

		if need_pad:
			eos = self.vocab.w2id(self.vocab.eos_token) if return_id else self.vocab.eos_token
			bos = self.vocab.w2id(self.vocab.bos_token) if return_id else self.vocab.bos_token
			eos_pad, bos_pad = [eos], [bos]
		else:
			eos_pad, bos_pad = [], []

		boxes = [self.infobox[ent] for ent in self.infobox]
		srcs = list(map(lambda x: x.feat_box, boxes))  # list of list of feats
		if return_id:
			tgts = list(map(lambda x: bos_pad + x.tok_desc[:max_tgt_len] + eos_pad, boxes))  # []
			tmpl_tgts = list(map(lambda x: bos_pad + x.tok_tmpl_desc[:max_tgt_len] + eos_pad, boxes))  # []
		else:
			tgts = list(map(lambda x: bos_pad + x.desc[:max_tgt_len] + eos_pad, boxes))  # []
			tmpl_tgts = list(map(lambda x: bos_pad + x.tmpl_desc[:max_tgt_len] + eos_pad, boxes))  # []

		pad_tok = self.vocab.stoi[self.vocab.pad_token] if return_id else self.vocab.pad_token

		srcs, src_lengths = self._pad_feats(srcs, max_rows=max_src_len, need_pad=need_pad)
		if need_pad:
			srcs = torch.tensor(srcs)  # n x src_len x feat=3
			src_lengths = torch.tensor(src_lengths) # n
			tgts = torch.tensor(self._pad_sents(tgts, pad_tok, max_tgt_len, need_pad))  # n x tgt_len
			tmpl_tgts = torch.tensor(self._pad_sents(tmpl_tgts, pad_tok, max_tgt_len, need_pad))

		return srcs, src_lengths, tgts, tmpl_tgts

	def batchify(self, bsz, max_src_len, max_val_words_num, max_tgt_len):
		'''
		:param bsz: batch size
		:param max_src_len: max src word number
		:param max_val_words_num: max word number per value
		:param max_tgt_len: max tgt word number
		:param batch_first: [b x *]
		:return: dataloader
		'''
		st = time.time()
		srcs, src_lengths, tgts, tmpl_tgts = self.truncate(max_src_len, max_val_words_num, max_tgt_len)

		dataset = TensorDataset(srcs, src_lengths, tgts, tmpl_tgts)
		loader = DataLoader(dataset=dataset, batch_size=bsz, shuffle=True,
		                    num_workers=4, drop_last=True)
		print('*** Batchified in %s. ***' % (cjj.TimeClock(time.time() - st)))
		return loader


if __name__ == "__main__":
	kb = KnowledgeBase('infobox/data10K/data10K_train.infobox',
	                   'infobox/data10K/data10K.vocab', 10000, role='train')
	a, b, c, d = kb.truncate(max_src_len=100, max_val_words_num=20, max_tgt_len=60, return_id=False, need_pad=False)
