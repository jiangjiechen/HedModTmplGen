# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2018/11/17 15:03
@Contact    : Mi0134sher@hotmail.com
@Description: induce head-modifier templates from type descriptions
'''

import sys
sys.path.append('..')
import cjjpy as cjj
from stanfordcorenlp import StanfordCoreNLP

# need to start the corenlp first
nlp = StanfordCoreNLP('../corenlp/stanford-corenlp-full-2017-06-09')

max_phrs_len = 5

punctuation = {'.', '!', ',', ';', ':', '?', '--', '-rrb-', '-lrb-'}

# from wikipedia
prepositions = {
	'aboard', 'about', 'above', 'absent', 'across', 'after', 'against', 'along', 'alongside', 'amid', 'among',
	'apropos', 'apud', 'around', 'as', 'astride', 'at', 'atop', 'bar', 'before', 'behind', 'below', 'beneath',
	'beside', 'besides', 'between', 'beyond', 'but', 'by', 'chez', 'circa', 'come', 'despite', 'down', 'during',
	'except', 'for', 'from', 'in', 'inside', 'into', 'less', 'like', 'minus', 'near', 'notwithstanding', 'of',
	'off', 'on', 'onto', 'opposite', 'out', 'outside', 'over', 'pace', 'past', 'per', 'plus', 'post', 'pre',
	'pro', 'qua', 're', 'sans', 'save', 'short', 'since', 'than', 'through', 'throughout', 'till', 'to', 'toward',
	'under', 'underneath', 'unlike', 'until', 'unto', 'up', 'upon', 'upside', 'versus', 'via', 'vice', 'aboard',
	'about', 'above', 'absent', 'across', 'after', 'against', 'along', 'alongside', 'amid', 'among', 'apropos',
	'apud', 'around', 'as', 'astride', 'at', 'atop', 'bar', 'before', 'behind', 'below', 'beneath', 'beside',
	'besides',
	'between', 'beyond', 'but', 'by', 'chez', 'circa', 'come', 'despite', 'down', 'during', 'except', 'for', 'from',
	'in',
	'inside', 'into', 'less', 'like', 'minus', 'near', 'notwithstanding', 'of', 'off', 'on', 'onto', 'opposite', 'out',
	'outside', 'over', 'pace', 'past', 'per', 'plus', 'post', 'pre', 'pro', 'qua', 're', 'sans', 'save', 'short',
	'since',
	'than', 'through', 'throughout', 'till', 'to', 'toward', 'under', 'underneath', 'unlike', 'until', 'unto', 'up',
	'upon',
	'upside', 'versus', 'via', 'vice', 'with', 'within', 'without', 'worth'}

splitters = {'and', ',', 'or', 'of', 'for', '--', 'also'}

goodsplitters = {',', 'of', 'for', '--', 'also'}  # leaves out and and or

stop_words = prepositions.union(splitters).union(cjj.LoadWords('stop_words_en.txt'))

def splitphrs(tokes, l, r, max_phrs_len, labelist):
	if r - l <= max_phrs_len:
		labelist.append((l, r, 0))
	else:
		i = r - 1
		found_a_split = False
		while i > l:
			if tokes[i] in goodsplitters or tokes[i] in prepositions:
				splitphrs(tokes, l, i, max_phrs_len, labelist)
				if i < r - 1:
					splitphrs(tokes, i + 1, r, max_phrs_len, labelist)
				found_a_split = True
				break
			i -= 1
		if not found_a_split:  # add back in and and or
			i = r - 1
			while i > l:
				if tokes[i] in splitters or tokes[i] in prepositions:
					splitphrs(tokes, l, i, max_phrs_len, labelist)
					if i < r - 1:
						splitphrs(tokes, i + 1, r, max_phrs_len, labelist)
					found_a_split = True
					break
				i -= 1
		if not found_a_split:  # just do something
			i = r - 1
			while i >= l:
				max_len = min(max_phrs_len, i - l + 1)
				labelist.append((i - max_len + 1, i + 1, 0))
				i = i - max_len


def stupid_search(tokes, fields):
	"""
	greedily assigns longest labels to spans from right to left
	"""
	PFL = 4
	labels = []
	i = len(tokes)
	wordsets = [set(toke for toke in v if toke not in punctuation) for k, v in fields.items()]
	pfxsets = [set(toke[:PFL] for toke in v if toke not in punctuation) for k, v in fields.items()]

	while i > 0:
		matched = False
		if tokes[i - 1] in punctuation:
			labels.append((i - 1, i, 0))  # all punctuation
			i -= 1
			continue
		if tokes[i - 1] in punctuation or tokes[i - 1] in prepositions or tokes[i - 1] in splitters:
			i -= 1
			continue
		for j in range(i):
			if tokes[j] in punctuation or tokes[j] in prepositions or tokes[j] in splitters:
				continue
			# then check if it matches stuff in the table
			tokeset = set(toke for toke in tokes[j:i] if toke not in punctuation)

			for vset in wordsets:
				if tokeset == vset or (tokeset.issubset(vset) and len(tokeset) > 1):
					if i - j > max_phrs_len:
						nugz = []
						splitphrs(tokes, j, i, max_phrs_len, nugz)
						labels.extend(nugz)
					else:
						labels.append((j, i, 0))
					i = j
					matched = True
					break
			if matched:
				break
			pset = set(toke[:PFL] for toke in tokes[j:i] if toke not in punctuation)
			for pfxset in pfxsets:
				if pset == pfxset or (pset.issubset(pfxset) and len(pset) > 1):
					if i - j > max_phrs_len:
						nugz = []
						splitphrs(tokes, j, i, max_phrs_len, nugz)
						labels.extend(nugz)
					else:
						labels.append((j, i, 0))
					i = j
					matched = True
					break
			if matched:
				break
		if not matched:
			i -= 1
	labels.sort(key=lambda x: x[0])
	return labels


class TypeDesc:
	def __init__(self, type_desc):
		assert len(type_desc) > 0
		self.type_desc = type_desc.lower()
		self.words = nlp.word_tokenize(type_desc)
		self.depparse = [(i[0], i[1] - 1, i[2] - 1) for i in nlp.dependency_parse(type_desc)]
		self.count = len(self.words)
		self.child_dict = self._build_child_dict()

		self.mod_ids = []
		self.hed_ids = []

		self.OTHER_TAG = 0
		self.HEADW_TAG = 1
		self.MODW_TAG = 2

		self.head_placeholder = "<hed>"
		self.mod_placeholder = "<mod>"

		self.roletag = [self.OTHER_TAG] * self.count

	def __str__(self):
		return self.type_desc

	def __repr__(self):
		return self.type_desc

	def find_heads_id(self):
		self.root_id = self.child_dict[-1]['ROOT'][0]
		conj_ids = self.child_dict[self.root_id].get('conj', [])
		conj_ids.insert(0, self.root_id)
		for i in conj_ids:
			self.roletag[i] = self.HEADW_TAG
		self.hed_ids = conj_ids

	def find_mods_id(self, infobox):
		if infobox:
			# TODO: needs further design
			box = self._convert_infobox(infobox)
			mod_spans = stupid_search(self.words, box)
			for span in mod_spans:
				start = span[0]
				end = span[1]
				for i in range(start, end):
					self.mod_ids.append(i)
					self.roletag[i] = self.MODW_TAG
		else:
			clause_words = []
			sub_dict = self.child_dict[self.root_id]
			for k in sub_dict:
				if 'acl' in k:
					clause_words += sub_dict[k]
			for i, w in enumerate(self.words):
				if w not in stop_words and w not in self.hed_ids and i not in clause_words:
					self.roletag[i] = self.MODW_TAG
					self.mod_ids.append(i)


	def tag(self, box=None):
		self.find_heads_id()
		self.find_mods_id(box)
		# for idx, w in enumerate(self.words):
		# 	if w in stop_words:
		# 		self.roletag[idx] = self.STOPW_TAG
		for i in self.hed_ids:
			self.roletag[i] = self.HEADW_TAG
		for i in self.mod_ids:
			if self.roletag[i] == self.OTHER_TAG:
				self.roletag[i] = self.MODW_TAG
		tag_list = []
		for i in range(self.count):
			if i in self.hed_ids:
				tag_list.append(self.head_placeholder)
			elif i in self.mod_ids:
				tag_list.append(self.mod_placeholder)
			else:
				tag_list.append(self.words[i])
		tag_desc = ''
		last_w = ''
		for w in tag_list:
			if w != last_w:
				tag_desc += w + ' '
			last_w = w
		return tag_desc.strip()

	def show_all(self):
		print([(i, j) for i, j in enumerate(self.words)])
		print(self.depparse)

	def _convert_infobox(self, box):
		# TODO: convert infobox
		'''
		:param box: {prop: val(|||)*}+
		:return:
		'''
		return box

	def _build_child_dict(self):
		child_dict = {}
		for i in range(self.count):
			child_dict[i] = {}
		child_dict[-1] = {}
		for dp in self.depparse:
			if child_dict[dp[1]].get(dp[0]):
				child_dict[dp[1]][dp[0]].append(dp[2])
			else:
				child_dict[dp[1]][dp[0]] = [dp[2]]
		return child_dict


def test():
	text = ['scientific article that is published in September 2004',
	        'street in paris, france']
	for t in text:
		sent = TypeDesc(t)
		print('*'*20)
		sent.show_all()
		s = sent.tag()
		print(s)


if __name__ == '__main__':
	test()
