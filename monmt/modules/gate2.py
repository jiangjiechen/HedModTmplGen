# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2019/2/23 14:11
@Contact    : mi0134sher@hotmail.com
@Description: ContextGate module
'''

import torch
import torch.nn as nn


def context_gate_factory2(gate_type, embeddings_size, decoder_size,
                          attention1_size, attention2_size, output_size):
	"""Returns the correct ContextGate class"""

	gate_types = {'source': SourceContextGate2,
	              'target': TargetContextGate2,
	              'both': BothContextGate2}

	assert gate_type in gate_types, "Not valid ContextGate type: {0}".format(
		gate_type)
	return gate_types[gate_type](embeddings_size, decoder_size, attention1_size,
	                             attention2_size, output_size)


class ContextGate2(nn.Module):
	"""
	Context gate is a decoder module that takes as input the previous word
	embedding, the current decoder state and the attention state, and
	produces a gate.
	The gate can be used to select the input from the target side context
	(decoder state), from the source context (attention state) or both.
	"""

	def __init__(self, embeddings_size, decoder_size, attention1_size,
	             attention2_size, output_size):
		super(ContextGate2, self).__init__()
		input_size = embeddings_size + decoder_size + attention1_size
		self.gate = nn.Linear(input_size, output_size, bias=True)
		self.sig = nn.Sigmoid()
		self.source1_proj = nn.Linear(attention1_size, output_size)
		self.source2_proj = nn.Linear(attention2_size, output_size)
		self.target_proj = nn.Linear(embeddings_size + decoder_size,
		                             output_size)

	def forward(self, prev_emb, dec_state, attn1_state, attn2_state):
		input_tensor1 = torch.cat((prev_emb, dec_state, attn1_state), dim=1)
		input_tensor2 = torch.cat((prev_emb, dec_state, attn2_state), dim=1)
		z1 = self.sig(self.gate(input_tensor1))
		z2 = self.sig(self.gate(input_tensor2))
		proj_source1 = self.source1_proj(attn1_state)
		proj_source2 = self.source2_proj(attn2_state)
		proj_target = self.target_proj(torch.cat((prev_emb, dec_state), dim=1))
		return z1, z2, proj_source1, proj_source2, proj_target


class SourceContextGate2(nn.Module):
	"""Apply the context gate only to the source context"""

	def __init__(self, embeddings_size, decoder_size,
	             attention1_size, attention2_size, output_size):
		super(SourceContextGate2, self).__init__()
		self.context_gate = ContextGate2(embeddings_size, decoder_size,
		                                 attention1_size, attention2_size, output_size)
		self.tanh = nn.Tanh()

	def forward(self, prev_emb, dec_state, attn1_state, attn2_state):
		z1, z2, source1, source2, target = self.context_gate(
			prev_emb, dec_state, attn1_state, attn2_state)
		return self.tanh(target + z1 * source1 + source2)


class TargetContextGate2(nn.Module):
	"""Apply the context gate only to the target context"""

	def __init__(self, embeddings_size, decoder_size,
	             attention1_size, attention2_size, output_size):
		super(TargetContextGate2, self).__init__()
		self.context_gate = ContextGate2(embeddings_size, decoder_size,
		                                 attention1_size, attention2_size, output_size)
		self.tanh = nn.Tanh()

	def forward(self, prev_emb, dec_state, attn1_state, attn2_state):
		z1, z2, source1, source2, target = self.context_gate(
			prev_emb, dec_state, attn1_state, attn2_state)
		return self.tanh(z1 * target + source1 + source2)


class BothContextGate2(nn.Module):
	"""Apply the context gate to both contexts"""

	def __init__(self, embeddings_size, decoder_size,
	             attention1_size, attention2_size, output_size):
		super(BothContextGate2, self).__init__()
		self.context_gate = ContextGate2(embeddings_size, decoder_size,
		                                 attention1_size, attention2_size, output_size)
		self.tanh = nn.Tanh()

	def forward(self, prev_emb, dec_state, attn1_state, attn2_state):
		z1, z2, source1, source2, target = self.context_gate(
			prev_emb, dec_state, attn1_state, attn2_state)
		return self.tanh((1. - z1) * target + z1 * source1 + source2)
