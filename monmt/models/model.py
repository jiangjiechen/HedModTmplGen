""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
	"""
	Core trainable object in OpenNMT. Implements a trainable interface
	for a simple, generic encoder + decoder model.

	Args:
	  encoder (:obj:`EncoderBase`): an encoder object
	  decoder (:obj:`RNNDecoderBase`): a decoder object
	"""

	def __init__(self, encoder, decoder):
		super(NMTModel, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, src, tgt, lengths, bptt=False):
		"""Forward propagate a `src` and `tgt` pair for training.
		Possible initialized with a beginning decoder state.

		Args:
			src (:obj:`Tensor`):
				a source sequence passed to encoder.
				typically for inputs this will be a padded :obj:`LongTensor`
				of size `[len x batch x features]`. however, may be an
				image or other generic input depending on encoder.
			tgt (:obj:`LongTensor`):
				 a target sequence of size `[tgt_len x batch]`.
			lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
			bptt (:obj:`Boolean`):
				a flag indicating if truncated bptt is set. If reset then
				init_state

		Returns:
			(:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

				 * decoder output `[tgt_len x batch x hidden]`
				 * dictionary attention dists of `[tgt_len x batch x src_len]`
		"""
		tgt = tgt[:-1]  # exclude last target from inputs

		enc_state, memory_bank, lengths = self.encoder(src, lengths)
		if bptt is False:
			self.decoder.init_state(src, memory_bank, enc_state)
		dec_out, attns = self.decoder(tgt, memory_bank,
		                              memory_lengths=lengths)
		return dec_out, attns, memory_bank, lengths


class Model2(nn.Module):
	def __init__(self, encoder2, decoder2):
		super(Model2, self).__init__()
		self.encoder = encoder2
		self.decoder = decoder2

	def forward(self, src2, tgt, lengths2,
	            src1=None, memory_bank1=None, lengths1=None, bptt=False):
		tgt = tgt[:-1]  # exclude last target from inputs

		enc_state2, memory_bank2, lengths2 = self.encoder(src2)
		if bptt is False:
			self.decoder.init_state(src2, memory_bank2, enc_state2)
		dec_out, attns = self.decoder(tgt, memory_bank1=memory_bank1, memory_bank2=memory_bank2,
		                              memory_lengths1=lengths1, memory_lengths2=lengths2)
		return dec_out, attns


class JointModel(nn.Module):
	def __init__(self, encoder1, encoder2, decoder1, decoder2):
		super(JointModel, self).__init__()
		self.encoder1 = encoder1
		self.decoder1 = decoder1
		self.encoder2 = encoder2
		self.decoder2 = decoder2

	def forward(self, src, tmpl, src2, tgt, lengths1, lengths2, bptt=False):
		_tgt1 = tmpl[:-1]
		enc1_state, memory_bank1, _lengths1 = self.encoder1(src, lengths1)
		if bptt is False:
			self.decoder1.init_state(src, memory_bank1, enc1_state)
		dec1_out, attns1 = self.decoder1(_tgt1, memory_bank1,
		                                 memory_lengths=_lengths1)
		enc2_state, memory_bank2, _lengths2 = self.encoder2(src2)
		_tgt2 = tgt[:-1]
		if bptt is False:
			self.decoder2.init_state(src2, memory_bank2, enc2_state)
		dec2_out, attns2 = self.decoder2(_tgt2, memory_bank1=memory_bank1, memory_bank2=memory_bank2,
		                                 memory_lengths1=_lengths1, memory_lengths2=_lengths2)
		return dec1_out, attns1, dec2_out, attns2
