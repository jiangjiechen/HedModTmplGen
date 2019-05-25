"""Define a minimal encoder."""
import torch.nn as nn
from monmt.modules import GlobalSelfAttention
from monmt.encoders.encoder import EncoderBase


class MeanEncoder(EncoderBase):
	"""A trivial non-recurrent encoder. Simply applies mean pooling.

	Args:
	   num_layers (int): number of replicated layers
	   embeddings (:obj:`monmt.modules.Embeddings`): embedding module to use
	"""

	def __init__(self, num_layers, embeddings, emb_size,
	             attn_hidden, dropout=0.0, attn_type='general', coverage_attn=False):
		super(MeanEncoder, self).__init__()
		self.num_layers = num_layers
		self.embeddings = embeddings
		self.dropout = nn.Dropout(p=dropout)
		self.attn = GlobalSelfAttention(emb_size, coverage=coverage_attn,
		                                attn_type=attn_type, attn_hidden=attn_hidden)

	def forward(self, src, lengths=None):
		"See :obj:`EncoderBase.forward()`"
		self._check_args(src, lengths)

		emb = self.dropout(self.embeddings(src))
		_, batch, emb_dim = emb.size()
		dec_out, p_attn = self.attn(emb.transpose(0, 1).contiguous(), emb.transpose(0, 1),
		                            memory_lengths=lengths)
		mean = dec_out.mean(0).expand(self.num_layers, batch, emb_dim)
		memory_bank = dec_out
		encoder_final = mean # GRU
		return encoder_final, memory_bank, lengths
