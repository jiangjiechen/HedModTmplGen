"""Module defining encoders."""
from monmt.encoders.encoder import EncoderBase
from monmt.encoders.transformer import TransformerEncoder
from monmt.encoders.rnn_encoder import RNNEncoder
from monmt.encoders.cnn_encoder import CNNEncoder
from monmt.encoders.mean_encoder import MeanEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "mean": MeanEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc"]
