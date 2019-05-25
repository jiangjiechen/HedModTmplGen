""" Main entry point of the ONMT library """
from __future__ import division, print_function

import monmt.inputters
import monmt.encoders
import monmt.decoders
import monmt.models
import monmt.utils
import monmt.modules
from monmt.trainer import Trainer
import sys
import monmt.utils.optimizers
monmt.utils.optimizers.Optim = monmt.utils.optimizers.Optimizer
sys.modules["monmt.Optim"] = monmt.utils.optimizers

# For Flake
__all__ = [monmt.inputters, monmt.encoders, monmt.decoders, monmt.models,
           monmt.utils, monmt.modules, "Trainer"]

__version__ = "0.8.1"
