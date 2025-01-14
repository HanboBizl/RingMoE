"""trainer of ringmoe_framework"""
from .wrapper import ClassificationMoeWrapper
from .trainer import build_wrapper
from .ema import EMACell
from .clip_grad import clip_by_global_norm
