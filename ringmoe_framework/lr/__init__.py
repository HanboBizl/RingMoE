"""lr of ringmo"""
from ringmoe_framework.lr.build_lr import build_lr,build_finetune_lr
from ringmoe_framework.lr.lr_schedule import WarmUpLR, WarmUpCosineDecayV2, WarmUpCosineDecayV1,\
    WarmUpMultiStepDecay, LearningRateWiseLayer, MultiEpochsDecayLR, CosineDecayLR
