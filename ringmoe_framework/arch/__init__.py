"""arch of ringmo"""
from .build_arch import build_model
from .mae import build_mae, mae_vit_base_p16,\
    mae_vit_large_p16, mae_vit_huge_p14
from .simmim import build_simmim, simmim_vit_base_p16, simmim_vit_large_p16,\
    simmim_swin_base_p4_w6, simmim_swin_base_p4_w7, simmim_swin_tiny_p4_w6, simmim_swin_tiny_p4_w7
from .ringmo import build_ringmo, ringmo_vit_base_p16, ringmo_vit_large_p16,\
    ringmo_swin_base_p4_w6, ringmo_swin_base_p4_w7, ringmo_swin_tiny_p4_w6, ringmo_swin_tiny_p4_w7
from .simmim_moe import build_simmim_moe
