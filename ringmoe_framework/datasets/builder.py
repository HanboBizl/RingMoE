from register import RingMoEClassFactory, RingMoEModule


def build_mask(config):
    return RingMoEClassFactory.get_instance_from_cfg(config, RingMoEModule.DATASET_MASK)
