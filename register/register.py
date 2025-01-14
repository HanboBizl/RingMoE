""" Class Register Module """

import inspect
import logging


class RingMoEModule:
    """Class module type for vision pretrain"""

    def __init__(self):
        pass

    DATASET = 'dataset'
    DATASET_MASK = 'dataset_mask'
    DATASET_LOADER = 'dataset_loader'
    DATASET_SAMPLER = 'dataset_sampler'
    TRANSFORMS = 'transforms'
    ARCH = 'arch'
    BACKBONE = 'backbone'
    CLASSIFIER = 'classifier'
    CONV = 'conv'
    NORM = 'norm'
    LINEAR = 'linear'
    MOE = 'moe'
    PATCH_EMBED = 'patch_embed'
    BLOCK = 'block'
    ATTENTION = 'attention'
    LOSS = 'loss'
    OPTIMIZER = 'optimizer'
    PARALLEL = 'context'
    WRAPPER = 'wrapper'
    COMMON = 'common'


class RingMoEClassFactory:
    """
    Module class factory for ring-mo.
    """

    def __init__(self):
        pass

    registry = {}

    @classmethod
    def register(cls, module_type=RingMoEModule.COMMON, alias=None):
        """Register class into registry
        Args:
            module_type (ModuleType) :
                module type name, default ModuleType.GENERAL
            alias (str) : class alias

        Returns:
            wrapper
        """

        def wrapper(register_class):
            """Register-Class with wrapper function.

            Args:
                register_class : class need to register

            Returns:
                wrapper of register_class
            """
            class_name = alias if alias is not None else register_class.__name__
            if module_type not in cls.registry:
                cls.registry[module_type] = {class_name: register_class}
            else:
                if class_name in cls.registry[module_type]:
                    logging.warning(
                        "Duplicate class ({}) will be replaced.".format(class_name)
                    )
                cls.registry[module_type][register_class.__name__] = register_class
            return register_class

        return wrapper

    @classmethod
    def register_cls(cls, register_class, module_type=RingMoEModule.COMMON, alias=None):
        """Register class with type name.

        Args:
            register_class : class need to register
            module_type :  module type name, default ModuleType.GENERAL
            alias : class name
        """
        class_name = alias if alias is not None else register_class.__name__
        if module_type not in cls.registry:
            cls.registry[module_type] = {class_name: register_class}
        else:
            if class_name in cls.registry[module_type]:
                logging.warning(
                    "Duplicate class ({}) will be replaced.".format(class_name)
                )
            cls.registry[module_type][register_class.__name__] = register_class
        return register_class

    @classmethod
    def is_exist(cls, module_type, class_name=None):
        """Determine whether class name is in the current type group.

        Args:
            module_type : Module type
            class_name : class name

        Returns:
            True/False
        """
        if not class_name:
            return module_type in cls.registry

        registered = module_type in cls.registry and class_name in cls.registry.get(module_type)

        return registered

    @classmethod
    def get_cls(cls, module_type, class_name=None):
        """Get class

        Args:
            module_type : Module type
            class_name : class name

        Returns:
            register_class
        """
        # verify
        if not cls.is_exist(module_type, class_name):
            raise ValueError("Can't find class type {} class name {} \
            in class registry".format(module_type, class_name))

        if not class_name:
            raise ValueError(
                "Can't find class. class type = {}".format(class_name)
            )

        register_class = cls.registry.get(module_type).get(class_name)
        return register_class

    @classmethod
    def get_instance_from_cfg(cls, cfg, module_type=RingMoEModule.COMMON, default_args=None):
        """Get instance.
        Args:
            cfg (dict) : Config dict. It should at least contain the key "type".
            module_type : module type
            default_args (dict, optional) : Default initialization arguments.

        Returns:
            obj : The constructed object.
        """
        if not isinstance(cfg, dict):
            raise TypeError(
                "Cfg must be a Config, but got {}".format(type(cfg))
            )
        if 'type' not in cfg:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type",'
                'but got {}\n{}'.format(cfg, default_args)
            )
        if not (isinstance(default_args, dict) or default_args is None):
            raise TypeError(
                'default_args must be a dict or None'
                'but got {}'.format(type(default_args))
            )

        args = cfg.copy()
        if default_args is not None:
            for k, v in default_args.items():
                args.setdefault(k, v)

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = cls.get_cls(module_type, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise ValueError("Can't find class type {} class name {} \
            in class registry".format(type, obj_type))

        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)('{}: {}'.format(obj_cls.__name__, e))

    @classmethod
    def get_instance(cls, module_type=RingMoEModule.COMMON,
                     obj_type=None, args=None):
        """Get instance.
        Args:
            module_type : module type
            obj_type : class type
            args (dict) : object arguments

        Returns:
            object : The constructed object
        """
        if obj_type is None:
            raise ValueError("Class name cannot be None.")

        if isinstance(obj_type, str):
            obj_cls = cls.get_cls(module_type, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise ValueError("Can't find class type {} class name {} \
            in class registry.".format(type, obj_type))

        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)('{}: {}'.format(obj_cls.__name__, e))
