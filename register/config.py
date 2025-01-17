# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Config dict parse module """

import argparse
import os
from argparse import Action

import yaml

BASE_CONFIG = 'base_config'


class RingMoEConfig(dict):
    """
    A Config class is inherit from dict.

    Config class can parse arguments from a config file of yaml or a dict.

    Args:
        args (list) : config filenames
        kwargs (dict) : config dictionary list

    Example:
        test.yaml:
            a:1
        >>> cfg = RingMoEConfig('./test.yaml')
        >>> cfg.a
        1

        >>> cfg = RingMoEConfig(**dict(a=1, b=dict(c=[0,1])))
        >>> cfg.b
        {'c': [0, 1]}
    """

    def __init__(self, *args, **kwargs):
        super(RingMoEConfig, self).__init__()
        cfg_dict = {}

        # load from file
        for arg in args:
            if isinstance(arg, str):
                if arg.endswith('yaml') or arg.endswith('yml'):
                    raw_dict = RingMoEConfig._file2dict(arg)
                    cfg_dict.update(raw_dict)

        # load dictionary configs
        if kwargs is not None:
            cfg_dict.update(kwargs)

        RingMoEConfig._dict2config(self, cfg_dict)

    def __getattr__(self, key):
        """Get a object attr by its `key`

        Args:
            key (str) : the name of object attr.

        Returns:
            attr of object that name is `key`
        """
        if key not in self:
            return None

        return self[key]

    def __setattr__(self, key, value):
        """Set a object value `key` with `value`

        Args:
            key (str) : The name of object attr.
            value : the `value` need to set to the target object attr.
        """
        self[key] = value

    def __delattr__(self, key):
        """Delete a object attr by its `key`.

        Args:
            key (str) : The name of object attr.
        """
        del self[key]

    def merge_from_dict(self, options):
        """Merge options into config file.

        Args:
            options (dict): dict of configs to merge from

        Examples:
            >>> options = {'model.backbone.depth': 101,
            ...            'model.rpn_head.in_channels': 512}
            >>> cfg = RingMoEConfig(**dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
        """
        option_cfg_dict = {}
        for full_key, value in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for sub_key in key_list[:-1]:
                d.setdefault(sub_key, RingMoEConfig())
                d = d[sub_key]
            sub_key = key_list[-1]
            d[sub_key] = value
        merge_dict = RingMoEConfig._merge_a_into_b(option_cfg_dict, self)
        RingMoEConfig._dict2config(self, merge_dict)

    @staticmethod
    def _merge_a_into_b(a, b):
        """Merge dict ``a`` into dict ``b``

        Values in ``a`` will overwrite ``b``

        Args:
            a (dict) : The source dict to be merged into b.
            b (dict) : The origin dict to be fetch keys from ``a``.
        Returns:
            dict: The modified dict of ``b`` using ``a``.
        """
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                b[k] = RingMoEConfig._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    @staticmethod
    def _file2dict(filename=None):
        """Convert config file to dictionary.

        Args:
            filename (str) : config file.
        """
        if filename is None:
            raise NameError('This {} cannot be empty.'.format(filename))

        filepath = os.path.realpath(filename)
        with open(filepath) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

        # Load base config file.
        if BASE_CONFIG in cfg_dict:
            cfg_dir = os.path.dirname(filename)
            base_filenames = cfg_dict.pop(BASE_CONFIG)
            base_filenames = base_filenames if isinstance(
                base_filenames, list) else [base_filenames]

            cfg_dict_list = list()
            for base_filename in base_filenames:
                cfg_dict_item = RingMoEConfig._file2dict(
                    os.path.join(cfg_dir, base_filename))
                cfg_dict_list.append(cfg_dict_item)

            base_cfg_dict = dict()
            for cfg in cfg_dict_list:
                base_cfg_dict.update(cfg)

            # Merge config
            base_cfg_dict = RingMoEConfig._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

        return cfg_dict

    @staticmethod
    def _dict2config(config, dic):
        """Convert dictionary to config.

        Args:
            config : Config object
            dic (dict) : dictionary
        Returns:

        Exceptions:

        """
        if isinstance(dic, dict):
            for key, value in dic.items():
                if isinstance(value, dict):
                    sub_config = RingMoEConfig()
                    dict.__setitem__(config, key, sub_config)
                    RingMoEConfig._dict2config(sub_config, value)
                else:
                    config[key] = dic[key]


class ActionDict(Action):
    """Argparse action to split an option into KEY=VALUE from on the first =
    and append to dictionary. List options can be passed as comma separated
    values, i.e. 'KEY=Val1,Val2,Val3' or with explicit brackets, i.e.
    'KEY=[Val1,Val2,Val3]'.
    """

    @staticmethod
    def _parse_int_float_bool(val):
        """convert string val to int or float or bool or do nothing."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.upper() in ['TRUE', 'FALSE']:
            return val.upper == 'TRUE'
        return val

    @staticmethod
    def find_next_comma(val_str):
        """find the position of next comma in the string.

        note:
            '(' and ')' or '[' and']' must appear in pairs or not exist.
        """
        assert (val_str.count('(') == val_str.count(')')) and \
               (val_str.count('[') == val_str.count(']'))

        end = len(val_str)
        for idx, char in enumerate(val_str):
            pre = val_str[:idx]
            if ((char == ',') and (pre.count('(') == pre.count(')'))
                    and (pre.count('[') == pre.count(']'))):
                end = idx
                break
        return end

    @staticmethod
    def _parse_value_iter(val):
        """Convert string format as list or tuple to python list object
        or tuple object.

        Args:
            val (str) : Value String

        Returns:
            list or tuple

        Examples:
            >>> ActionDict._parse_value_iter('1,2,3')
            [1,2,3]
            >>> ActionDict._parse_value_iter('[1,2,3]')
            [1,2,3]
            >>> ActionDict._parse_value_iter('(1,2,3)')
            (1,2,3)
            >>> ActionDict._parse_value_iter('[1,[1,2],(1,2,3)')
            [1, [1, 2], (1, 2, 3)]
        """
        # strip ' and " and delete whitespace
        val = val.strip('\'\"').replace(" ", "")

        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            # remove start '(' and end ')'
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            # remove start '[' and end ']'
            val = val[1:-1]
        elif ',' not in val:
            return ActionDict._parse_int_float_bool(val)

        values = []
        len_of_val = len(val)
        while len_of_val > 0:
            comma_idx = ActionDict.find_next_comma(val)
            ele = ActionDict._parse_value_iter(val[:comma_idx])
            values.append(ele)
            val = val[comma_idx + 1:]
            len_of_val = len(val)

        if is_tuple:
            values = tuple(values)

        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for key_value in values:
            key, value = key_value.split('=', maxsplit=1)
            options[key] = self._parse_value_iter(value)
        setattr(namespace, self.dest, options)


def parse_args():
    """
    Parse arguments from `yaml` config file.

    Returns:
        object: argparse object.
    """
    parser = argparse.ArgumentParser("RingMo-Framework Config.")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="",
                        help='Enter the path of the model config file.')

    return parser.parse_args()
