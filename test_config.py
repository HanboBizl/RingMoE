import argparse

from register import RingMoConfig, ActionDict


def parse_arguments():
    """parse train arguments"""
    parser = argparse.ArgumentParser(description='Test config')
    parser.add_argument('--config', type=str, default="", help='train config file path')
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument('--seed', default=1, help='the random seed')
    parser.add_argument(
        '--options',
        nargs='+',
        action=ActionDict,
        help='override some settings in the used config, the key-value pair'
             'in xxx=yyy format will be merged into config file')

    args_ = parser.parse_args()
    return args_


if __name__ == "__main__":
    args = parse_arguments()
    cfg = RingMoConfig(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(cfg)
