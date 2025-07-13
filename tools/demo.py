import argparse

from easycv.config import Config, DictAction
from easydemo.registry import RUNNERS
from easydemo.runners import Runner, FPSTestRunner


def parse_args():
    parser = argparse.ArgumentParser(description='Demo Tool for Jetson')
    parser.add_argument('config', help='demo config path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    elif cfg['runner_type'] == 'FPSTestRunner':
        runner = FPSTestRunner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # run runner
    runner.run()


if __name__ == "__main__":
    main()
