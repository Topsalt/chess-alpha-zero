"""
Manages starting off each of the separate processes involved in ChessZero -
self play, training, and evaluation.
"""
import argparse

from logging import getLogger,disable

from .lib.logger import setup_logger
from .config import Config

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'eval', 'sl', 'uci',
            'karyotype_self', 'karyotype_opt', 'karyotype_eval']


def create_parser():
    """
    Parses each of the arguments from the command line
    :return ArgumentParser representing the command line arguments that were supplied to the command line:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("--new", help="run from new best model", action="store_true")
    parser.add_argument("--type", help="use normal setting", default="mini")
    parser.add_argument("--total-step", help="set TrainerConfig.start_total_steps", type=int)
    return parser


def setup(config: Config, args):
    """
    Sets up a new config by creating the required directories and setting up logging.

    :param Config config: config to create directories for and to set config from based on the args
    :param ArgumentParser args: args to use to control config.
    """
    config.opts.new = args.new
    if args.total_step is not None and hasattr(config.trainer, 'start_total_steps'):
        config.trainer.start_total_steps = args.total_step
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)


def start():
    """
    Starts one of the processes based on command line arguments.

    :return : the worker class that was started
    """
    parser = create_parser()
    args = parser.parse_args()
    config_type = args.type

    if args.cmd == 'uci':
        disable(999999) # plz don't interfere with uci

    # Karyotype commands automatically use the karyotype config type
    if args.cmd in ('karyotype_self', 'karyotype_opt', 'karyotype_eval'):
        config_type = 'karyotype'

    config = Config(config_type=config_type)
    setup(config, args)

    logger.info(f"config type: {config_type}")

    if args.cmd == 'self':
        from .worker import self_play
        return self_play.start(config)
    elif args.cmd == 'opt':
        from .worker import optimize
        return optimize.start(config)
    elif args.cmd == 'eval':
        from .worker import evaluate
        return evaluate.start(config)
    elif args.cmd == 'sl':
        from .worker import sl
        return sl.start(config)
    elif args.cmd == 'uci':
        from .play_game import uci
        return uci.start(config)
    elif args.cmd == 'karyotype_self':
        from .worker import karyotype_self_play
        return karyotype_self_play.start(config)
    elif args.cmd == 'karyotype_opt':
        from .worker import karyotype_optimize
        return karyotype_optimize.start(config)
    elif args.cmd == 'karyotype_eval':
        from .worker import karyotype_evaluate
        return karyotype_evaluate.start(config)
