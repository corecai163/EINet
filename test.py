
from tools.test_net import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os

import torch

if __name__ == '__main__':
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    dist_utils.init_dist("pytorch")
    # re-set gpu_ids with distributed training mode
    _, world_size = dist_utils.get_dist_info()
    args.world_size = world_size

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    # config
    config = get_config(args, logger = logger)
    # batch size
    assert config.total_bs % world_size == 0
    config.dataset.train.others.bs = config.total_bs // world_size

    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation

    assert args.local_rank == torch.distributed.get_rank() 

    test_net(args, config)
