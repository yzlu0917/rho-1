import logging

import torch

# 创建一个 logger
if torch.distributed.is_initialized():
    logging.basicConfig(
        format="%(asctime)s - "
               + f"Rank: {torch.distributed.get_rank()}"
               + " - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
else:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
logger = logging.getLogger(__name__)


def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def print_rank_0(msg, rank=None):
    if rank is not None:
        logger.info(f"[Rank {rank}] {msg}")  # 附加 rank 信息
    elif is_rank_0():
        logger.info(msg)


def get_rank_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return -1
