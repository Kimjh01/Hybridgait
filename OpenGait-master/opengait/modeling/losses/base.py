from ctypes import ArgumentError
import torch.nn as nn
import torch
from utils import Odict
import functools
from utils import ddp_all_gather
import torch.distributed as dist

def _dist_on():
    return dist.is_available() and dist.is_initialized()

def gather_and_scale_wrapper(func):
    """
    Decorator:
    - 분산학습이면 → 입력 텐서를 all_gather 하고 loss 를 world_size 로 스케일
    - 단일 GPU이면 → 그대로 실행
    """
    @functools.wraps(func)
    def inner(*args, **kwds):
        try:
            if _dist_on():                          # ✅ 분산 여부 체크
                # 각 키워드 인수가 Tensor이면 all_gather
                kwds = {k: ddp_all_gather(v) if torch.is_tensor(v) else v
                        for k, v in kwds.items()}
                loss, info = func(*args, **kwds)
                loss = loss * torch.distributed.get_world_size()
            else:                                   # ✅ 단일 GPU 우회
                loss, info = func(*args, **kwds)
            return loss, info
        except Exception as e:
            raise ArgumentError from e
    return inner


class BaseLoss(nn.Module):
    """
    Base class for all losses.

    Your loss should also subclass this class.
    """

    def __init__(self, loss_term_weight=1.0):
        """
        Initialize the base class.

        Args:
            loss_term_weight: the weight of the loss term.
        """
        super(BaseLoss, self).__init__()
        self.loss_term_weight = loss_term_weight
        self.info = Odict()

    def forward(self, logits, labels):
        """
        The default forward function.

        This function should be overridden by the subclass. 

        Args:
            logits: the logits of the model.
            labels: the labels of the data.

        Returns:
            tuple of loss and info.
        """
        return .0, self.info
