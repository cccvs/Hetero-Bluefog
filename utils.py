import torch
import torch.distributed as dist
import bluefoglite.torch_api as bfl

import functools
import argparse
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Union, Callable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bluefog-Lite Example on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="input batch size for training"
    )
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    args =  parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def metric_average(val):
    tensor = torch.tensor(val)
    avg_tensor = bfl.allreduce(tensor)
    return avg_tensor.item()


def broadcast_parameters(params, root_rank):
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError("invalid params of type: %s" % type(params))

    # Run asynchronous broadcasts.
    async_works = []
    for name, p in params:
        async_work = bfl.broadcast_nonblocking(p, inplace=True, root_rank=root_rank)
        async_works.append(async_work)

    # Wait for completion.
    for async_work in async_works:
        async_work.wait()


def average_parameters(params, group=None):
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError("invalid params of type: %s" % type(params))

    # Run asynchronous broadcasts.
    async_works = []
    for name, p in params:
        if torch.is_floating_point(p):
            async_work = bfl.allreduce_nonblocking(p, op=bfl.ReduceOp.AVG, inplace=True, group=group)
            async_works.append(async_work)

    # Wait for completion.
    for async_work in async_works:
        async_work.wait()


def neighbor_allreduce_lite_parameters(params, group=None):
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError("invalid params of type: %s" % type(params))

    # run neighbor_allreduce
    async_works = []
    for name, p in params:
        if torch.is_floating_point(p):
            async_work = neighbor_allreduce_lite(p)
            async_works.append(async_work)

    # Wait for completion.
    for async_work in async_works:
        async_work.wait()

def neighbor_allreduce_lite(tensor: torch.Tensor):
    self_weight = bfl._global_group._topology_and_weights.default_self_weight
    src_weights = bfl._global_group._topology_and_weights.default_src_weights
    dst_weights = bfl._global_group._topology_and_weights.default_dst_weights
    process_group = bfl._global_group._process_group
    
    recv_tensors = {neighbor: torch.zeros_like(tensor) for neighbor in src_weights.keys()}

    # for dst, weight in dst_weights.items():
    #     req = dist.isend(tensor.mul(weight), dst, process_group)
    #     requests.append(req)
        
    # for src, weight in src_weights.items():
    #     # dist.recv(recv_tensors[src], src, )
    #     req = dist.irecv(recv_tensors[src], src, process_group)
    #     requests.append(req)
    
    # # wait for all recv to complete
    # for req in requests:
    #     req.wait()

    # # accumulate tensor
    # tensor.mul_(self_weight)
    # for src, weight in src_weights.items():
    #     tensor.add_(recv_tensors[src].mul_(weight))
    op_list = []
    for dst, weight in dst_weights.items():
        op_list.append(
            dist.P2POp(dist.isend, tensor.mul(weight), peer=dst, group=process_group)
        )
    for src, tmp_tensor in recv_tensors.items():
        op_list.append(
            dist.P2POp(dist.irecv, tmp_tensor, peer=src, group=process_group)
        )
    reqs = dist.batch_isend_irecv(op_list)
    
    def post_func(
        tensor: torch.Tensor,
        recv_tensors: Dict[int, torch.Tensor],
        self_weight: float,
        src_weights: Dict[int, float],
    ) -> torch.Tensor:
        tensor.mul_(self_weight)
        for src, weight in src_weights.items():
            tensor.add_(recv_tensors[src].mul_(weight))
        return tensor

    return AsyncWork(
        reqs,
        functools.partial(
            post_func,
            tensor=tensor,
            recv_tensors=recv_tensors,
            self_weight=self_weight,
            src_weights=src_weights,
        ),
    )
    

class AsyncWork:
    def __init__(
        self,
        work: Union[dist.Work, List[dist.Work]],
        post_func: Optional[Callable] = None,
    ):
        self._work = work
        self._post_func = post_func

    def wait(self) -> Optional[Any]:
        if isinstance(self._work, Iterable):
            for w in self._work:
                w.wait()
        else:
            self._work.wait()
        if self._post_func:
            return self._post_func()
        return None