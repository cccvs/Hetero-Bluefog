import argparse
import torch
import bluefoglite.torch_api as bfl


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