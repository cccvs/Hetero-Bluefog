import numpy as np

from torchvision import datasets, transforms, models
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import bluefoglite.torch_api as bfl
from bluefoglite.common import topology
from bluefoglite.common.torch_backend import AsyncWork, BlueFogLiteGroup, ReduceOp
from model import MLP

from utils import parse_args, broadcast_parameters, metric_average

bfl.init()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Dataloader
class HeteroMNIST(torch.utils.data.Dataset):
    def __init__(self, args, root="./data/mnist/", num_clients=bfl.size(), partition="iid"):
        self.root = root
        self.num_clients = num_clients
        self.partition = partition
        self.generate_data()
        
    def generate_data(self):
        self.train_dataset = datasets.MNIST(
            root=self.root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        self.test_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        
        if self.partition == "iid":
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, num_replicas=bfl.size(), rank=bfl.rank()
            )

            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset, num_replicas=bfl.size(), rank=bfl.rank()
            )
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=args.batch_size, sampler=train_sampler
            )
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=args.batch_size, sampler=test_sampler
            )
        elif self.partition == "hetero":
            pass
        else:
            raise ValueError("partition not supported")


    def get_loader(self, split = "train"):
        if split == "train":
            return self.train_loader
        elif split == "test":
            return self.test_loader
        else:
            raise ValueError("split must be either train or test")

   
            
class PSGDTrainer(object):
    def __init__(self, args, model, optimizer, dataset):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.args = args

    def step(self):
        size = float(bfl.size())
        for param in model.parameters():
            dist.reduce(param.grad.data, dst=0, op=dist.ReduceOp.SUM)
            if bfl.rank() == 0:
                param.grad.data /= bfl.size()
        if bfl.rank() == 0:
            self.optimizer.step()
        broadcast_parameters(model.state_dict(), root_rank=0)
 
    def run(self):
        for e in range(self.args.epochs):
            self.train_one_epoch(e)
            self.test_one_epoch(e)
        
    def train_one_epoch(self, epoch):
        num_data = 0
        train_loader = self.dataset.get_loader(split="train")
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.step()
            num_data += len(data)
            if (batch_idx + 1) % self.args.log_interval == 0:
                print(
                    "[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t".format(
                        bfl.rank(),
                        epoch,
                        num_data,
                        len(train_loader.sampler),
                        100.0 * num_data / len(train_loader.sampler),
                        loss.item(),
                    )
                )

    def test_one_epoch(self, epoch):
        # We only test rank 0, because this is PSGD
        if bfl.rank() != 0:
            return
        test_loader = self.dataset.get_loader(split="test")
        self.model.eval()
        test_loss, test_accuracy, total = 0.0, 0.0, 0.0
        for data, target in test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum().item()
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            total += len(target)
        test_loss /= total
        test_accuracy /= total
        # Not need to average metric values across workers.
        # test_loss = metric_average(test_loss)
        # test_accuracy = metric_average(test_accuracy)
        print(
            "\nTest Epoch: {}\tAverage loss: {:.6f}\tAccuracy: {:.4f}%\n".format(
                epoch, test_loss, 100.0 * test_accuracy
            ),
            flush=True,
        )

if __name__ == "__main__":
    args = parse_args()
    model = MLP()
    if args.cuda:
        print("using cuda.")
        model.cuda()
        device_id = bfl.rank() % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)
    else:
        print("using cpu")
        torch.manual_seed(args.seed)
    broadcast_parameters(model.state_dict(), root_rank=0)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    dataset = HeteroMNIST(args, root="./data/mnist/", num_clients=bfl.size(), partition="iid")
    trainer = PSGDTrainer(args, model, optimizer, dataset)
    trainer.run()
    bfl.barrier()
    print(f"rank {bfl.rank()} finished.")
