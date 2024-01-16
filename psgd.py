import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

import bluefoglite.torch_api as bfl
from model import MLP
from dataset import HeteroMNIST


from utils import parse_args, broadcast_parameters


bfl.init()

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
        # No need to average metric values across workers.
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
    dataset = HeteroMNIST(args, root="./data/mnist/", num_clients=bfl.size(), rank=bfl.rank(), partition="iid")
    trainer = PSGDTrainer(args, model, optimizer, dataset)
    trainer.run()
    bfl.barrier()
    print(f"rank {bfl.rank()} finished.")
