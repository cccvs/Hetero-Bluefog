from torchvision import datasets, transforms
import torch



class HeteroMNIST(torch.utils.data.Dataset):
    def __init__(self, args, rank, num_clients, root="./data/mnist/", partition="iid"):
        self.args = args
        self.rank = rank
        self.num_clients = num_clients
        self.root = root
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
                self.train_dataset, num_replicas=self.num_clients, rank=self.rank
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset, num_replicas=self.num_clients, rank=self.rank
            )
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.args.batch_size, sampler=train_sampler
            )
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.args.batch_size, sampler=test_sampler
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
