import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def neighbor_allreduce(tensor, rank, neighbors):
    recv_tensors = {neighbor: torch.zeros_like(tensor) for neighbor in neighbors}

    requests = []
    for neighbor in neighbors:
        # 向每个邻居发送tensor的副本
        req = dist.isend(tensor.clone(), neighbor)
        requests.append(req)
        # 从每个邻居接收数据
        dist.recv(recv_tensors[neighbor], neighbor)

    # 等待所有通信完成
    for req in requests:
        req.wait()

    # 将收到的数据累加到本地tensor上
    for neighbor in neighbors:
        tensor.add_(recv_tensors[neighbor])

def main(rank, world_size):
    setup(rank, world_size)

    # 定义图的邻接矩阵
    adjacency_matrix = [
        [1/3, 1/3, 0.0, 1/3],  # Node 0 is connected to Node 1 and Node 3
        [1/3, 1/3, 1/3, 0.0],  # Node 1 is connected to Node 0 and Node 2
        [0.0, 1/3, 1/3, 1/3],  # Node 2 is connected to Node 1 and Node 3
        [1/3, 0.0, 1/3, 1/3]   # Node 3 is connected to Node 0 and Node 2
    ]

    # 为当前节点确定邻居
    neighbors = [i for i in range(world_size) if adjacency_matrix[rank][i] == 1]

    tensor = torch.ones(1) * (rank + 1)
    print(f'Rank {rank}, before allreduce: {tensor}')

    neighbor_allreduce(tensor, rank, neighbors, world_size)
    print(f'Rank {rank}, after allreduce: {tensor}')


