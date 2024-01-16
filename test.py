import torch
from bluefoglite.common import topology
import bluefoglite.torch_api as bfl
import torch.distributed as dist

bfl.init()
topo = topology.RingGraph(bfl.size())
bfl.set_topology(topo)
bfl.barrier()
print(f"[{bfl.rank()}][self]: {bfl._global_group._topology_and_weights.default_self_weight}")
print(f"[{bfl.rank()}][src ]: {bfl._global_group._topology_and_weights.default_src_weights}")
print(f"[{bfl.rank()}][dst ]: {bfl._global_group._topology_and_weights.default_dst_weights}")


def neighbor_allreduce(tensor: torch.Tensor):
    self_weight = bfl._global_group._topology_and_weights.default_self_weight
    src_weights = bfl._global_group._topology_and_weights.default_src_weights
    dst_weights = bfl._global_group._topology_and_weights.default_dst_weights
    
    recv_tensors = {neighbor: torch.zeros_like(tensor) for neighbor in src_weights.keys()}

    requests = []
    for dst, weight in dst_weights.items():
        req = dist.isend(tensor.mul(weight) , dst)
        requests.append(req)
        
    for src, weight in src_weights.items():
        req = dist.irecv(recv_tensors[src], src)
        requests.append(req)
    
    # wait for all recv to complete
    for req in requests:
        req.wait()

    # accumulate tensor
    tensor.mul_(self_weight)
    for src, weight in src_weights.items():
        tensor.add_(recv_tensors[src].mul_(weight))

tensor = torch.Tensor([bfl.rank()])
neighbor_allreduce(tensor)
print(f"[{bfl.rank()}] tensor: {tensor}")