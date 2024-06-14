import torch
import custom_knn
import numpy as np
import tqdm
import time
from torch.profiler import profile, record_function, ProfilerActivity


num_epochs = 100
lr = 1e-5


class CoordMLP(torch.nn.Module):
    """
    3x128x128x128x3 Linear + RELU
    """

    def __init__(self):
        super().__init__()
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor):
        return self.sequence(x)


def my_loss_fn(flowed_pc1: torch.Tensor, pc2: torch.Tensor):
    with record_function("loss_computation"):
        knn_res = custom_knn.knn_points(flowed_pc1.half(), pc2.half())
    return knn_res.dists.mean().float()


def make_pcs(n_pcs: int) -> list[torch.Tensor]:
    # Seed the random number generator
    torch.manual_seed(69420)
    np.random.seed(69420)

    def _make_pc():
        # Make between 50,000 and 90,000 points
        n_points = np.random.randint(50000, 90000)
        random_offset = np.random.rand(3) * 2 - 1
        pc = torch.randn(
            n_points, 3, requires_grad=True, device="cuda", dtype=torch.float32
        ) + torch.tensor(random_offset, device="cuda", dtype=torch.float32)
        return pc

    return [_make_pc() for _ in range(n_pcs)]


pcs = make_pcs(20)
model = CoordMLP().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in tqdm.tqdm(range(num_epochs)):
        for pc1, pc2 in zip(pcs, pcs[1:]):
            optimizer.zero_grad()
            flowed_pc1 = model(pc1)
            loss = my_loss_fn(flowed_pc1, pc2)

            loss.backward()
            optimizer.step()

# Save the profiling results to a file
with open(f"profiling_results_{time.time()}.txt", "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export the profiling results to a Chrome trace file
prof.export_chrome_trace("profiler_trace.json")
