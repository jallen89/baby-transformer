import torch
import torch.nn as nn 

class RMSNorm(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.epsilon = 1e-6
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor):
        mean = torch.mean(torch.pow(x, 2),dim=-1, keepdim=True)
        rms = torch.sqrt(mean + self.epsilon)
        rms_norm = (x / rms) * self.gamma
        return rms_norm


norm = RMSNorm(2)
norm.epsilon = 0


# x = torch.tensor(
#     [
#         [[1, 1], [2, 2], [3, 3], [4, 4]],
#         [[5, 5], [6, 6], [7, 7], [8, 8]]
#     ], dtype=torch.float
# )

# expected = torch.tensor(
#     [
#         [[1, 1], [1, 1], [1, 1], [1, 1]],
#         [[1, 1], [1, 1], [1, 1], [1, 1]],
#     ], dtype=torch.float
# )

# o = norm(x)
# assert torch.equal(o, expected), f"{o} != {expected}"
