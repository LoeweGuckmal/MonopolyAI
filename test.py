import torch
import time

# this is a test to check if cuda is being used
x = torch.randn(10000, 10000, device='cuda')
start = time.time()
for _ in range(100):
    y = x @ x
torch.cuda.synchronize()
print("Time:", time.time() - start)
