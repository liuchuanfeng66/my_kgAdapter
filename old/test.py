import torch 
import torch.nn as nn

a = torch.tensor([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10]])
al = [a, a]
print(al)
b = torch.stack(al, dim=0)
print(b)