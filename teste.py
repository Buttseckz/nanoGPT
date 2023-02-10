import torch

print("Pytorch CUDA Version is ", torch.version.cuda)
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())