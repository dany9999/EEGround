
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_properties(0))
print(torch.cuda.get_arch_list())  # <-- lista degli sm_xx supportati


print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_name(1))