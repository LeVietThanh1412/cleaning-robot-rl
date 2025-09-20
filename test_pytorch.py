import torch
import gymnasium
print(torch.cuda.is_available())  # Nếu True là OK
print(torch.cuda.get_device_name(0))  # Tên GPU của bạn
print(gymnasium.__version__)  # Phiên bản gym