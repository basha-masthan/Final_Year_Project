import torch

if torch.cuda.is_available():
    print(f" PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
else:
    print(" PyTorch is NOT using GPU!")
