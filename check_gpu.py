import torch

print("=== KIỂM TRA HỆ THỐNG ===")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
    print("So GPU tim thay:", torch.cuda.device_count())
else:
    print("⚠️ DANG CHAY TREN CPU - Chua nhan GPU!")