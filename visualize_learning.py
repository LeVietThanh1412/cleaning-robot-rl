import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu từ file log (giả sử bạn lưu reward trong "training_rewards.txt")
log_file = "training_rewards.txt"

try:
    with open(log_file, "r") as f:
        rewards = [float(line.strip()) for line in f.readlines()]
except FileNotFoundError:
    print(f"Không tìm thấy file {log_file}, dùng dữ liệu giả lập!")
    rewards = np.random.randn(100).cumsum()  # Fake data nếu file không tồn tại

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Reward")
plt.xlabel("Training Steps")
plt.ylabel("Reward")
plt.title("Training Progress - Cleaning Robot 6x6")
plt.legend()
plt.grid()
plt.show()
