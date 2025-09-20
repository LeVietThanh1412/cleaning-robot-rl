# 🤖 Cleaning Robot RL

Dự án này xây dựng môi trường và huấn luyện robot quét dọn tự động bằng thuật toán Reinforcement Learning (PPO) trên lưới 6x6, sử dụng Python, Gymnasium, Stable-Baselines3, PyTorch, Pygame và Matplotlib.

## 📦 Yêu cầu
- Python >= 3.8 (rcm 3.12)
- Các thư viện: numpy, gymnasium, stable-baselines3, matplotlib, pygame, torch

Cài đặt nhanh:
```bash
pip install -r requirements.txt
```

## 🚀 Chạy huấn luyện
```bash
python train_robot.py
```
- Mô hình sẽ được lưu vào file `ppo_cleaning_robot.zip`.
- Reward từng bước sẽ lưu vào `training_rewards.txt`.

## 🎮 Chạy mô phỏng robot
```bash
python test_robot.py
```
- Hiển thị giao diện Pygame mô phỏng robot quét dọn trên lưới, vật cản, ô đã quét sạch.

## 📊 Visualize quá trình học
```bash
python visualize_learning.py
```
- Vẽ biểu đồ reward theo thời gian.

## 🗂️ Cấu trúc file chính
- `cleaning_env.py`: Định nghĩa môi trường Gym cho robot quét dọn.
- `train_robot.py`: Huấn luyện agent bằng PPO.
- `test_robot.py`: Mô phỏng robot đã học trên giao diện Pygame.
- `visualize_learning.py`: Vẽ biểu đồ reward.
- `requirements.txt`: Danh sách thư viện cần thiết.

## �️ Công nghệ sử dụng (Technical Stack)

### **Machine Learning & AI:**
- **Reinforcement Learning (RL)**: Proximal Policy Optimization (PPO) algorithm
- **Deep Learning Framework**: PyTorch 2.0+
- **RL Library**: Stable-Baselines3 (SB3) - state-of-the-art RL algorithms
- **Environment Framework**: Gymnasium (OpenAI Gym) - standard RL environment interface

### **Programming & Development:**
- **Programming Language**: Python 3.8+ (recommended 3.12)
- **Scientific Computing**: NumPy - numerical computations and array operations
- **Data Visualization**: Matplotlib - plotting training progress and performance metrics
- **Game Development**: Pygame - real-time visualization and simulation interface

### **AI/ML Concepts Applied:**
- **Markov Decision Process (MDP)**: Environment modeling
- **Policy Gradient Methods**: PPO for continuous learning
- **Value Function Approximation**: Neural network-based value estimation
- **Exploration vs Exploitation**: Balancing learning strategies
- **Reward Engineering**: Designing incentive systems for optimal behavior

### **Software Engineering:**
- **Version Control**: Git & GitHub
- **Package Management**: pip, requirements.txt
- **Virtual Environment**: Python venv
- **Code Organization**: Modular design patterns

## �📚 Tài liệu tham khảo
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PyTorch](https://pytorch.org/)

---
Nếu gặp lỗi hoặc cần hỗ trợ, hãy liên hệ hoặc tạo issue trên repo!  
**Author**: LeVietThanh