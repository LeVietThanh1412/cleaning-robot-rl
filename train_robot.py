import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from cleaning_env import CleaningRobotEnv

def print_banner():
    """In banner ngầu cho chương trình"""
    print("\n" + "="*60)
    print("🤖 CLEANING ROBOT AI TRAINING SYSTEM 🤖")
    print("="*60)
    print("🚀 Powered by Deep Reinforcement Learning")
    print("⚡ Algorithm: Proximal Policy Optimization (PPO)")
    print("🎯 Mission: Train the Ultimate Cleaning Robot")
    print("="*60 + "\n")

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """In thanh tiến trình ngầu"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def train_episode(model, env, episode_num):
    """Huấn luyện một episode và trả về reward"""
    obs = env.reset()
    episode_rewards = []
    steps = 0
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_rewards.append(reward)
        steps += 1
    
    total_reward = sum(episode_rewards)
    print(f"📈 Episode {episode_num:2d}: Reward = {total_reward:8.2f} | Steps = {steps:4d}")
    return total_reward

# Khởi tạo hệ thống
print_banner()
print("🔧 Initializing Training Environment...")
env = CleaningRobotEnv()
print("✅ Environment loaded successfully!")

print("🧠 Creating PPO Neural Network...")
model = PPO("MlpPolicy", env, verbose=0)
print("✅ AI Model initialized!")

print("\n🎮 Starting Training Sequence...")
print("-" * 60)

# Cấu hình training
rewards = []
TIMESTEPS = 100000
TRAINING_ROUNDS = 10
STEPS_PER_ROUND = TIMESTEPS // TRAINING_ROUNDS

start_time = time.time()

for i in range(TRAINING_ROUNDS):
    round_start = time.time()
    
    # Hiển thị thông tin round
    print(f"\n🔥 TRAINING ROUND {i+1}/{TRAINING_ROUNDS}")
    print(f"⏱️  Target Steps: {STEPS_PER_ROUND:,}")
    
    # Training với progress bar
    print("🚀 Learning in progress...")
    model.learn(total_timesteps=STEPS_PER_ROUND)
    
    # Test performance
    print("🎯 Testing robot performance...")
    episode_reward = train_episode(model, env, i+1)
    rewards.append(episode_reward)
    
    # Thống kê round
    round_time = time.time() - round_start
    print(f"⏰ Round completed in {round_time:.1f}s")
    
    # Progress bar tổng thể
    print_progress_bar(i+1, TRAINING_ROUNDS, prefix='Overall Progress:', suffix='Complete', length=40)

# Thống kê cuối cùng
training_time = time.time() - start_time
print(f"\n🎉 TRAINING COMPLETED!")
print("="*60)
print(f"⏱️  Total Training Time: {training_time:.1f} seconds")
print(f"📊 Average Reward: {np.mean(rewards):.2f}")
print(f"🏆 Best Reward: {np.max(rewards):.2f}")
print(f"📈 Improvement: {((rewards[-1] - rewards[0]) / abs(rewards[0]) * 100):.1f}%")
print("="*60)

# Lưu mô hình
print("\n💾 Saving AI Model...")
model.save("ppo_cleaning_robot")
print("✅ Model saved as 'ppo_cleaning_robot'")

# Lưu dữ liệu training
print("📊 Saving training data...")
np.savetxt("training_rewards.txt", rewards)
print("✅ Rewards saved to 'training_rewards.txt'")

print("\n🎯 MISSION ACCOMPLISHED!")
print("🤖 Your cleaning robot is now ready to dominate the world of cleanliness!")
print("🚀 Use visualize_learning.py to see the epic training journey!")
print("="*60 + "\n")
