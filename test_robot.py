import pygame
import numpy as np
from stable_baselines3 import PPO
from cleaning_env import CleaningRobotEnv

# Khởi tạo pygame
pygame.init()
CELL_SIZE = 100
GRID_SIZE = 6
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
FONT = pygame.font.Font(None, 30)

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

def draw_grid(screen, env, score):
    screen.fill(WHITE)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (x, y) in env.obstacles:
                pygame.draw.rect(screen, BLACK, rect)  # Vẽ vật cản
            elif (x, y) in env.cleaned_cells:
                pygame.draw.rect(screen, GREEN, rect)  # Ô đã quét sạch
            pygame.draw.rect(screen, GRAY, rect, 1)  # Viền ô
    
    # Vẽ robot
    robot_x, robot_y = env.robot_pos
    pygame.draw.circle(screen, BLUE, (robot_y * CELL_SIZE + CELL_SIZE // 2, robot_x * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
    
    # Hiển thị điểm số
    score_text = FONT.render(f"Score: {score:.1f}", True, RED)
    screen.blit(score_text, (10, 10))

    pygame.display.update()

# Load mô hình đã train
env = CleaningRobotEnv()
model = PPO.load("ppo_cleaning_robot")

# Khởi tạo cửa sổ hiển thị
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cleaning Robot Visualization")
clock = pygame.time.Clock()

done = False
obs, _ = env.reset()  # Unpack tuple (obs, info)
score = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)  # Unpack 5 values
    done = terminated or truncated
    score += reward

    draw_grid(screen, env, score)
    clock.tick(5)  # Giữ FPS ổn định

    if done:
        pygame.time.delay(1000)  # Dừng 1s trước khi reset
        obs, _ = env.reset()  # Unpack tuple (obs, info)
        score = 0
        done = False  # Reset trạng thái
