import numpy as np
import gym
from gym import spaces
import random
from heapq import heappop, heappush

class CleaningRobotEnv(gym.Env):
    def __init__(self):
        super(CleaningRobotEnv, self).__init__()
        self.grid_size = 6
        self.num_obstacles = 6  
        self.action_space = spaces.Discrete(4)  
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int32)
        self.reset()

    def reset(self):
        while True:
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            self.robot_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            obstacles = set()
            while len(obstacles) < self.num_obstacles:
                obs_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
                if obs_pos != tuple(self.robot_pos):
                    obstacles.add(obs_pos)
            for obs in obstacles:
                self.grid[obs] = -1  
            self.obstacles = obstacles  

            if self._is_valid_env():
                break  
    
        self.cleaned_cells = set()
        return self.grid

    def _is_valid_env(self):
        """Kiểm tra môi trường có bị cô lập không (BFS)."""
        queue = [tuple(self.robot_pos)]
        visited = set(queue)

        while queue:
            x, y = queue.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in visited:
                    if self.grid[nx, ny] != -1:
                        queue.append((nx, ny))
                        visited.add((nx, ny))
        return len(visited) == (self.grid_size * self.grid_size - self.num_obstacles)

    def step(self, action):
        """Robot tìm đường bằng A* để đến ô chưa quét gần nhất."""
        next_pos = self._a_star_find_next_move()
        if next_pos:
            self.robot_pos = list(next_pos)
        
        new_pos = tuple(self.robot_pos)
        is_new_cell = new_pos not in self.cleaned_cells

        if is_new_cell:
            self.cleaned_cells.add(new_pos)
    
        reward = 10 if is_new_cell else -1
        done = len(self.cleaned_cells) == (self.grid_size * self.grid_size - self.num_obstacles)
        if done:
            reward += 500
    
        print(f"Robot tại {self.robot_pos}, Reward: {reward}, Ô đã quét: {len(self.cleaned_cells)}/{self.grid_size * self.grid_size - self.num_obstacles}")
        return self.grid, reward, done, {}

    def _a_star_find_next_move(self):
        """Dùng A* để tìm đường đến ô chưa quét gần nhất."""
        target = self._find_nearest_uncleaned_cell()
        if not target:
            return None  # Không còn ô nào chưa quét

        start = tuple(self.robot_pos)
        heap = [(0, start)]  # (cost, vị trí)
        came_from = {start: None}
        cost_so_far = {start: 0}

        while heap:
            _, current = heappop(heap)

            if current == target:
                break  

            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[nx, ny] != -1:
                    new_cost = cost_so_far[current] + 1  

                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + self._manhattan_distance(next_pos, target)
                        heappush(heap, (priority, next_pos))
                        came_from[next_pos] = current

        return self._reconstruct_path(came_from, start, target)

    def _find_nearest_uncleaned_cell(self):
        """Tìm ô chưa quét gần nhất bằng BFS."""
        queue = [tuple(self.robot_pos)]
        visited = set(queue)

        while queue:
            x, y = queue.pop(0)
            if (x, y) not in self.cleaned_cells:
                return (x, y)  # Ô chưa quét gần nhất

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in visited:
                    if self.grid[nx, ny] != -1:
                        queue.append((nx, ny))
                        visited.add((nx, ny))
        return None  

    def _manhattan_distance(self, pos1, pos2):
        """Tính khoảng cách Manhattan."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _reconstruct_path(self, came_from, start, target):
        """Dựng lại đường đi từ A*."""
        current = target
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path[0] if path else None  

    def render(self, mode='human'):
        display_grid = self.grid.copy()
        x, y = self.robot_pos
        display_grid[x, y] = 2  
        print(display_grid)

if __name__ == "__main__":
    env = CleaningRobotEnv()
    env.reset()
    env.render()
