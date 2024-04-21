import torch
import random
import csv
import heapq
from tqdm import tqdm  # tqdm을 import합니다.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A* 알고리즘을 위한 우선순위 큐 항목
class Node:
    def __init__(self, city, g_score, f_score, path):
        self.city = city
        self.g_score = g_score
        self.f_score = f_score
        self.path = path

    def __lt__(self, other):
        return self.f_score < other.f_score

# A* 알고리즘 구현
def a_star(cities, start, goal):
    open_set = []
    start_node = Node(start, 0, 0, [start])
    heapq.heappush(open_set, start_node)
    visited = set()

    while open_set:
        current_node = heapq.heappop(open_set)
        current_city = current_node.city
        current_path = current_node.path
        visited.add(current_city)

        if current_city == goal:
            return current_path

        for i in range(len(cities)):
            if i not in visited:
                g_score = current_node.g_score + torch.norm(cities[current_city] - cities[i])
                f_score = g_score  # Euclidean distance is used for heuristic
                child_node = Node(i, g_score, f_score, current_path + [i])

                # Add node to open set
                heapq.heappush(open_set, child_node)

    return []  # Return empty path if goal is not reachable

# 유전 알고리즘 내에서 A* 알고리즘을 사용하는 부분
def a_star_crossover(parent1, parent2, cities):
    start = random.choice(parent1)  # 무작위 시작점 선택
    goal = random.choice(parent2)  # 무작위 목적지 선택
    path = a_star(cities, start, goal)
    return torch.tensor(path, dtype=torch.long, device=device)

# 나머지 유전 알고리즘의 함수들은 이전 코드에서 가져올 수 있습니다.

# 파일에서 도시 데이터 읽기 및 유전 알고리즘 실행
cities = read_csv_file('2024_AI_TSP.csv')
start_time = time.time()
best_path, best_score = genetic_algorithm_tsp(cities)
print(f"Best path found with total distance: {best_score}")
print(f"Path: {best_path}")
print(f"Execution time: {time.time() - start_time}")
