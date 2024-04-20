import torch
import heapq
import time
import csv
# PyTorch를 사용하여 MPS 활성화
device = torch.device("mps")

def read_csv_file(file_path):
    with open(file_path, mode='r', newline='') as file:
        data = [list(map(float, row)) for row in csv.reader(file)]
        tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    return tensor_data

def euclidean_distance(a, b):
    return torch.norm(a - b)

def heuristic(path, cities):
    # 현재 경로에서 마지막 도시와 첫 도시 사이의 유클리드 거리 반환
    return euclidean_distance(cities[path[-1]], cities[path[0]])

def a_star_tsp(cities):
    num_cities = cities.size(0)
    paths = []
    # 우선순위 큐 초기화 (휴리스틱 값, 현재 경로, 현재 경로의 길이)
    priority_queue = [(0, [0], 0)]
    
    while priority_queue:
        current_f, current_path, current_g = heapq.heappop(priority_queue)
        
        if len(current_path) == num_cities and current_path[0] == current_path[-1]:
            # 모든 도시를 방문하고 시작점으로 돌아온 경로
            return current_path, current_g
        
        last_city = current_path[-1]
        for next_city in range(num_cities):
            if next_city not in current_path:
                new_path = current_path + [next_city]
                new_g = current_g + euclidean_distance(cities[last_city], cities[next_city])
                new_h = heuristic(new_path, cities)
                new_f = new_g + new_h
                heapq.heappush(priority_queue, (new_f, new_path, new_g))
                
    return [], float('inf')  # 경로를 찾지 못한 경우

# CSV 파일에서 도시 데이터 읽기
cities = read_csv_file('2024_AI_TSP.csv')

start_time = time.time()
best_path, best_score = a_star_tsp(cities)
print(f"Best path found with total distance: {best_score}")
print(f"Path: {best_path}")
print(f"Execution time: {time.time() - start_time}")
