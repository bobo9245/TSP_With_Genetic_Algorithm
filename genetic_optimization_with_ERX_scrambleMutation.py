
import torch
import random
import csv
import time
from tqdm import tqdm  # tqdm을 import합니다.

device = torch.device("mps")

def read_csv_file(file_path):
    with open(file_path, mode='r', newline='') as file:
        data = [list(map(float, row)) for row in csv.reader(file)]
        tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    return tensor_data

def path_length(cities, path):
    path_cities = cities[path]
    rolled_cities = torch.roll(path_cities, -1, dims=0)
    return torch.norm(path_cities - rolled_cities, dim=1).sum()

def create_population(num_cities, population_size):
    return torch.stack([torch.roll(torch.arange(num_cities, device=device), -i) for i in range(population_size)])

def edge_recombination_crossover(parent1, parent2):
    size = len(parent1)
    adjacency = [set() for _ in range(size)]
    for p in [parent1, parent2]:
        for i in range(size):
            adjacency[p[i]].update([p[(i-1) % size], p[(i+1) % size]])
    
    child = []
    current = parent1[random.randint(0, size-1)]
    while len(child) < size:
        child.append(current.item())
        if len(child) == size:
            break
        next_city_candidates = list(adjacency[current] - set(child))
        if next_city_candidates:
            current = min(next_city_candidates, key=lambda x: len(adjacency[x]))
        else:
            remaining = list(set(range(size)) - set(child))
            current = random.choice(remaining)
    return torch.tensor(child, device=device)

def scramble_mutation(path, mutation_rate):
    if random.random() < mutation_rate:
        size = len(path)
        start, end = sorted(random.sample(range(size), 2))
        path[start:end + 1] = path[torch.randperm(end + 1 - start) + start]
    return path

def select_parents(population, scores, tournament_size):
    tournament_indices = torch.randperm(len(population), device=device)[:tournament_size]
    tournament_scores = scores[tournament_indices]
    parents_indices = tournament_indices[tournament_scores.argsort()[:2]]
    return population[parents_indices[0]], population[parents_indices[1]]

def genetic_algorithm_tsp(cities, population_size=100, generations=500, mutation_rate=0.01, tournament_size=5):
    num_cities = cities.size(0)
    population = create_population(num_cities, population_size)
    scores = torch.stack([path_length(cities, p) for p in population])

    for generation in tqdm(range(generations), desc="Evolving Generations"):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, scores, tournament_size)
            child1 = scramble_mutation(edge_recombination_crossover(parent1, parent2), mutation_rate)
            child2 = scramble_mutation(edge_recombination_crossover(parent2, parent1), mutation_rate)
            new_population.extend([child1, child2])
        
        new_population = torch.stack(new_population)
        new_scores = torch.stack([path_length(cities, p) for p in new_population])
        combined_population = torch.cat((population, new_population))
        combined_scores = torch.cat((scores, new_scores))
        best_indices = combined_scores.argsort()[:population_size]
        population, scores = combined_population[best_indices], combined_scores[best_indices]

    best_index = scores.argmin()
    return population[best_index], scores[best_index]

# 파일에서 도시 데이터 읽기 및 유전 알고리즘 실행
cities = read_csv_file('2024_AI_TSP.csv')
start_time = time.time()
best_path, best_score = genetic_algorithm_tsp(cities)
print(f"Best path found with total distance: {best_score}")
print(f"Path: {best_path}")
print(f"Execution time: {time.time() - start_time}")
