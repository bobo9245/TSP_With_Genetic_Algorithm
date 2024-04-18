import torch
import csv
import time
import random

print(torch.cuda.is_available())

# GPU 설정을 CUDA로 활성화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_csv_file(file_path):
    with open(file_path, mode='r', newline='') as file:
        # CSV 데이터를 읽어 실수형 텐서로 변환
        data = torch.tensor([[float(cell) for cell in row] for row in csv.reader(file)], dtype=torch.float32).to(device)
    return data

def path_length(cities, path):
    path_cities = cities[path]
    rolled_cities = torch.roll(path_cities, -1, dims=0)
    return torch.norm(path_cities - rolled_cities, dim=1).sum()

def create_population(num_cities, population_size):
    return torch.stack([torch.roll(torch.arange(num_cities), -i) for i in range(population_size)]).to(device)

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
    tournament_indices = torch.randperm(len(population))[:tournament_size]
    tournament_scores = scores[tournament_indices]
    parents_indices = tournament_indices[torch.topk(tournament_scores, 2, largest=False).indices]
    return population[parents_indices[0]], population[parents_indices[1]]

def genetic_algorithm_tsp(cities, population_size=100, generations=500, mutation_rate=0.01, tournament_size=5):
    num_cities = cities.size(0)
    population = create_population(num_cities, population_size)
    scores = torch.tensor([path_length(cities, p) for p in population], device=device)

    for _ in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, scores, tournament_size)
            child1 = scramble_mutation(edge_recombination_crossover(parent1, parent2), mutation_rate)
            child2 = scramble_mutation(edge_recombination_crossover(parent2, parent1), mutation_rate)
            new_population.append(child1)
            new_population.append(child2)
        
        new_population = torch.stack(new_population)
        new_scores = torch.tensor([path_length(cities, p) for p in new_population], device=device)

        combined_population = torch.cat((population, new_population))
        combined_scores = torch.cat((scores, new_scores))
        elite_indices = torch.topk(combined_scores, population_size, largest=False).indices
        population, scores = combined_population[elite_indices], combined_scores[elite_indices]

    best_index = scores.argmin()
    return population[best_index], scores[best_index]

cities = read_csv_file('2024_AI_TSP.csv')
start_time = time.time()
best_path, best_score = genetic_algorithm_tsp(cities)
print(f"Best path found with total distance: {best_score}")
print(f"Path: {best_path}")
print(f"Execution time: {time.time() - start_time}")
