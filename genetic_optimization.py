import torch
import random
import csv
import time
from tqdm import tqdm

#window용
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#mac용
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
    return torch.stack([torch.randperm(num_cities, device=device) for _ in range(population_size)])

def initialize_pheromone_matrix(num_cities, initial_pheromone=0.1):
    return torch.full((num_cities, num_cities), initial_pheromone, device=device)

def update_pheromone_matrix(partial_solution, pheromone_matrix, decay_rate=0.1, deposit_value=0.5):
    pheromone_matrix *= (1 - decay_rate)
    path_length = len(partial_solution)
    for i in range(path_length - 1):
        from_city = partial_solution[i]
        to_city = partial_solution[i + 1]
        pheromone_matrix[from_city, to_city] += deposit_value
        pheromone_matrix[to_city, from_city] += deposit_value
    return pheromone_matrix

def ant_based_mutation(path, pheromone_matrix):
    # Introduce random mutations based on pheromone levels
    num_cities = len(path)
    for _ in range(2):  # Perform two random mutations
        i, j = random.sample(range(num_cities), 2)
        if random.random() < pheromone_matrix[path[i], path[j]]:  # Use pheromone level as the probability threshold
            path[i], path[j] = path[j], path[i]  # Swap cities
    return path

def ant_based_crossover(parent1, parent2, pheromone_matrix):
    size = len(parent1)
    child = torch.empty(size, dtype=torch.long, device=device)
    mask = torch.rand(size, device=device) > 0.5  # Randomly choose genes from either parent based on pheromone influence
    child[mask] = parent1[mask]
    child[~mask] = parent2[~mask]
    return child

def genetic_algorithm_tsp_aco(cities, population_size=100, generations=500, mutation_rate=0.01, tournament_size=5):
    num_cities = cities.size(0)
    population = create_population(num_cities, population_size)
    scores = torch.stack([path_length(cities, p) for p in population])
    pheromone_matrix = initialize_pheromone_matrix(num_cities)

    for generation in tqdm(range(generations), desc="Evolving Generations"):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, scores, tournament_size)
            child1 = ant_based_mutation(ant_based_crossover(parent1, parent2, pheromone_matrix), pheromone_matrix)
            child2 = ant_based_mutation(ant_based_crossover(parent2, parent1, pheromone_matrix), pheromone_matrix)
            new_population.extend([child1, child2])
        
        new_population = torch.stack(new_population)
        new_scores = torch.stack([path_length(cities, p) for p in new_population])
        combined_population = torch.cat((population, new_population))
        combined_scores = torch.cat((scores, new_scores))
        best_indices = combined_scores.argsort()[:population_size]
        population, scores = combined_population[best_indices], combined_scores[best_indices]

        for solution in new_population:
            update_pheromone_matrix(solution, pheromone_matrix)

    best_index = scores.argmin()
    return population[best_index], scores[best_index]

# Example execution
cities = read_csv_file('2024_AI_TSP.csv')
start_time = time.time()
best_path, best_score = genetic_algorithm_tsp_aco(cities)
print(f"Best path found with total distance: {best_score}")
print(f"Path: {best_path}")
print(f"Execution time: {time.time() - start_time}")
