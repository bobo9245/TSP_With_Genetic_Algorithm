import numpy as np
import random
import csv
import time

def read_csv_file(file_path):
    with open(file_path, mode='r', newline='') as file:
        return np.array(list(csv.reader(file)), dtype=float)

def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def path_length(cities, path):
    path_cities = cities[path]
    return np.sum(np.linalg.norm(path_cities - np.roll(path_cities, -1, axis=0), axis=1))

def create_population(num_cities, population_size):
    return np.array([np.roll(np.arange(num_cities), -i) for i in range(population_size)])

def edge_recombination_crossover(parent1, parent2):
    size = len(parent1)
    adjacency = {i: set() for i in range(size)}
    for p in [parent1, parent2]:
        for i in range(size):
            adjacency[p[i]].update([p[(i-1) % size], p[(i+1) % size]])
    
    child = []
    current = random.choice(parent1)
    while len(child) < size:
        child.append(current)
        if len(child) == size:
            break
        next_city_candidates = list(adjacency[current] - set(child))
        if next_city_candidates:
            current = min(next_city_candidates, key=lambda x: len(adjacency[x]))
        else:
            remaining = list(set(range(size)) - set(child))
            current = random.choice(remaining)
    return np.array(child)

def scramble_mutation(path, mutation_rate):
    if random.random() < mutation_rate:
        size = len(path)
        start, end = sorted(random.sample(range(size), 2))
        path[start:end + 1] = np.random.permutation(path[start:end + 1])
    return path

def select_parents(population, scores, tournament_size):
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament = population[tournament_indices]
    tournament_scores = scores[tournament_indices]
    parents_indices = tournament_indices[np.argsort(tournament_scores)[:2]]
    return population[parents_indices[0]], population[parents_indices[1]]

def genetic_algorithm_tsp(cities, population_size=100, generations=500, mutation_rate=0.01, tournament_size=5):
    num_cities = len(cities)
    population = create_population(num_cities, population_size)
    scores = np.array([path_length(cities, p) for p in population])

    for _ in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, scores, tournament_size)
            child1, child2 = scramble_mutation(edge_recombination_crossover(parent1, parent2), mutation_rate), scramble_mutation(edge_recombination_crossover(parent2, parent1), mutation_rate)
            new_population.extend([child1, child2])
        new_population = np.array(new_population)
        new_scores = np.array([path_length(cities, p) for p in new_population])
        combined_population = np.concatenate((population, new_population))
        combined_scores = np.concatenate((scores, new_scores))
        best_indices = np.argsort(combined_scores)[:population_size]
        population, scores = combined_population[best_indices], combined_scores[best_indices]

    best_index = np.argmin(scores)
    return population[best_index], scores[best_index]

cities = read_csv_file('2024_AI_TSP.csv')

start_time = time.time()
best_path, best_score = genetic_algorithm_tsp(cities)
print(f"Best path found with total distance: {best_score}")
print(f"Path: {best_path}")
print(f"Execution time: {time.time() - start_time}")
