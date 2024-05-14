import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Parameters
NUM_FISH = 20
PREDATOR_PERCEPTION_RADIUS = 70
PREDATOR_MAX_SPEED = 10
PREDATOR_CATCH_RANGE = 15
PREDATOR_REFLEX = 1.1
PREDATOR_VELCITY_SCALE = 0.5
WIDTH, HEIGHT, DEPTH = 100, 100, 100
STEPS = 100

class Genome:
    def __init__(self, perception_radius, attraction_dist, repulsion_dist, max_speed, speed_boost, acc_throttle):
        self.perception_radius = perception_radius
        self.attraction_dist = attraction_dist
        self.repulsion_dist = repulsion_dist
        self.max_speed = max_speed
        self.speed_boost = speed_boost
        self.acc_throttle = acc_throttle
        
def create_initial_population(size):
    population = []
    for _ in range(size):
        genome = Genome(
            perception_radius=random.uniform(10, 25),
            attraction_dist=random.uniform(1, 20),
            repulsion_dist=random.uniform(1, 20),
            max_speed=random.uniform(1, 2.5),
            speed_boost=random.uniform(1.1, 2.5),
            acc_throttle=random.uniform(1.1, 3.5)
        )
        population.append(genome)
    return population

def calculate_cohesion(fishes):
    positions = np.array([fish.position for fish in fishes])
    center_of_mass = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center_of_mass, axis=1)
    return np.mean(distances)

def calculate_separation(fishes):
    distances = []
    for fish in fishes:
        nearest_neighbor = fish.find_nearest_neighbor(fishes)
        if nearest_neighbor:
            distance = np.linalg.norm(fish.position - nearest_neighbor.position)
            distances.append(distance)
    return np.mean(distances) if distances else 0

def calculate_alignment(fishes):
    alignments = []
    for fish in fishes:
        nearest_neighbor = fish.find_nearest_neighbor(fishes)
        if nearest_neighbor:
            alignment = np.dot(fish.velocity / np.linalg.norm(fish.velocity),
                               nearest_neighbor.velocity / np.linalg.norm(nearest_neighbor.velocity))
            alignments.append(alignment)
    return np.mean(alignments) if alignments else 0


class Predator:
    def __init__(self):
        self.position = np.random.rand(3) * np.array([WIDTH, HEIGHT, DEPTH])
        self.velocity = np.random.randn(3) * PREDATOR_VELCITY_SCALE

    def update_position(self):
        self.position += self.velocity
        self.position = self.position % np.array([WIDTH, HEIGHT, DEPTH])

    def hunt(self, fishes):
        distances = [np.linalg.norm(fish.position - self.position) for fish in fishes]
        if distances:
            nearest_fish = fishes[np.argmin(distances)]
            distance_to_nearest = np.min(distances)
            if distance_to_nearest < PREDATOR_PERCEPTION_RADIUS:
                acceleration_vector = (nearest_fish.position - self.position) / distance_to_nearest
                self.velocity += acceleration_vector * PREDATOR_REFLEX
                # Limiting the predator's speed
                if np.linalg.norm(self.velocity) > PREDATOR_MAX_SPEED:
                    self.velocity = self.velocity / np.linalg.norm(self.velocity) * PREDATOR_MAX_SPEED
            
            if distance_to_nearest < PREDATOR_CATCH_RANGE:  # Predator catches the fish
                return nearest_fish
        return None


class Fish:
    def __init__(self, x, y, z, vx, vy, vz, genome):
        self.position = np.array([x, y, z])
        self.velocity = np.array([vx, vy, vz])
        self.genome = genome

    def update_position(self):
        self.position += self.velocity
        # Boundary conditions
        self.position = self.position % np.array([WIDTH, HEIGHT, DEPTH])

    def apply_behaviors(self, fishes, predator):
        nearest_neighbor = self.find_nearest_neighbor(fishes)
        if nearest_neighbor is not None:
            self.apply_attraction(nearest_neighbor)
            self.apply_repulsion(nearest_neighbor)
        
            self.react_to_predator(predator)
        
    def react_to_predator(self, predator):
        if np.linalg.norm(predator.position - self.position) < self.genome.perception_radius:
            # Increase speed away from the predator
            escape_direction = self.position - predator.position
            self.velocity += escape_direction / np.linalg.norm(escape_direction) * (self.genome.max_speed / self.genome.acc_throttle)
            # Limit speed
            if np.linalg.norm(self.velocity) > self.genome.max_speed * self.genome.speed_boost:
                self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.genome.max_speed * self.genome.speed_boost

    def find_nearest_neighbor(self, fishes):
        distances = [np.linalg.norm(fish.position - self.position) for fish in fishes if fish != self]
        if distances:
            nearest_neighbor = fishes[np.argmin(distances)]
            if np.min(distances) < self.genome.perception_radius:
                return nearest_neighbor
        return None

    def apply_attraction(self, neighbor):
        if np.linalg.norm(neighbor.position - self.position) > self.genome.attraction_dist:
            self.velocity += (neighbor.position - self.position) / self.genome.attraction_dist

    def apply_repulsion(self, neighbor):
        if np.linalg.norm(neighbor.position - self.position) < self.genome.repulsion_dist:
            self.velocity -= (neighbor.position - self.position) / self.genome.repulsion_dist

        # Limiting the speed
        if np.linalg.norm(self.velocity) > self.genome.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.genome.max_speed

def run_simulation_with_metrics(genome, num_fish):
    fishes = [
        Fish(
            np.random.rand() * WIDTH,
            np.random.rand() * HEIGHT,
            np.random.rand() * DEPTH,
            np.random.randn(),
            np.random.randn(),
            np.random.randn(),
            genome
        ) for _ in range(num_fish)
    ]
    predator = Predator()

    for _ in range(STEPS):
        caught_fish = predator.hunt(fishes)
        if caught_fish:
            fishes.remove(caught_fish)

        for fish in fishes:
            fish.apply_behaviors(fishes, predator)
            fish.update_position()




    return len(fishes), calculate_cohesion(fishes), calculate_separation(fishes), calculate_alignment(fishes)



def run_random_search_for_population_size(num_fish, runs=1, generations=5, population_size=10):
    generation_survival_rates = [[] for _ in range(generations)]
    generation_traits = {trait: [[] for _ in range(generations)] for trait in ['perception_radius', 'attraction_dist', 'repulsion_dist', 'max_speed', 'speed_boost', 'acc_throttle']}

    for run in range(runs):
        for generation in range(generations):
            # Generate new random population
            population = create_initial_population(population_size)

            # Running simulation and getting survival rates and metrics
            results = [run_simulation_with_metrics(genome, num_fish) for genome in population]
            survival_rates = [result[0] / num_fish * 100 for result in results]
            avg_survival_rate = np.mean(survival_rates)
            
            generation_survival_rates[generation].append(avg_survival_rate)
            
            # Storing traits for this generation
            for trait in generation_traits:
                generation_traits[trait][generation].append(np.mean([getattr(genome, trait) for genome in population]))
            
            print(f"Run {run + 1}, Fish Population Size {num_fish}, Generation {generation + 1} completed. Average Survival Rate: {avg_survival_rate:.2f}%")

        avg_survival_rates = [np.mean(gen_rates) for gen_rates in generation_survival_rates]
        avg_traits_per_generation = {trait: [np.mean(traits) for traits in generation_traits[trait]] for trait in generation_traits}
        
        print(f"Run {run + 1} completed. Average Survival Rates over all generations:")
        for gen_index, rate in enumerate(avg_survival_rates):
            print(f"  Generation {gen_index + 1}: {rate:.2f}%")
        
        # Get the best genome and its survival rate
        best_generation_index = np.argmax(avg_survival_rates)
        best_generation_traits = {trait: avg_traits_per_generation[trait][best_generation_index] for trait in generation_traits}

    return best_generation_traits, avg_survival_rates[best_generation_index]

# Running the simulation for different fish population sizes
fish_population_sizes = range(10, 31, 10)
best_results = [run_random_search_for_population_size(num_fish) for num_fish in fish_population_sizes]

best_traits_per_size = [result[0] for result in best_results]
best_survival_rates_per_size = [result[1] for result in best_results]


metrics_per_population_size = []
for idx, num_fish in enumerate(fish_population_sizes):
    avg_traits = best_traits_per_size[idx]
    avg_genome = Genome(**avg_traits)
    _, cohesion, separation, alignment = run_simulation_with_metrics(avg_genome, num_fish)
    metrics_per_population_size.append((cohesion, separation, alignment))



import matplotlib.pyplot as plt

# Plotting the Best Average Survival Rates
plt.figure(figsize=(10, 5))
plt.plot(fish_population_sizes, best_survival_rates_per_size, marker='o')
plt.title('Best Average Survival Rate vs Fish Population Size')
plt.xlabel('Fish Population Size')
plt.ylabel('Best Average Survival Rate (%)')
plt.grid(True)
plt.show()

# Plotting the Behavioral Metrics
plt.figure(figsize=(15, 5))
metrics = ['Cohesion', 'Separation', 'Alignment']
for i, metric in enumerate(metrics):
    metric_values = [m[i] for m in metrics_per_population_size]
    plt.subplot(1, 3, i + 1)
    plt.plot(fish_population_sizes, metric_values, marker='o')
    plt.title(f'{metric} vs Fish Population Size')
    plt.xlabel('Fish Population Size')
    plt.ylabel(metric)
    plt.grid(True)
plt.tight_layout()
plt.show()
