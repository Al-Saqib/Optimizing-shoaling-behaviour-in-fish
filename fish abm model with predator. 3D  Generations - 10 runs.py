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

# roulette wheel selection
def select(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probabilities = [fitness / total_fitness for fitness in fitness_scores]

    selected_population = []
    for _ in range(len(population) // 2):
        selected_population.append(population[np.random.choice(len(population), p=selection_probabilities)])
    
    return selected_population



def crossover(parent1, parent2):
    # Randomly select two crossover points
    crossover_points = sorted(random.sample(range(6), 2))

    # Create offspring by combining genes from parents
    child1_genes = [
        parent1.perception_radius, parent1.attraction_dist, parent1.repulsion_dist,
        parent1.max_speed, parent1.speed_boost, parent1.acc_throttle
    ]
    child2_genes = [
        parent2.perception_radius, parent2.attraction_dist, parent2.repulsion_dist,
        parent2.max_speed, parent2.speed_boost, parent2.acc_throttle
    ]

    # Swap the genes between the crossover points
    child1_genes[crossover_points[0]:crossover_points[1]], child2_genes[crossover_points[0]:crossover_points[1]] = \
    child2_genes[crossover_points[0]:crossover_points[1]], child1_genes[crossover_points[0]:crossover_points[1]]

    # Create children from the gene lists
    child1 = Genome(*child1_genes)
    child2 = Genome(*child2_genes)

    return child1, child2


def mutate(genome, mutation_rate=0.1):
    if random.random() < mutation_rate:
        genome.perception_radius *= random.uniform(0.9, 1.1)
        genome.attraction_dist *= random.uniform(0.9, 1.1)
        genome.repulsion_dist *= random.uniform(0.9, 1.1)
        genome.max_speed *= random.uniform(0.9, 1.1)
        genome.speed_boost *= random.uniform(0.9, 1.1)
        genome.acc_throttle *= random.uniform(0.9, 1.1)


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

def run_simulation_with_genome(genome):
    fishes = [
        Fish(
            np.random.rand() * WIDTH,
            np.random.rand() * HEIGHT,
            np.random.rand() * DEPTH,
            np.random.randn(),
            np.random.randn(),
            np.random.randn(),
            genome
        ) for _ in range(NUM_FISH)
    ]
    predator = Predator()

    for _ in range(STEPS):
        caught_fish = predator.hunt(fishes)
        if caught_fish:
            fishes.remove(caught_fish)

        for fish in fishes:
            fish.apply_behaviors(fishes, predator)
            fish.update_position()

    return len(fishes)

# ... Selection, Crossover, Mutation Functions ...

# Updated function to run the genetic algorithm, collect data, and print progress messages

def run_genetic_algorithm_for_analysis_with_logging(runs=1, generations=30, population_size=20):
    # Data structures to store traits and survival rates for each generation across all runs
    traits_data = {
        'perception_radius': [[] for _ in range(generations)],
        'attraction_dist': [[] for _ in range(generations)],
        'repulsion_dist': [[] for _ in range(generations)],
        'max_speed': [[] for _ in range(generations)],
        'speed_boost': [[] for _ in range(generations)],
        'acc_throttle': [[] for _ in range(generations)],
        'survival_rate': [[] for _ in range(generations)]
    }

    best_genomes = []  # To store the final best genome of each run for the correlation analysis

    for run in range(runs):
        population = create_initial_population(population_size)

        
        for generation in range(generations):
            # Calculate survival rate
            survival_rates = [run_simulation_with_genome(genome) / NUM_FISH * 100 for genome in population]
            avg_survival_rate = np.mean(survival_rates)
            traits_data['survival_rate'][generation].append(avg_survival_rate)


            # Record traits for this generation
            for trait in traits_data.keys():
                if trait != 'survival_rate':
                    traits_data[trait][generation].append(np.mean([getattr(genome, trait) for genome in population]))

            # Selection, Crossover, Mutation
            parents = select(population, survival_rates)  # Assuming survival rate is used as fitness score
            next_generation = []
            for _ in range(len(population) // 2):
                parent1, parent2 = random.sample(parents, 2)
                offspring1, offspring2 = crossover(parent1, parent2)
                mutate(offspring1)
                mutate(offspring2)
                next_generation.extend([offspring1, offspring2])
            population = next_generation
            
            
            
            # Logging progress
            print(f"Run {run + 1}/{runs}, Generation {generation + 1}/{generations} completed.")
            

            
        


    # Calculating average values for each trait for each generation across all runs
    average_traits_over_generations = {
        trait: [np.mean(generation_values) for generation_values in traits_data[trait]]
        for trait in traits_data
    }
    
    # Identify the generation with the highest average survival rate
    best_generation_index = np.argmax(average_traits_over_generations['survival_rate'])

    # Extract the average traits for the best generation
    best_generation_traits = {trait: average_traits_over_generations[trait][best_generation_index] for trait in traits_data if trait != 'survival_rate'}
    
    best_generation_avg_survival_rate = average_traits_over_generations['survival_rate'][best_generation_index]
    

    return average_traits_over_generations, best_generation_traits, best_generation_avg_survival_rate

# Running the modified genetic algorithm to collect data for analysis with logging
average_traits_data, best_generation_traits, best_generation_avg_survival_rate = run_genetic_algorithm_for_analysis_with_logging(runs=10, generations=10, population_size=16)

print("Average Traits Data:" + str(average_traits_data))

# Plotting the distribution of each trait and survival rate across generations


# Plotting traits
# Number of subplots required (6 traits + 1 survival rate)
num_plots = len(average_traits_data)
num_rows = (num_plots // 3) + (num_plots % 3 > 0)  # Calculate number of rows needed

# Create subplots
fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
fig.suptitle('Average Traits and Survival Rate Across Generations', fontsize=16)

# Flatten the axes array for easy indexing
axs = axs.flatten()

# Plot each trait in a separate subplot
for i, (trait, values) in enumerate(average_traits_data.items()):
    axs[i].plot(values, label=trait.replace('_', ' ').title())
    axs[i].set_title(trait.replace('_', ' ').title())
    axs[i].set_xlabel('Generation')
    axs[i].set_ylabel('Average Value')
    axs[i].legend()

# Hide any unused subplots
for j in range(i + 1, len(axs)):
    axs[j].set_visible(False)

plt.tight_layout(pad=4.0)
plt.show()



# Calculate the average for each trait per generation
avg_traits_per_generation = {
    trait: [np.mean(generation_values) if generation_values else np.nan for generation_values in average_traits_data[trait]]
    for trait in average_traits_data


}# Converting to DataFrame for correlation analysis
df_traits_per_generation = pd.DataFrame(average_traits_data)

# Calculate the correlation matrix
correlation_matrix = df_traits_per_generation.corr()

# Generate the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Average Traits and Survival Rate Per Generation')
plt.show()


print("Best Generation's Average Traits:")
for trait, value in best_generation_traits.items():
    print(f"{trait}: {value}")
print(f"Average Survival Rate: {best_generation_avg_survival_rate}%")