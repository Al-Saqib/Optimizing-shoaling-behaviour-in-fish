# Evading the Predator: Using Genetic Algorithm to Optimize Shoaling Fish Survival

## Description

This project explores the optimization of survival strategies in mosquitofish (Gambusia holbrooki) against a predator using a novel agent-based model (ABM). The model incorporates attraction and repulsion behaviors based on the nearest neighbor, which has proven effective in smaller shoals. The primary aim is to use a genetic algorithm to evolve the behavioral parameters of the fish, enhancing their survival rate in larger shoals. The approach is benchmarked against a random search algorithm for comparative analysis.

## Installation

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install numpy matplotlib
```

## Usage
To use this project, follow these steps:
1. Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/your-repo-name.git
```
2. Navigate to the project directory:
```bash
cd your-repo-name
```
3. Run the Python scripts:
- For the genetic algorithm simulation with 10 runs:
```bash
python "fish abm model with predator. 3D Generations - 10 runs_GA.py"
```

- For the genetic algorithm simulation with varying shoal size:
```bash
python "fish abm model with predator. 3D varying shoal size_GA.py"
```

- For genetic algorithm standard deviation analysis:

```bash
python "vanilla stddev.py"
```

- For the random search algorithm simulation with 10 runs:
```bash
python "fish abm model with predator. 3D Generations - 10 runs_RS.py"
```

- For the random search algorithm simulation with varying shoal size:
```bash
python "fish abm model with predator. 3D varying shoal size_RS.py"
```

- For random search standard deviation analysis:
```bash
python "random search stddev.py"
```



## Project Structure
- fish abm model with predator. 3D Generations - 10 runs_GA.py: Script to run the genetic algorithm for 10 generations.
- fish abm model with predator. 3D varying shoal size_GA.py: Script to run the genetic algorithm with varying shoal sizes.
- vanilla stddev.py: Script for vanilla standard deviation analysis.
- fish abm model with predator. 3D Generations - 10 runs_RS.py: Script to run the random search for 10 generations.
- fish abm model with predator. 3D varying shoal size_RS.py: Script to run the random search with varying shoal sizes.
- random search stddev.py: Script for random search standard deviation analysis.

## Features
### Model Setup
- Fish Agent: Simulates the behavior of fish using 6 parameters: perception_radius, attraction_dist, repulsion_dist, max_speed, speed_boost, and acc_throttle.
- Predator Agent: Simulates the predator with 5 parameters: PREDATOR_PERCEPTION_RADIUS, PREDATOR_MAX_SPEED, PREDATOR_CATCH_RANGE, PREDATOR_REFLEX, and PREDATOR_VELCITY_SCALE.
### Algorithms
- Genetic Algorithm: Optimizes fish behavior parameters to increase survival rates.
- Random Search Algorithm: Benchmarks performance against the genetic algorithm.
### Metrics
- Cohesion: Average distance of all fish from their collective center of mass.
- Separation: Average distance between each fish and its nearest neighbor.
- Alignment: Average alignment of each fish with its nearest neighbor.
  
## Results
The project compares the outcomes of the genetic algorithm and random search algorithm, highlighting the differences in fish behaviors and survival rates under different shoal sizes and parameter optimizations.

## Contributing
If you would like to contribute to this project, please follow these guidelines:
- Fork the repository.
- Create a new branch with a descriptive name.
- Make your changes and commit them with clear and concise messages.
- Push your changes to your forked repository.
- Create a pull request detailing the changes you have made.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For questions or inquiries, please me at saqib.majumder01@gmail.com.
