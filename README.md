This repository contains two projects:
- CartPole Agent with Genetic Neural Networks
- RRT-Based Path Planning in C++ with OpenCV
  
Below I have mentioned detailed description on both the Projects.

# CartPole Agent with Genetic Neural Networks
This project showcases training a reinforcement learning (RL) agent for the OpenAI Gym `CartPole-v0` environment 
using a Genetic Algorithm (GA) to optimize the weights and biases of a feedforward neural network—entirely without backpropagation
(`genetic_nn.py`).

## Key Features
- Reinforcement Learning via Evolutionary Computation: Train an agent to balance the pole by evolving neural network parameters.
- Neural Network Architecture:
  - Input layer: observation space of size 4 (cart position, cart velocity, pole angle, pole angular velocity)
  - Hidden layer: 6 neurons with tanh activation
  - Output layer: 2 actions (left or right) with softmax selection
- Genetic Algorithm Components:
  - Population of candidate networks (weights & biases)
  - Fitness evaluation: run each network for one episode, total reward as fitness
  - Selection: choose top performers
  - Crossover: blend parent DNA to produce offspring
  - Mutation: random perturbations to maintain diversity
- Visualization: Plot fitness improvement over generations and track the best network’s performance

## Requirements
- Python 3.7+
- gym (OpenAI Gym)
- numpy
- matplotlib
Install dependencies via:
```bash
pip install gym numpy matplotlib
```

## How It Works

- Initialize Population
  Create pop_size random neural networks (DNA vectors of weights & biases).
- Evaluate Fitness
  For each candidate network:
  - Reset CartPole-v0 environment
  - Roll out one episode, using the network’s forward pass to select actions
  - Sum rewards—this is the candidate’s fitness
- Evolve Over Generations
  - Selection: keep top networks by fitness
  - Crossover: pair parents and mix their DNA
  - Mutation: with probability mutation_rate, randomly alter genes
  - Build new population and repeat for num_generations
- Test the Best Agent
  After evolution, run the top network in CartPole-v0 and observe its performance

## Core Components

- Neural Network (`NeuralNet class)`
  - `forward_prop(obs, w1, b1, w2, b2)`: computes action probabilities via a hidden layer and softmax
  - Utilities: `tanh, relu, softmax`
- Genetic Algorithm (`GA class`)
  - `init_weight_list`: initial population of weight matrices and biases
  - `init_fitness_list`: fitness scores for initial population
  - `crossover()`: combine parent DNA slices to produce new children
  - `mutation()`: apply Gaussian noise based on mutation_rate
  - `evolve()`: full pipeline over specified generations, tracks the best individual

- Training & Testing (`trainer()` and `test_run_env()`)
  - `trainer()`: runs GA, returns best network parameters
  - `test_run_env(params)`: loads parameters into NeuralNet and runs one episode, printing time steps and final score

## Results & Visuals
- Fitness Curve: plotted over generations—shows how average and best fitness improve
- Final Performance: best agent should achieve near-maximum reward (200) consistently

## Key Learnings

- Genetic Algorithms can effectively train RL agents when gradient-based methods are not used.
- Evolutionary approaches provide robustness to noisy and non-differentiable reward landscapes.
- Even a simple feedforward network (one hidden layer) can solve CartPole when properly evolved.

---

# RRT-Based Path Planning in C++ with OpenCV
This project provides clean and functional implementations of two popular sampling-based path planning algorithms:
- RRT* (Rapidly-exploring Random Tree Star)
- RRT-Connect (Bi-directional RRT with efficient tree connection)
Both algorithms are visualized using OpenCV, and are compatible with 2D mazes that contain clear start and goal markers.

## Algorithms Implemented
- RRT*(`rrtstar.cpp`)
  - Grows a single tree from the start.
  - Selects the lowest-cost parent among nearby nodes.
  - Rewires neighbors for optimal cost.
  - Stops when the tree first reaches the goal.
- RRT-Connect(`RrtConnect.cpp`)
  - Grows two trees: one from start and one from goal.
  - Alternates extending and connecting between the two.
  - Merges trees when a collision-free connection is possible.

## Input Maze Format
- Input should be a PNG or JPEG maze image.
- Start point: a red circle.
- Goal point: a green circle.
- Obstacles: black regions.
- Free space: white or other bright regions.

📌 Make sure both red and green circles are clearly visible and centered within free space.

## Dependencies
- C++17 or higher
- OpenCV (tested with OpenCV 4.x)
Install OpenCV (if not already installed):
```bash
brew install opencv
```

## Build & Run Instructions
- Clone the Repository
  ```bash
  git clone https://github.com/your-username/your-rrt-repo.git
  cd your-rrt-repo
  ```
- Setup CMake Build
  ```bash
  mkdir build
  cd build
  ```
- Modify CMakeLists.txt
  Edit `CMakeLists.txt` to choose which algorithm to compile:
  ```bash
  cmake_minimum_required(VERSION 3.10)
  project(MyProject)
  find_package(OpenCV REQUIRED)
  
  # Choose which file to build:
  add_executable(MyProject RrtConnect.cpp)  # or RrtStar.cpp
  
  target_link_libraries(MyProject ${OpenCV_LIBS})
  ```
- Compile
  ```bash
  cmake ..
  make
  ```
- Run
  ```bash
  ./MyProject
  ```
🔍 Make sure the maze image is named `maze1.png` and is placed in the same folder under `build`.

## Output
- A window pops up displaying the algorithm in action.
- Intermediate exploratory edges are shown.
- The final optimal path is drawn in red.

---

## License

MIT License. Free to use, modify, and distribute.

---

## 💼 Author

Aditya Kariwal
[GitHub](https://github.com/adityakariwal) • [LinkedIn](https://www.linkedin.com/in/aditya-kariwal-730a04295/)  

---
