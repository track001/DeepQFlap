# Flappy Bird AI using Deep Q-Learning

This repository contains an implementation of an AI agent that learns to play the popular game Flappy Bird using deep Q-learning. The agent uses a deep neural network to approximate the Q-values and learns from experience replay.

## Game Description
Flappy Bird is a simple side-scrolling game where the player controls a bird, navigating it through a series of gaps between vertical pipes. The player's objective is to keep the bird flying as long as possible without hitting any pipes or the top/bottom of the screen. The game ends if the bird collides with any obstacle.

## How the AI Works
The AI agent learns to play Flappy Bird by using a variant of Q-learning called deep Q-learning. It employs a deep neural network to approximate the Q-values, which represent the expected future rewards for each possible action in a given game state. The agent learns from its experiences by storing them in a replay memory and periodically replaying them to train the neural network.

## Functions of the Game

The Flappy Bird AI game consists of the following functions:

- `__init__()`: Initializes the game variables, Pygame, and the deep Q-learning model.
- `build_model()`: Builds the deep Q-learning model using Keras.
- `add_pipe()`: Adds a new pair of pipes to the deque.
- `move_pipes()`: Moves the pipes to the left.
- `bird_jump()`: Makes the bird jump.
- `check_collision()`: Checks for collision with pipes or window boundaries.
- `get_state()`: Gets the current state of the game.
- `act(state)`: Selects an action based on the epsilon-greedy policy.
- `remember(state, action, reward, next_state, done)`: Stores the experience in the replay memory.
- `replay()`: Performs experience replay to train the model.
- `update_score()`: Updates the score when passing through pipes.
- `reset()`: Resets the game state.
- `run()`: The main game loop that handles game logic and rendering.

## Getting Started
To run the Flappy Bird AI, follow these steps:

1. Install the necessary dependencies: `pygame`, `numpy`, and `tensorflow`.
2. Clone this repository: `git clone https://github.com/your-username/flappy-bird-ai.git`.
3. Navigate to the project directory: `cd flappy-bird-ai`.
4. Run the main script: `python flappy_bird.py`.

## DQL Pseudocode 
```python
Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1 to M do
    Initialize state s_1
    for t = 1 to T do
        With probability ϵ select random action a_t
        otherwise select a_t = argmax_a Q(s_t, a; θ_i)
        Execute action a_t in emulator and observe reward r_t and next state s_(t+1)
        Store transition (s_t, a_t, r_t, s_(t+1)) in D
        Sample a minibatch of transitions (s_j, a_j, r_j, s_(j+1)) from D
        Set target y_j = 
            r_j for terminal state s_(j+1)
            r_j + γ * max_(a') Q(s_(j+1), a'; θ_i) for non-terminal state s_(j+1)
        Perform a gradient step on (y_j - Q(s_j, a_j; θ_i))^2 with respect to θ
    end for
end for
```
<br> References:
[1] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Petersen, S. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

## Network Architecture
The neural network model used in this program follows a simple architecture:

- Input Layer: The input layer takes a state representation of the game as input, consisting of two values representing the x-coordinates of the top and bottom pipes.

- Hidden Layers: The model has two hidden layers, each with 24 units and ReLU activation function. These layers help in learning complex patterns and representations from the input.

- Output Layer: The output layer consists of two units, representing the Q-values for two possible actions (jump and no jump). It uses a linear activation function.

The model is compiled using the mean squared error (MSE) loss function and the Adam optimizer with a learning rate of 0.001. This configuration enables the model to approximate the action-value function Q(s, a) and make predictions about the expected future rewards for each action given a state.

The network architecture is designed to learn and update its weights through experience replay and gradient descent. By interacting with the game environment and storing experiences in a replay memory, the model can learn to make better decisions over time by updating its Q-values based on the rewards obtained.

## Repository Structure
- `flappy_bird.py`: The main script that contains the FlappyBirdGame class and the implementation of the AI agent using deep Q-learning.
- `README.md`: This file, providing an overview of the project.
- `LICENSE`: The license file for the project (MIT License).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
The code implementation is inspired by various online tutorials and resources on deep reinforcement learning.
<br>Good links: 
- https://www.tensorflow.org/agents/tutorials/0_intro_rl
- https://cs229.stanford.edu/proj2015/362_report.pdf
- https://aspram.medium.com/learning-flappy-bird-agents-with-reinforcement-learning-d07f31609333
- https://github.com/nikitasrivatsan/DeepLearningVideoGames
