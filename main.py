import pygame                      # Import the Pygame library for creating games
import random                      # Import the Random module for generating random numbers
import numpy as np                 # Import the NumPy library for numerical operations
from collections import deque     # Import the Deque class from the Collections module for creating a circular buffer
from tensorflow.keras.models import Sequential    # Import the Sequential class from Keras for building a neural network model
from tensorflow.keras.layers import Dense         # Import the Dense layer from Keras for creating fully connected layers
from tensorflow.keras.optimizers import Adam     # Import the Adam optimizer from Keras for model optimization

# Define the game constants
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800
FPS = 30

# Define the bird properties
BIRD_WIDTH = 64
BIRD_HEIGHT = 48
BIRD_X = 50
BIRD_Y = 350
BIRD_JUMP = -10

# Define the pipe properties
PIPE_WIDTH = 100
PIPE_HEIGHT = 500
PIPE_GAP = 200
PIPE_VELOCITY = 5

# Define the colors
BACKGROUND_COLOR = (255, 255, 255) # White
BIRD_COLOR = (255, 0, 0) # Red
PIPE_COLOR = (0, 255, 0) # Green

class FlappyBirdGame:
    def __init__(self):
    pygame.init()  # Initialize the Pygame library
    self.clock = pygame.time.Clock()  # Create a Pygame clock object for controlling the frame rate
    self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))  # Create a Pygame window
    pygame.display.set_caption("Flappy Bird")  # Set the window caption to "Flappy Bird"

    # Initialize game variables
    self.bird_y = BIRD_Y  # Vertical position of the bird
    self.bird_velocity = 0  # Vertical velocity of the bird
    self.score = 0  # Player's score
    self.pipes = deque()  # Queue for storing pipe pairs
    self.pipe_timer = 0  # Timer for adding new pipes
    self.game_over = False  # Flag indicating whether the game is over

    # Initialize deep Q-learning variables
    self.epsilon = 1.0  # Exploration rate for selecting random actions
    self.epsilon_decay = 0.995  # Decay rate for decreasing exploration over time
    self.epsilon_min = 0.01  # Minimum exploration rate
    self.batch_size = 32  # Batch size for training the neural network
    self.memory = deque(maxlen=2000)  # Replay memory for storing experiences
    self.gamma = 0.95  # Discount factor for future rewards in Q-learning
    self.model = self.build_model()  # Build the deep Q-learning model

    def build_model(self):
    # Build the deep Q-learning model
    model = Sequential()  # Create a sequential model object from Keras
    model.add(Dense(24, input_shape=(2,), activation="relu"))  # Add a fully connected layer with 24 units and ReLU activation function
    model.add(Dense(24, activation="relu"))  # Add another fully connected layer with 24 units and ReLU activation function
    model.add(Dense(2, activation="linear"))  # Add the output layer with 2 units (representing the Q-values) and linear activation function
    model.compile(loss="mse", optimizer=Adam(lr=0.001))  # Compile the model with mean squared error loss and Adam optimizer
    return model  # Return the compiled model

    def add_pipe(self):
    # Add a new pair of pipes to the deque
    random_gap = random.randint(150, 350)  # Generate a random gap between the pipes
    top_pipe = pygame.Rect(WINDOW_WIDTH, 0, PIPE_WIDTH, PIPE_HEIGHT)  # Create a rectangle representing the top pipe
    bottom_pipe = pygame.Rect(WINDOW_WIDTH, PIPE_HEIGHT + random_gap, PIPE_WIDTH, PIPE_HEIGHT)  # Create a rectangle representing the bottom pipe
    self.pipes.append((top_pipe, bottom_pipe))  # Add the new pipe pair to the deque

    def move_pipes(self):
    # Move the pipes to the left
    for top_pipe, bottom_pipe in self.pipes:
        top_pipe.x -= PIPE_VELOCITY  # Move the top pipe to the left by subtracting the velocity
        bottom_pipe.x -= PIPE_VELOCITY  # Move the bottom pipe to the left by subtracting the velocity

        if top_pipe.right < 0:  # If the right edge of the top pipe is off the screen
            self.pipes.popleft()  # Remove the leftmost pipe pair from the deque

    def bird_jump(self):
    # Make the bird jump
    self.bird_velocity = BIRD_JUMP  # Set the bird's velocity to the predefined jump value

    def check_collision(self):
    # Check for collision with pipes or window boundaries
    bird_rect = pygame.Rect(BIRD_X, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT)  # Create a rectangle representing the bird
    for top_pipe, bottom_pipe in self.pipes:
        if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):  # Check if the bird collides with any of the pipes
            return True
    if self.bird_y <= 0 or self.bird_y + BIRD_HEIGHT >= WINDOW_HEIGHT:  # Check if the bird collides with the top or bottom of the window
        return True
    return False  # If no collision is detected, return False

    def get_state(self):
    # Get the current state of the game
    if len(self.pipes) == 0:  # If there are no pipes
        return (WINDOW_HEIGHT, 0)  # Return a state representing the bird's position at the bottom of the screen
    else:
        next_pipe = self.pipes[0]  # Get the next pipe pair
        return (next_pipe[0].x, next_pipe[1].x)  # Return a state representing the x-coordinates of the top and bottom pipes

    def act(self, state):
    # Select an action based on the epsilon-greedy policy
    if np.random.rand() <= self.epsilon:  # With epsilon probability, select a random action
        return random.choice([0, 1])
    else:
        q_values = self.model.predict(np.array([state]))  # Predict the Q-values for the given state
        return np.argmax(q_values[0])  # Select the action with the highest Q-value

    def remember(self, state, action, reward, next_state, done):
        # Store the experience in the memory
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
    # Experience replay to train the model
    if len(self.memory) < self.batch_size:  # If there are not enough experiences in memory
        return

    batch = random.sample(self.memory, self.batch_size)  # Sample a batch of experiences from memory
    for state, action, reward, next_state, done in batch:
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
        target_f = self.model.predict(np.array([state]))  # Predict the Q-values for the current state
        target_f[0][action] = target  # Update the Q-value for the selected action
        self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)  # Train the model using the updated target Q-value

    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay  # Decay the exploration rate epsilon over time

    def update_score(self):
    # Update the score when passing through pipes
    for top_pipe, _ in self.pipes:
        if top_pipe.right == BIRD_X:  # If the bird has passed through a pipe
            self.score += 1  # Increment the score by 1

    def reset(self):
    # Reset the game state
    self.bird_y = BIRD_Y  # Reset the bird's vertical position
    self.bird_velocity = 0  # Reset the bird's velocity
    self.score = 0  # Reset the score
    self.pipes.clear()  # Clear the deque of pipes
    self.pipe_timer = 0  # Reset the pipe timer
    self.game_over = False  # Reset the game over flag

    def run(self):
    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if not self.game_over:
            # Game logic

            # Get the current state of the game
            state = self.get_state()

            # Select an action based on the epsilon-greedy policy
            action = self.act(state)
            if action == 1:
                self.bird_jump()

            # Move the pipes and update the bird's position
            self.move_pipes()
            self.bird_velocity += 1
            self.bird_y += self.bird_velocity

            # Check for collision with the boundaries or pipes
            if self.bird_y + BIRD_HEIGHT >= WINDOW_HEIGHT:
                self.game_over = True

            if self.check_collision():
                self.game_over = True

            # Add a new pipe and update the score
            if self.pipe_timer == 100:
                self.add_pipe()
                self.pipe_timer = 0
                self.update_score()
            else:
                self.pipe_timer += 1

            # Get the next state and reward
            next_state = self.get_state()
            reward = 1 if self.score > 0 else 0
            done = self.game_over

            # Store the experience in the memory
            self.remember(state, action, reward, next_state, done)

            if self.game_over:
                # Print the final score and reset the game
                print("Score:", self.score)
                self.reset()

            # Experience replay to train the model
            self.replay()

        # Draw the game elements on the screen
        self.screen.fill(BACKGROUND_COLOR)
        for top_pipe, bottom_pipe in self.pipes:
            pygame.draw.rect(self.screen, PIPE_COLOR, top_pipe)
            pygame.draw.rect(self.screen, PIPE_COLOR, bottom_pipe)
        pygame.draw.rect(self.screen, BIRD_COLOR, (BIRD_X, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT))

        # Update the display and limit the frame rate
        pygame.display.update()
        self.clock.tick(FPS)


if __name__ == "__main__":
    game = FlappyBirdGame()
    game.run()

