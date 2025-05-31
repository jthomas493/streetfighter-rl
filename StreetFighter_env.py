import retro
# Using gymnasium for Env and spaces for modern API compatibility
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box, MultiBinary
import numpy as np
import cv2
import random

class StreetFighter(Env):
    # Constructor now accepts savestates and render_mode
    def __init__(self, savestates=None):
        super().__init__()
        
        # Define observation and action spaces
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        
        
        # Initialize the game instance, passing render_mode to retro.make
        self.game = retro.make(
            game='StreetFighterIISpecialChampionEdition-Genesis',
            use_restricted_actions=retro.Actions.FILTERED,
        )
        
        # Store the list of savestates for random initialization
        # Ensure it's always a list, even if None is passed
        self.savestates = savestates if savestates is not None else []
        
        # Initialize attributes that will be used in step and reset
        self.previous_frame = None
        self.score = 0

    # The single, correct reset method
    # It accepts seed and options for Gymnasium API compatibility
    def reset(self, seed=None, options=None):
        # Set seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            # Note: retro.make itself doesn't directly take a seed for game start,
            # but setting Python's random and numpy's random is good practice.

        # Randomly choose a savestate if any are provided
        if self.savestates:
            selected_state = random.choice(self.savestates)
            self.game.load_state(selected_state) # retro.load_state does not return obs
            obs = self.game.get_screen() # Get the screen after loading the state
            # print(f"Loaded savestate: {selected_state}") # Optional: for debugging
        else:
            # Fallback to default game reset if no savestates are configured
            obs = self.game.reset() # retro.reset() returns the initial observation
            # print("Resetting to default game start.") # Optional: for debugging

        # Preprocess the initial observation
        obs = self.preprocess(obs)
        self.previous_frame = obs # Store for frame delta calculation

        # Reset score for the new episode
        self.score = 0

        # Gymnasium API requires returning observation and an info dictionary
        return obs, {} # Return obs and an empty info dict

    def preprocess(self, observation):
        # Grayscaling
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # Resize to 84x84
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        # Add a channel dimension (84, 84, 1)
        channels = np.reshape(resize, (84, 84, 1))
        return channels

    # The step method, returning 5 values as per Gymnasium API
    def step(self, action):
        # Take a step in the game
        # retro.step returns obs, reward, done, info (4 values)
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)

        # Calculate frame delta
        # Ensure previous_frame is initialized (should be by reset)
        if self.previous_frame is None: # Fallback, should not happen if reset is called
            self.previous_frame = np.zeros_like(obs)
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs # Update previous frame for next step

        # Reshape the reward based on score delta
        current_score = info['score']
        step_reward = current_score - self.score
        self.score = current_score # Update current score

        # Gymnasium API expects 5 return values: obs, reward, terminated, truncated, info
        # 'done' from retro typically means the episode is terminated (game over)
        terminated = done
        truncated = False # Set to True if episode ends due to time limit or other non-game-over reason

        return frame_delta, step_reward, terminated, truncated, info

    # The render method
    # It will return an rgb_array if render_mode='rgb_array' was set in __init__
    # Otherwise, it will return None (or handle display if render_mode='human')
    def render(self, *args, **kwargs):
        if self.render_mode is not None:
            return self.game.render()
        return None # Return None if no render_mode is set or if it's not 'rgb_array'



    def close(self):
        self.game.close()
