import os
import retro
from StreetFighter_env import StreetFighter
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
import torch as th

LOG_DIR = './logs/'
OPT_DIR = './opt/'
MODEL_PATH = './train/best_model_5460000.zip'
CHECKPOINT_DIR = './train/'
CHECK_FREQ = 10000
TOTAL_TIMESTEPS = 1000000
N_ENVS  = 4

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
# env = StreetFighter()


env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, N_ENVS, channels_order='last')


# Load pre-trained model
try:
    model = PPO("MlpPolicy", env, tensorboard_log=LOG_DIR, verbose=1)
    print(f"Successfully loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}. Initializing a new model instead.")
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1)

# Continue training the loaded model
print(f"Continuing training for {TOTAL_TIMESTEPS} timesteps using {N_ENVS} environments.")
model.learn(total_timesteps=TOTAL_TIMESTEPS)

print("Training finished.")

# Save the final trained model
final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.zip")
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")
