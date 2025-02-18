import os
import gym
import slimevolleygym
import neat
import numpy as np
import glob
import datetime
import time
import pickle
import shutil
from collections import deque

# Import the FrameStack class from the slimevolley.py file
from slimevolleygym.slimevolley import FrameStack  # Adjust the import path as necessary

# Create the gym environment
generations = 5000
env = gym.make("SlimeVolley-v0")
n_frames = 4  # Number of frames to stack
env = FrameStack(env, n_frames)  # Use the imported FrameStack

# ============================ saving ============================
# Define checkpoint directory with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = os.path.join(os.path.dirname(__file__), f"neat_checkpoints_{timestamp}")
subdirectory_name = "genomes"
subdirectory_path = os.path.join(checkpoint_dir, subdirectory_name)
os.makedirs(subdirectory_path, exist_ok=True)

def create_checkpoint_dir():
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

# ============================ saving ============================

# Global variable to track the current generation
current_generation = 0

def eval_genomes(genomes, config):
    global current_generation  # Use the global variable

    genome_list = []
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            input_data = obs.flatten()  # Flatten the stacked frames for input
            action_probs = net.activate(input_data)
            action = np.argmax(action_probs)
            action_vector = np.zeros(3)
            action_vector[action] = 1
            
            obs, reward, done, _ = env.step(action_vector)
            total_reward += reward

        genome.fitness = total_reward
        genome_list.append(genome)

    with open(os.path.join(subdirectory_path, f'genomes_generation_{current_generation}.pkl'), 'wb') as f:
        pickle.dump(genome_list, f)

def run():

    create_checkpoint_dir()

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    global current_generation  # Declare as global to modify it
    for current_generation in range(generations):
        winner = p.run(eval_genomes, 1)  # Run 1 generation at a time

        with open(os.path.join(checkpoint_dir, 'best_genome.pkl'), 'wb') as f:
            pickle.dump(winner, f)

    env.close()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-4')
    replay_file = os.path.join(local_dir, 'replay.py')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)


    shutil.copy(__file__, checkpoint_dir)
    shutil.copy(replay_file, checkpoint_dir)
    shutil.copy(config_path, checkpoint_dir)
    
    run()
