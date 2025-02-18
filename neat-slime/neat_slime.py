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

# Create the gym environment
env = gym.make("SlimeVolley-v0")

# ============================ saving ============================
# Define checkpoint directory with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = os.path.join(os.path.dirname(__file__), f"neat_checkpoints_{timestamp}")
# Define the subdirectory path
subdirectory_name = "genomes"
subdirectory_path = os.path.join(checkpoint_dir, subdirectory_name)
# Create the subdirectory
os.makedirs(subdirectory_path, exist_ok=True)

def create_checkpoint_dir():
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

# ============================ saving ============================

# Global variable to track the current generation
current_generation = 0

def eval_genomes(genomes, config):
    global current_generation  # Use the global variable

    # Save genomes after evaluation for this generation
    genome_list = []
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            input_data = obs
            action_probs = net.activate(input_data)
            action = np.argmax(action_probs)
            action_vector = np.zeros(3)
            action_vector[action] = 1
            
            obs, reward, done, _ = env.step(action_vector)
            total_reward += reward

            # env.render()
        
        genome.fitness = total_reward
        # print(f"Genome ID: {genome_id}, Total Reward: {total_reward}")
        
        # Store genome for saving
        genome_list.append(genome)

    # Save the genomes for this generation in the same directory (in its own folder)
    with open(os.path.join(subdirectory_path, f'genomes_generation_{current_generation}.pkl'), 'wb') as f:
        pickle.dump(genome_list, f)

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    create_checkpoint_dir()

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # checkpointer = neat.Checkpointer(1, filename_prefix=os.path.join(checkpoint_dir, 'neat-checkpoint-'))
    # p.add_reporter(checkpointer)

    # Run for a specified number of generations
    global current_generation  # Declare as global to modify it
    for current_generation in range(5000):
        winner = p.run(eval_genomes, 1)  # Run 1 generation at a time
        # print('\nBest genome:\n{!s}'.format(winner))

        # Save the best genome to a file
        with open(os.path.join(checkpoint_dir, 'best_genome.pkl'), 'wb') as f:
            pickle.dump(winner, f)

    env.close()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-4')
    replay_file = os.path.join(local_dir, 'replay.py')

    # Copy the running script to the checkpoint directory
    shutil.copy(__file__ , checkpoint_dir)
    shutil.copy(replay_file, checkpoint_dir)

    # Copy the configuration file to the checkpoint directory
    shutil.copy(config_path, checkpoint_dir)
    run(config_path)
