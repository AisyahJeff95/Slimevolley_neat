import os
import gym
import slimevolleygym
import neat
import numpy as np
import pickle
import imageio
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from slimevolleygym.slimevolley import FrameStack  # Ensure the import path is correct
from slimevolleygym.visualize import draw_net  # Import the draw_net function

# Create the gym environment
env = gym.make("SlimeVolley-v0")
n_frames = 4  # Number of frames to stack
env = FrameStack(env, n_frames)  # Use the imported FrameStack

def get_next_replay_filename(local_dir):
    base_name = "replay"
    extension = ".mp4"
    counter = 1
    while True:
        filename = f"{base_name}{counter}{extension}"
        if not os.path.exists(os.path.join(local_dir, filename)):
            return filename
        counter += 1

def plot_average_fitness(stats):
    """ Plot average fitness over generations. """
    generations = range(len(stats.get_fitness_mean()))
    avg_fitness = stats.get_fitness_mean()

    plt.figure(figsize=(10, 5))
    plt.plot(generations, avg_fitness, label='Average Fitness', color='blue')
    plt.title('Average Fitness Over Generations')
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'average_fitness.png'))  # Save the plot as a file
    plt.show()  # Show the plot

def replay_best_genome():
    # Load the best genome
    local_dir = os.path.dirname(__file__)
    best_genome_path = os.path.join(local_dir, 'best_genome.pkl')
    
    with open(best_genome_path, 'rb') as f:
        best_genome = pickle.load(f)

    # Load the stats object
    stats_path = os.path.join(local_dir, 'stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # Create the neural network from the best genome
    config_path = os.path.join(local_dir, 'config-feedforward-4')  # Ensure this path is correct
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    # Draw the network structure of the best genome
    draw_net(config, best_genome, view=False, filename=os.path.join(local_dir, 'best_genome_structure'))

    obs = env.reset()
    total_reward = 0
    done = False

    # Get the next replay filename
    video_path = os.path.join(local_dir, get_next_replay_filename(local_dir))
    writer = imageio.get_writer(video_path, fps=30)  # Set frames per second

    while not done:
        input_data = obs.flatten()  # Flatten the stacked frames for input
        action_probs = net.activate(input_data)
        action = np.argmax(action_probs)
        action_vector = np.zeros(3)
        action_vector[action] = 1  # Assuming action is a one-hot vector

        obs, reward, done, _ = env.step(action_vector)
        total_reward += reward
        
        # Capture the rendered frame
        frame = env.render(mode='rgb_array')  # Get the frame as an RGB array
        writer.append_data(frame)  # Write frame to video

    print(f'Total Reward: {total_reward}')

    writer.close()  # Close the video writer

    # Plot the average fitness after replaying
    plot_average_fitness(stats)

    env.close()

if __name__ == '__main__':
    replay_best_genome()
