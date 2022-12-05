import argparse
import bz2
from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
import torch
from tqdm import tqdm

from agent import Agent
from components import ReplayMemory
from env import Env
from test import test


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path: str, disable_bzip: bool):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path: str, disable_bzip: bool):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')

# Initialization params
parser.add_argument('--image-file', type=str, default='', required=True, help='Path to load environment image')
parser.add_argument('--max-depth', type=int, default=10, help='Max depth of modeled environment. Must be 1 - 100')
parser.add_argument('--sensor-range', type=int, default=15, help='Max depth of modeled environment. Must be 5 - 100')
parser.add_argument('--sensor-angle', type=int, default=120, help='Max angle of modeled sonar beam. Must be 1 - 180')
parser.add_argument('--auxiliary-plots', action='store_true', help='Show auxiliary plots. Increases CPU load.')

parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=2022, help='Random seed')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')

# Training params
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(20e3), metavar='LENGTH',
                    help='Max episode length in game frames (0 to disable)')
parser.add_argument('--learn-start', type=int, default=int(4e3), metavar='STEPS',
                    help='Number of steps before starting training')

# Evaluation params
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=500e3, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
                    help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=1000, metavar='N',
                    help='Number of transitions to use for validating Q')

# Neural Network params
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model file (state dict)')
parser.add_argument('--architecture', type=str, default='canonical',
                    choices=['canonical', 'data-efficient', 'covnext'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--checkpoint-interval', default=5e5,
                    help='How often to checkpoint the model, defaults to 0 (never checkpoint)')

# Memory params
parser.add_argument('--memory', help='File to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true',
                    help='Don\'t zip the memory file. Not recommended (zipping is slower and much smaller)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')

# Rainbow RL params
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial std dev. of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')

# Setup
args = parser.parse_args()

print(' ' * 26 + 'Current Settings')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

results_dir = Path('F:/Rainbow/results').joinpath(args.id)
if not results_dir.exists():
    results_dir.mkdir(parents=True, exist_ok=True)

memory_file = None
if args.memory is not None:
    memory_file = results_dir.joinpath(args.memory)

np.random.seed(args.seed)
torch.manual_seed(np.random.choice(10000))

if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.choice(10000))
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    args.device = torch.device('cpu')

metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}

env = Env(args)
env.make()
env.train()

# Agent
dqn_agent = Agent(args, results_dir, env)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
    if not args.memory:
        raise ValueError('Cannot resume training without memory save path. Aborting...')
    if not memory_file.exists():
        raise ValueError(f'Could not find memory file at {memory_file.__str__()}. Aborting...')
    mem = load_memory(memory_file.__str__(), args.disable_bzip_memory)
else:
    mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

# Construct validation memory
val_mem, T, done = ReplayMemory(args, args.evaluation_size), 0, True
while T < args.evaluation_size:
    if done:
        state = env.reset()
    next_state, _, done, _ = env.step(np.random.choice(env.action_space))
    val_mem.append(state, -1, 0.0, done)
    state = next_state
    T += 1

if args.evaluate:
    dqn_agent.eval()  # Set DQN (online network) to evaluation mode
    avg_reward, avg_Q = test(args, 0, dqn_agent, val_mem, metrics, results_dir, evaluate=True)  # Test
    print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
    # Training loop
    dqn_agent.train()

    done = True
    interrupt = int(0)
    reward = 0.0

    for T in tqdm(range(1, args.T_max + 1), initial=interrupt + 1):

        T += interrupt

        if T > args.T_max:
            break

        if args.render:
            action_key = env.render()

        if done:
            state = env.reset()

        if T % args.replay_frequency == 0:
            dqn_agent.reset_noise()  # Draw a new set of noisy weights

        action = dqn_agent.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done, _ = env.step(action)  # Step
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        mem.append(state, action, reward, done)  # Append transition to memory

        # Train and test
        if T >= args.learn_start:
            # Anneal importance sampling weight β to 1
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

            if T % args.replay_frequency == 0:
                dqn_agent.learn(mem)  # Train with n-step distributional double-Q learning

            if T % args.evaluation_interval == 0:
                dqn_agent.eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = test(args, T, dqn_agent, val_mem, metrics, results_dir)  # Test
                log(
                    'T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' +
                    str(avg_reward) + ' | Avg. Q: ' + str(avg_Q)
                )

                env = Env(args)
                env.make()
                env.train()
                dqn_agent.train()  # Set DQN (online network) back to training mode

                # If memory path provided, save it
                if args.memory is not None:
                    save_memory(mem, memory_file.__str__(), args.disable_bzip_memory)

            # Update target network
            if T % args.target_update == 0:
                dqn_agent.update_target_net()

            # Checkpoint the network
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                dqn_agent.save(results_dir, 'checkpoint.pth')

        state = next_state

env.close()
