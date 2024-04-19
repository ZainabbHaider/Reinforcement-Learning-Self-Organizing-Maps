import numpy as np
import gym
import random
import math

env1 =["SHFF", "FFFH", "FHFH", "HFFG"]
env2= ["SFFFFF", "FFFHFF", "FHFHHH", "HFFFFG"]
env3 = ['SFFHFFHH', 'HFFFFFHF', 'HFFHHFHH', 'HFHHHFFF', 'HFHHFHFF', 'FFFFFFFH', 'FHHFHFHH', 'FHHFHFFG']

selectedEnv = env3
env = gym.make('FrozenLake-v1', desc=selectedEnv, render_mode="human", is_slippery = False)
env.reset()
env.render()

# change-able parameters:
discount_factor = 0.99
delta_threshold = 0.00001
epsilon = 1

def value_iteration(env, gamma, epsilon):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the value function
    V = np.zeros(num_states)

    #Write your code to implement value iteration main loop
    while True:
        delta = 0
        # For each state, perform a full Bellman update
        for s in range(num_states):
            v = V[s]
            # Compute the value function for state s
            Qs = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    Qs[a] += prob * (reward + gamma * V[next_state])
            V[s] = max(Qs)
            delta = max(delta, abs(v - V[s]))
        # Check if convergence is reached
        if delta < epsilon:
            break

    # For each state, the policy will tell you the action to take
    policy = np.zeros(num_states, dtype=int)

    # Write your code here to extract the optimal policy from value function.
    for s in range(num_states):
        Qs = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                Qs[a] += prob * (reward + gamma * V[next_state])
        policy[s] = np.argmax(Qs)

    return policy, V


# Run value iteration
policy, V = value_iteration(env, discount_factor, delta_threshold)

# Print results
print("Optimal Value Function:")
print(V.reshape(len(selectedEnv), len(selectedEnv[0])))

print("\nOptimal Policy (0=Left, 1=Down, 2=Right, 3=Up):")
print(policy.reshape(len(selectedEnv), len(selectedEnv[0])))

# resetting the environment and executing the policy
state = env.reset()
#state = state[0]
step = 0
done = False
state = state[0]

max_steps = 100
for step in range(max_steps):

    # Getting max value against that state, so that we choose that action
    
    action = policy[state]
    new_state, reward, done, truncated, info = env.step(action) # information after taking the action
    env.render()
    if done:
        print("number of steps taken:", step)
        break

    state = new_state
    
env.close()