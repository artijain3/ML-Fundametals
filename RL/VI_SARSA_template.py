import sys, time, argparse
import gym
import numpy as np
from tqdm import tqdm
from lib.common_utils import TabularUtils
from lib.regEnvs import *



class Tabular_DP:
    def __init__(self, args):
        self.env = args.env # comes from openai gym
        self.gamma = 0.99
        self.theta = 1e-5
        self.max_iterations = 1000
        self.nA = self.env.action_space.n # number of all actions available
        self.nS = self.env.observation_space.n # number of all states avaiable


    def compute_q_value_cur_state(self, s, value_func):
        q_s = np.zeros(self.nA) # making q_s which is the expected reward to taking action a 
        # all each possible action a, get the action-value function
        for a in range(self.nA):
            transition_tuples = self.env.P[s][a]
            for tup in transition_tuples:
                probability = tup[0]
                next_state = tup[1]
                reward = tup[2]
                done = tup[3]
                
                q_s[a] += probability * (reward+self.gamma*value_func[next_state])
                
        return q_s


    def action_to_onehot(self, a):
        """ convert single action to onehot vector"""
        # reward a is a 1, 0s in nA 
        a_onehot = np.zeros(self.nA)
        a_onehot[a] = 1
        return a_onehot


    def value_iteration(self):
        # we have to compute q[s] in each iteration from scratch
        # and compare it with the q value in previous iteration

        # choose the optimal action and optimal value function in current state
        # output the deterministic policy with optimal value function
          
        # initialize the value function
        value_func = np.zeros(self.nS) #values of each state
        
        # initialize the policy 
        policy_func = np.zeros([self.nS, self.nA])
        
        for n_iter in range(1, self.max_iterations+1): 
            delta = 0
            for s in range(self.nS):
                value_function_prev = value_func[s] # saving previous v_s
                q_func = self.compute_q_value_cur_state(s, value_func) # getting all the rewards for actions taken being in state s
                
                v_s = max(q_func) # best reward
                value_func[s] = v_s # updating value_funcs
                
                delta = max(delta, abs(value_function_prev - v_s))
                policy_func[s] = self.action_to_onehot(int(np.where(q_func==v_s)[0][0])) # getting the optimal action
                
            if delta <= self.theta:
                break
        
        V_optimal = value_func
        policy_optimal = policy_func
                
        return V_optimal, policy_optimal

class Tabular_TD:
    def __init__(self, args):
        self.env = args.env
        self.num_episodes=10000
        self.gamma = 0.99
        self.alpha = 0.05
        self.env_nA = self.env.action_space.n
        self.env_nS = self.env.observation_space.n
        self.tabularUtils = TabularUtils(self.env)
        
    def action_to_onehot(self, a):
        """ convert single action to onehot vector"""
        # reward a is a 1, 0s in nA 
        a_onehot = np.zeros(self.env_nA)
        a_onehot[a] = 1
        return a_onehot
            
    def sarsa(self):
        """sarsa: on-policy TD control"""
        Q = np.zeros((self.env_nS, self.env_nA))
        policy_func = np.zeros([self.env_nS, self.env_nA])
        epsilon = 0.2
        
        for epi in range(self.num_episodes):
            s = self.env.reset()
            a = self.tabularUtils.epsilon_greedy_policy(Q[s], epsilon) 
            
            while True:
                new_s, reward, done, info = self.env.step(a)
                new_a = self.tabularUtils.epsilon_greedy_policy(Q[new_s], epsilon)
                
                Q[s][a] += self.alpha * (reward + self.gamma * Q[new_s, new_a] - Q[s, a])
                s = new_s
                a = new_a
                
                if done == True: #terminal state reached
                    break
                
        greedy_policy_raw = np.argmax(Q, axis=1).astype(int)
        for i in range(self.env_nS):
            policy_func[i] = self.action_to_onehot(greedy_policy_raw[i])
        
        return Q, policy_func


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        # default="FrozenLake-Deterministic-v1", 
                        default="FrozenLake-Deterministic-8x8-v1",
                        choices=[""])
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    args.env = gym.make(args.env_name, map_name="8x8")
    tabularUtils = TabularUtils(args.env)

    # test value iteration
    dp = Tabular_DP(args)
    print("================Running value iteration=====================")
    V_optimal, policy_optimal = dp.value_iteration()
    print("Optimal value function: ")
    print(V_optimal)
    print("Optimal policy: ")
    print(tabularUtils.onehot_policy_to_deterministic_policy(policy_optimal))
    print(policy_optimal)
    
    # test SARSA
    td = Tabular_TD(args)
    Q_sarsa, policy_sarsa = td.sarsa()
    print("Policy from sarsa")
    print(tabularUtils.onehot_policy_to_deterministic_policy(policy_sarsa))

    # render
    tabularUtils.render(policy_optimal)
    tabularUtils.render(policy_sarsa)