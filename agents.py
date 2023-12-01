'''
Classes for implementing the learning methods.
'''
import numpy as np

dash_line = '-'*20

class Agent :
    '''
    Defines the basic methods for the agent.
    '''

    def __init__(self, parameters:dict):
        self.parameters = parameters
        self.nS = self.parameters['nS']
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.epsilon = self.parameters['epsilon']
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.policy = np.ones((self.nS, self.nA)) * 1/self.nA
        self.Q = np.zeros((self.nS, self.nA))
        self.seed = None

    def make_decision(self):
        '''
        Agent makes a decision according to its model.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.nA)
        state = self.states[-1]
        weights = [self.policy[state, action] for action in range(self.nA)]
        action = np.random.choice(range(self.nA), p = weights)
        return action

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]

    def reset(self):
        '''
        Resets the agent for a new simulation.
        '''
        self.restart()
        self.policy = np.ones((self.nS, self.nA)) * 1/self.nA
        self.Q = np.zeros((self.nS, self.nA))

    def max_Q(self, s):
        '''
        Determines the max Q value in state s
        '''
        return max([self.Q[s, a] for a in range(self.nA)])

    def argmaxQ(self, s):
        '''
        Determines the action with max Q value in state s
        '''
        maxQ = self.max_Q(s)
        opt_acts = [a for a in range(self.nA) if self.Q[s, a] == maxQ]
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.choice(opt_acts)

    def update_policy(self, s):
        opt_act = self.argmaxQ(s)
        prob_epsilon = lambda action: 1 - self.epsilon if action == opt_act else self.epsilon/(self.nA-1)
        self.policy[s] = [prob_epsilon(a) for a in range(self.nA)]

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        TO BE DEFINED BY SUBCLASS
        '''
        pass


class MC(Agent) :
    '''
    Implements a learning rule with Monte Carlo optimization.
    '''

    def __init__(self, parameters:dict):
        super().__init__(parameters)
        self.first_visit = self.parameters['first_visit']
        self.N = np.zeros((self.nS, self.nA))
        self.debug = False
   
    def restart(self):
        super().restart()
        self.N = np.zeros((self.nS, self.nA))

    def reset(self):
        super().reset()
        self.N = np.zeros((self.nS, self.nA))

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        '''
        if done:
            rewards = [r for r in self.rewards] + [reward]
            T = len(rewards) - 1
            G = 0
            for t in range(T - 1, -1, -1):
                reward = rewards[t+1]
                G  = self.gamma*G + reward
                state = self.states[t]
                if (not self.first_visit) or state not in self.states[:t]:
                    action = self.actions[t]
                    self.N[state, action] += 1
                    prev_Q = self.Q[state, action]
                    self.Q[state, action] += 1/self.N[state, action] * (G - self.Q[state, action])
                    if self.debug:
                        print('')
                        print(dash_line)
                        print(f'Learning log round {t}:')
                        print(f'state:{state}')
                        print(f'action:{action}')
                        print(f'reward:{reward}')
                        print(f'G:{G}')
                        print(f'Previous Q:{prev_Q}')
                        print(f'New Q:{self.Q[state, action]}')
            for s in range(self.nS):
                self.update_policy(s)



