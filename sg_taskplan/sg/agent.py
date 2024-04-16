import numpy as np
import torch
from .model import LeaderQNet, FollowerQNet


class Leader:
    """
    This class implemets leader related functions.
    """
    def __init__(self, parameters, env) -> None:
        env_prop = env.get_task_info()
        self.dims = env_prop['dims']
        self.dimAl, self.dimAf = env_prop['dimAl'], env_prop['dimAf']
        self.dimal, self.dimaf = env_prop['dimal'], env_prop['dimaf']

        self.rng = np.random.default_rng(parameters['seed'])
        self.device = parameters['device']

        # double q learning parameter
        self.lr = parameters['leader']['learning_rate']
        self.mom = parameters['leader']['momentum']     # only for SGD
        self.gam = parameters['leader']['reward_decay']
        self.eps = parameters['leader']['epsilon']
        self.tau = parameters['leader']['tau']
        
        self.onlineQ = LeaderQNet(parameters, self.dims, self.dimAl, self.dimAf)
        self.targetQ = LeaderQNet(parameters, self.dims, self.dimAl, self.dimAf)
        self.onlineQ.to(self.device)
        self.targetQ.to(self.device)

    
    def get_action_from_Qonline(self):
        '''
        Get the optimal action from online Q-funciton.
        '''
        s = s.float() if torch.is_tensor(s) else torch.from_numpy(s).float()
        with torch.no_grad():
            Qnet = self.onlineQ(s)
        al = Qnet.argmax().item() - 1   # get action from index
        return al


    def get_action_from_policy(self, pil):
        '''
        Get an action from a mixed strategy. 
        '''
        al = self.rng.choice(np.arange(-1, self.dimAl-1), p=pil)    # action set = [-1, ..., dimAl-1]
        return al
    
    
    def get_action_from_epsgreedy(self, al, eps=None):
        '''
        Get the action from eps greedy policy.
        '''
        if eps is None:
            eps = self.eps
        if eps == 0:
            return al
        
        # generate eps greedy policy
        p = np.ones(self.dimAl) * self.eps / (self.dimAl-1)
        p[al+1] = 1 - eps

        al_greedy = self.rng.choice(np.arange(-1, self.dimAl-1), p=p)   # action set = [-1, ..., dimAl-1]
        return al_greedy


    def update_target_Q_parameter(self, tau=None):
        """
        This function replaces the target Q network parameters
        """
        if tau is None:
            tau = self.tau
        d_online = self.onlineQ.state_dict()
        d_target = self.targetQ.state_dict()
        for key in d_online.keys():
            d_target[key] = (1-tau) * d_target[key] + tau * d_online[key]
        
        self.targetQ.load_state_dict(d_target)
        return
        


class Follower:
    """
    This class implements follower related functions.
    """
    def __init__(self, parameters, env) -> None:
        env_prop = env.get_task_info()
        self.dims = env_prop['dims']
        self.dimAl, self.dimAf = env_prop['dimAl'], env_prop['dimAf']
        self.dimal, self.dimaf = env_prop['dimal'], env_prop['dimaf']

        self.rng = np.random.default_rng(parameters['seed'])
        self.device = parameters['device']

        # double q learning parameter
        self.lr = parameters['follower']['learning_rate']
        self.mom = parameters['follower']['momentum']     # only for SGD
        self.gam = parameters['follower']['reward_decay']
        self.eps = parameters['follower']['epsilon']
        self.tau = parameters['follower']['tau']

        self.onlineQ = FollowerQNet(parameters, self.dims, self.dimAl, self.dimAf)
        self.targetQ = FollowerQNet(parameters, self.dims, self.dimAl, self.dimAf)
        self.onlineQ.to(self.device)
        self.targetQ.to(self.device)


    def get_action_from_Qonline(self):
        '''
        Get the optimal action from online Q-funciton.
        '''
        s = s.float() if torch.is_tensor(s) else torch.from_numpy(s).float()
        with torch.no_grad():
            Qnet = self.onlineQ(s)
        af = Qnet.argmax().item() - 1   # get action from index
        return af


    def get_action_from_policy(self, pif):
        '''
        Get an action from a mixed strategy. 
        '''
        af = self.rng.choice(np.arange(-1, self.dimAf-1), p=pif)    # action set = [-1, ..., dimAf-1]
        return af
    
    
    def get_action_from_epsgreedy(self, af, eps=None):
        '''
        Get the action from eps greedy policy. Return the same af if eps=0.
        '''
        if eps is None:
            eps = self.eps
        if eps == 0:
            return af

        # generate eps greedy policy
        p = np.ones(self.dimAf) * self.eps / (self.dimAf-1)
        p[af+1] = 1 - eps

        af_greedy = self.rng.choice(np.arange(-1, self.dimAf-1), p=p)   # action set = [-1, ..., dimAf-1]
        return af_greedy


    def update_target_Q_parameter(self, tau=None):
        """
        This function replaces the target Q network parameters
        """
        if tau is None:
            tau = self.tau
        d_online = self.onlineQ.state_dict()
        d_target = self.targetQ.state_dict()
        for key in d_online.keys():
            d_target[key] = (1-tau) * d_target[key] + tau * d_online[key]
        
        self.targetQ.load_state_dict(d_target)
        return
