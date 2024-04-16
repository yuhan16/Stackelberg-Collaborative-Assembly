"""This module implements Stackelberg double Q-learning."""
import time
import numpy as np
import torch
from ..env import Environment
from ..utils import ReplayBuffer, save_results
from .agent import Leader, Follower


class StackelbergLearn:
    '''
    This class defines functions to perform Stackelberg DDQN.
    '''
    def __init__(self, parameters, env) -> None:
        env_prop = env.get_task_info()
        self.dims = env_prop['dims']
        self.dimAl, self.dimAf = env_prop['dimAl'], env_prop['dimAf']
        self.dimal, self.dimaf = env_prop['dimal'], env_prop['dimaf']
        #self.task_prop = env_prop['task_prop']
        
        self.device = parameters['device']
        self.n_episode = parameters['episode_size']
        self.n_step_ep = parameters['step_per_episode']
        self.n_buffer = parameters['buffer_size']
        self.n_batch = parameters['batch_size']
        self.n_update_qtarget = parameters['update_target_step']
    
    
    def generate_replay_buffer(self, env, buffer, N):
        '''
        Generate buffer of size N using random actions, not exceeding the total buffer_size
        '''
        if N > self.n_buffer:
            N = self.n_buffer
        
        action_set = np.arange(-1, env.task_board.shape[1])
        while len(buffer) < N:
            env.reset_env()
            s0, _ = env.get_current_state()
            iter = 0
            while not np.all(s0 == 0) and iter < 100:   # aviod too many failures in completing the task
                al = env.rng.choice(action_set)
                af = env.rng.choice(action_set)
                rl, rf = env.reward(s0, al, af)        
                env.step(al, af)
                s_new, _ = env.get_current_state()
                samp = np.concatenate((s0, np.array([al, af, rl, rf]), s_new))
                buffer.add(samp)
        
                s0 = s_new
                iter += 1
        env.reset_env()     # reset env after generating the buffer
        return


    def compute_se(self, ll, ff, s):
        '''
        Compute Stackelberg equilibrium using leader and follower's online Q-networks.
        '''
        s = s[None,:].float().to(self.device) if torch.is_tensor(s) else torch.from_numpy(s[None, :]).float().to(self.device)
        with torch.no_grad():
            Ql = ll.onlineQ(s).cpu().reshape([self.dimAl, self.dimAf])
            Qf = ff.onlineQ(s).cpu().reshape([self.dimAl, self.dimAf])
        
        val = -100
        al_se, af_se = 0, 0     # index, not specific actions
        for i in range(self.dimAl):
            af = Qf[i, :].argmax().item()   # compute best response index
            if val < Ql[i, af]:
                val = Ql[i, af]
                al_se, af_se = i, af
        return al_se-1, af_se-1     # restore action from index


    def train(self, parameters, env, exp_id=None):
        '''
        Stackelberg DDQN algorithm. traj = [s, al, af, rl, rf, s_new]
        '''
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s')
        logger = logging.getLogger('Stackelberg train')
        logger.setLevel(logging.INFO)

        torch.manual_seed(parameters['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(parameters['seed'])

        leader = Leader(parameters, env)
        follower = Follower(parameters, env)
        buffer = ReplayBuffer(parameters)
        self.generate_replay_buffer(env, buffer, N=self.n_buffer)
        cnt = 0     # counter for updating target Q network
        #bd_list = env.generate_random_state()

        loss_fn = torch.nn.MSELoss()
        #opt_l = torch.optim.SGD(leader.onlineQ.parameters(), lr=leader.lr, momentum=leader.mom)
        #opt_f = torch.optim.SGD(follower.onlineQ.parameters(), lr=follower.lr, momentum=follower.mom)
        opt_l = torch.optim.Adam(leader.onlineQ.parameters(), lr=leader.lr)
        opt_f = torch.optim.Adam(follower.onlineQ.parameters(), lr=follower.lr)

        # allocate learning statistics
        train_ll = np.zeros((self.n_episode, self.n_step_ep))     # leader's training loss
        train_lf = np.zeros((self.n_episode, self.n_step_ep))     # follower's training loss
        train_step = np.zeros(self.n_episode)
        train_rl = np.zeros((self.n_episode, self.n_step_ep))     # leader's training reward
        train_rf = np.zeros((self.n_episode, self.n_step_ep))     # follower's training loss
        eval_step = np.zeros(self.n_episode)
        eval_rl = np.zeros((self.n_episode, self.n_step_ep))
        eval_rf = np.zeros((self.n_episode, self.n_step_ep))
        
        logger.info(f'Training task {env.task_id}...')
        for ep in range(self.n_episode):
            st = time.time()
            logger.info(f'-------------Episode {ep+1}-------------')
            # use initial state for each episode
            env.reset_env()
            s0, _ = env.get_current_state()

            # use random state for each episode
            #env.set_env(env.rng.choice(bd_list))
            #s0, _ = env.get_current_state()

            for t in range(self.n_step_ep):
                al_se, af_se = self.compute_se(leader, follower, s0)
                al = leader.get_action_from_epsgreedy(al_se)
                af = follower.get_action_from_epsgreedy(af_se)

                # simulate next step and get reward
                rl, rf = env.reward(s0, al, af)
                env.step(al, af)
                s_new, _ = env.get_current_state()
                train_rl[ep, t] = rl
                train_rf[ep, t] = rf

                # store simulated result to the buffer
                samp = np.concatenate((s0, np.array([al, af, rl, rf]), s_new))
                buffer.add(samp)

                # sample mini-batch transitions, convert to torch tensor
                D = buffer.sample_buffer(N=self.n_batch)
                D = torch.from_numpy(D).float().to(self.device)
                s_batch = D[:, :self.dims]
                al_batch, af_batch = D[:, self.dims].type(torch.int64).unsqueeze(1), D[:, self.dims+self.dimal].type(torch.int64).unsqueeze(1)
                rl_batch, rf_batch = D[:, self.dims+self.dimal+self.dimaf].unsqueeze(1), D[:, self.dims+self.dimal+self.dimaf+1].unsqueeze(1)
                snew_batch = D[:, -self.dims:]
                idx_a = (al_batch+1) * self.dimAf + (af_batch+1)
                
                # use online Q to compute SE at snew, batch operation
                with torch.no_grad():
                    Ql = leader.onlineQ(snew_batch).reshape([self.n_batch, self.dimAl, self.dimAf])
                    Qf = follower.onlineQ(snew_batch).reshape([self.n_batch, self.dimAl, self.dimAf])
                idx_f = Qf.argmax(dim=2).unsqueeze(-1)
                qll = Ql.gather(2, idx_f)
                idx_l = qll.argmax(dim=1)
                idx_se = idx_l * self.dimAf + idx_f.gather(1, idx_l.unsqueeze(-1)).squeeze(-1)
                
                # compute bellman operator using target Q at snew, batch operation
                with torch.no_grad():
                    ql = leader.targetQ(snew_batch).gather(1, idx_se)
                    qf = follower.targetQ(snew_batch).gather(1, idx_se)
                done = 1 * torch.all(snew_batch == 0, dim=1).unsqueeze(1)
                yl = rl_batch + leader.gam * (1-done) * ql
                yf = rf_batch + follower.gam * (1-done) * qf
                loss_l = loss_fn(leader.onlineQ(s_batch).gather(1, idx_a), yl)
                loss_f = loss_fn(follower.onlineQ(s_batch).gather(1, idx_a), yf)                    
                
                # update theta using y and D for one step GD
                opt_l.zero_grad()
                loss_l.backward()
                opt_l.step()
                train_ll[ep, t] = loss_l.item()

                opt_f.zero_grad()
                loss_f.backward()
                opt_f.step()
                train_lf[ep, t] = loss_f.item()

                logger.debug(f'loss_l: {loss_l.item():.3f}, loss_f: {loss_f.item():.3f}')
                
                # soft update target network
                cnt += 1
                if cnt % self.n_update_qtarget == 0:
                    leader.update_target_Q_parameter()
                    follower.update_target_Q_parameter()
                
                # check if s_new is terminal state
                if np.all(s_new == 0):
                    logger.debug(f'current episode ends at step {t}.')
                    break
                else:
                    s0 = s_new
            
            logger.debug(f'{100*(time.time()-st):.5f} ms.')
            
            # evaluate current online Q network
            ss, rll, rff = self.eval_policy(parameters, leader, follower)
            eval_step[ep] = ss
            eval_rl[ep] = rll
            eval_rf[ep] = rff

            # save completion step for each episode
            train_step[ep] = t + 1

        logger.info('Stackelberg training completed.')
        save_flag = True
        if save_flag:
            import os
            from .. import train_data_dir
            dname = os.path.join(train_data_dir, 'sg', f'task{parameters["task_id"]}')
            if exp_id is not None:
                dname = os.path.join(dname, f'exp{exp_id}')     # append exp_id to dir
            
            # specify filename and format
            res_save = {
                'train_step.npy': train_step,
                'train_ll.npy': train_ll,
                'train_lf.npy': train_lf,
                'train_rl.npy': train_rl,
                'train_rf.npy': train_rf,
                'eval_step.npy': eval_step,
                'eval_rl.npy': eval_rl,
                'eval_rf.npy': eval_rf,
                'l_onlineQ.pt': leader.onlineQ.state_dict(),
                'l_targetQ.pt': leader.targetQ.state_dict(),
                'f_onlineQ.pt': follower.onlineQ.state_dict(),
                'f_targetQ.pt': follower.targetQ.state_dict(),
            }
            save_results(dname, res_save)
        return


    def eval_policy(self, p, leader, follower):
        '''
        Evaluate the current Q-network. Record action, reward, and completion steps.
        '''
        env_tmp = Environment(p)
        env_tmp.reset_env()
        s0, _ = env_tmp.get_current_state()
        total_step, rl_array, rf_array = 0, np.zeros(self.n_step_ep), np.zeros(self.n_step_ep)
        for t in range(self.n_step_ep):
            al, af = self.compute_se(leader, follower, s0)
            #al = leader.get_action_from_epsgreedy(al)
            #af = follower.get_action_from_epsgreedy(af)
            rl, rf = env_tmp.reward(s0, al, af)
            env_tmp.step(al, af)
            s_new, _ = env_tmp.get_current_state()
            rl_array[t] = rl
            rf_array[t] = rf

            if np.all(s_new == 0):
                #print(f'current eval ends at step {t+1}.')
                break
            
            s0 = s_new
        total_step = t + 1
        return total_step, rl_array, rf_array
        

'''
if __name__ == '__main__':
    import json
    task_id = 1
    p = json.load(open('data/task'+str(task_id)+'/parameters.json'))
    env = Environment(p)
    sgtrainner = StackelbergLearn(p, env)
    sgtrainner.train(p, env)
'''