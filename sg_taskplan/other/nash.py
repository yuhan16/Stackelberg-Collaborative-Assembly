"""This module implements Nash Q-learning."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import torch
from ..env import Environment
from ..utils import ReplayBuffer, save_results
from .agent import LeaderNash, FollowerNash


class NashLearn:
    '''
    This class defines functions to perform Nash double Q-learning.
    '''
    def __init__(self, parameters, env) -> None:
        env_prop = env.get_task_info()
        self.dims = env_prop['dims']
        self.dimAl, self.dimAf = env_prop['dimAl'], env_prop['dimAf']
        self.dimal, self.dimaf = env_prop['dimal'], env_prop['dimaf']
        self.task_prop = env_prop['task_prop']
        
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


    def compute_ne_qonline(self, ll, ff, s):
        '''
        Use online Q-network to compute NE in state s.
        '''
        s = s[None,:].float().to(self.device) if torch.is_tensor(s) else torch.from_numpy(s[None, :]).float().to(self.device)
        with torch.no_grad():
            Ql = ll.onlineQ(s).cpu().reshape([self.dimAl, self.dimAf]).numpy().astype(float)
            Qf = ff.onlineQ(s).cpu().reshape([self.dimAl, self.dimAf]).numpy().astype(float)
        
        al_ne, af_ne = self.compute_ne_pure(Ql, Qf)
        if al_ne is None or af_ne is None:
            pil_ne, pif_ne = self.compute_ne_mix(Ql, Qf)
            al_ne = ll.get_action_from_policy(pil_ne)
            af_ne = ff.get_action_from_policy(pif_ne)
        return al_ne, af_ne

    
    def compute_ne_qtarget(self, ll, ff, s):
        '''
        Use target Q-network to compute NE in state s.
        '''
        s = s[None,:].float().to(self.device) if torch.is_tensor(s) else torch.from_numpy(s[None, :]).float().to(self.device)
        with torch.no_grad():
            Ql = ll.targetQ(s).cpu().reshape([self.dimAl, self.dimAf]).numpy().astype(float)
            Qf = ff.targetQ(s).cpu().reshape([self.dimAl, self.dimAf]).numpy().astype(float)
        
        al_ne, af_ne = self.compute_ne_pure(Ql, Qf)
        if al_ne is None or af_ne is None:
            pil_ne, pif_ne = self.compute_ne_mix(Ql, Qf)
            al_ne = ll.get_action_from_policy(pil_ne)
            af_ne = ff.get_action_from_policy(pif_ne)
        return al_ne, af_ne


    def compute_ne_pure(self, Ql, Qf):
        '''
        Compute NE in pure strategy of a bimatrix game generated from the leader and follower's Q network. Both are maximizer.
        Inputs:
            - Ql: leader's (negative) Q matrix at state s.
            - Qf: follower's (negative) Q matrix at state s.
        Outputs:
            - al_ne, af_ne: leader and follower's pure strategy. None for no pure strategy.
        '''
        al_ne, af_ne = None, None
        
        br_l = Ql.argmax(axis=0)
        br_f = Qf.argmax(axis=1)
        idx_l = np.ravel_multi_index(np.vstack((np.arange(Ql.shape[0]), br_f)), Ql.shape)
        idx_f = np.ravel_multi_index(np.vstack((br_l, np.arange(Qf.shape[1]))), Qf.shape)
        tmp = np.intersect1d(idx_l, idx_f, assume_unique=True)
        if tmp.shape[0] > 0:
            ii = np.random.choice(tmp)  # or use Ql value to select
            idx = np.unravel_index(ii, Ql.shape)
            al_ne, af_ne = idx[0]-1, idx[1]-1
        
        return al_ne, af_ne
    
    
    def compute_ne_mix(self, Ql, Qf):
        '''
        Compute NE in mixed strategy of a bimatrix game generated from the leader and follower's Q network. Both are minimizer.
        Inputs:
            - Ql: leader's (negative) Q matrix at state s.
            - Qf: follower's (negative) Q matrix at state s.
        Outputs:
            - pil, pif: leader and follower's mixed strategy.
        '''
        Ql, Qf = -Ql, -Qf  # leader and follower are maximizers. scipy use minimize
        m, n = Ql.shape[0], Ql.shape[1]

        # generate random initial conditions
        def myinit():
            pil0 = np.random.rand(m)
            pil0 /= pil0.sum()
            pif0 = np.random.rand(n)
            pif0 /= pif0.sum()
            p0, q0 = 0., 0.
            X = np.concatenate((pil0, pif0, np.array([p0, q0])))
            return X

        # formulate bilevel optimization problem, X = [pil, pif, p, q]
        def myobj(X):
            pil, pif, p, q = X[: m], X[m: m + n], X[m + n], X[-1]
            J = pil @ Ql @ pif + pil @ Qf @ pif + p + q
            return J

        def myjac(X):
            jac = np.zeros(m + n + 2)
            pil, pif, p, q = X[: m], X[m: m + n], X[m + n], X[-1]

            jac[: m] = Ql @ pif + Qf @ pif
            jac[m: m + n] = pil @ Ql + pil @ Qf
            jac[m + n] = 1
            jac[-1] = 1
            return jac

        # formulate constraints
        def formulate_constr():
            A1 = np.zeros((m, m + n + 2))
            A1[:, m:m + n] = Ql
            A1[:, m + n] = np.ones(m)
            lb1, ub1 = np.zeros(m), np.inf * np.ones(m)

            A2 = np.zeros((n, m + n + 2))
            A2[:, :m] = Qf.T
            A2[:, -1] = np.ones(n)
            lb2, ub2 = np.zeros(n), np.inf * np.ones(n)

            A3 = np.zeros((m + n, m + n + 2))
            A3[:m, :m] = np.eye(m)
            A3[m:, m:m + n] = np.eye(n)
            lb3 = np.ones(m + n) * 1e-6
            ub3 = np.inf * np.ones(m + n)
            # lb3, ub3 = np.zeros(m + n), np.inf * np.ones(m + n)

            A4 = np.zeros((1, m + n + 2))
            A4[0, :m] = np.ones(m)
            lb4, ub4 = np.ones(1), np.ones(1)

            A5 = np.zeros((1, m + n + 2))
            A5[0, m: m + n] = np.ones(n)
            lb5, ub5 = np.ones(1), np.ones(1)

            A = np.vstack((A1, A2, A3))
            lb = np.concatenate((lb1, lb2, lb3))
            ub = np.concatenate((ub1, ub2, ub3))

            Aeq = np.vstack((A4, A5))
            lb_eq = np.concatenate((lb4, lb5))
            ub_eq = np.concatenate((lb4, lb5))
            return A, lb, ub, Aeq, lb_eq, ub_eq

        A, lb, ub, Aeq, lb_eq, ub_eq = formulate_constr()
        constr = [LinearConstraint(A, lb, ub), LinearConstraint(Aeq, lb_eq, ub_eq)]
        
        """
        # gradient check
        from scipy.optimize import check_grad, approx_fprime
        diff = np.zeros(10)
        for i in range(diff.shape[0]):
            diff[i] = check_grad(myobj, myjac, np.random.rand(m+n+2))
        print(diff)
        """
        X0 = myinit()
        res = minimize(myobj, X0, jac=myjac, constraints=constr)
        # print('status {}: {}.'.format(res.status, res.message))
        iter = 0
        while res.status != 0 and iter < 10:    # only iter for finite times
            X0 = myinit()
            res = minimize(myobj, X0, jac=myjac, constraints=constr)
            iter += 1
            # print("reinitialize...")
        Xopt = res.x
        pil, pif = Xopt[:m], Xopt[m: m + n]
        pil[pil < 0] = 0
        pif[pif < 0] = 0
        pil /= pil.sum()
        pif /= pif.sum()
        
        return pil, pif


    def train(self, parameters, env, exp_id=None):
        '''
        Implements Nash DQN algorithm. traj = [s, al, af, rl, rf, s_new]
        '''
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s')
        logger = logging.getLogger('Nash train')
        logger.setLevel(logging.INFO)

        torch.manual_seed(parameters['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(parameters['seed'])

        leader = LeaderNash(parameters, env)
        follower = FollowerNash(parameters, env)
        buffer = ReplayBuffer(parameters)
        self.generate_replay_buffer(env, buffer, N=self.n_buffer)
        cnt = 0     # counter for updating target Q network

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
            st1 = time.time()
            logger.info(f'-------------Episode {ep+1}-------------')

            env.reset_env()     # reset to initial state each episode
            s0, _ = env.get_current_state()

            for t in range(self.n_step_ep):
                al, af = self.compute_ne_qonline(leader, follower, s0)

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
                
                loss_l, loss_f = 0, 0
                for i in range(D.shape[0]):
                    s_i, al_i, af_i = D[i, :self.dims], D[i, self.dims].int().item(), D[i, self.dims+self.dimal].int().item()
                    rl_i, rf_i, s_ip1 = D[i, self.dims+self.dimal+self.dimaf], D[i, self.dims+self.dimal+self.dimaf+1], D[i, -self.dims:]
                    idx_a = (al_i+1) * self.dimAf + (af_i+1)    # index of composite actions
                    
                    # use target Q network to compute bellman equation
                    al_ne_i, af_ne_i = self.compute_ne_qtarget(leader, follower, s_ip1)
                    idx_ne = (al_ne_i+1) * self.dimAf + (af_ne_i+1)
                    with torch.no_grad():
                        ql = leader.targetQ(s_ip1)[idx_ne]
                        qf = follower.targetQ(s_ip1)[idx_ne]

                    # compute loss
                    if torch.all(s_ip1 == 0):   # terminal state
                        loss_l += loss_fn(leader.onlineQ(s_i[None,:])[:, [idx_a]], torch.tensor([[rl_i]], device=self.device)) / self.n_batch
                        loss_f += loss_fn(follower.onlineQ(s_i[None,:])[:, [idx_a]], torch.tensor([[rf_i]], device=self.device)) / self.n_batch
                    else:
                        loss_l += loss_fn(leader.onlineQ(s_i[None,:])[:, [idx_a]], torch.tensor([[rl_i + leader.gam * ql]], device=self.device)) / self.n_batch
                        loss_f += loss_fn(follower.onlineQ(s_i[None,:])[:, [idx_a]], torch.tensor([[rf_i + follower.gam * qf]], device=self.device)) / self.n_batch
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
                
                # hard update target network
                cnt += 1
                if cnt % self.n_update_qtarget == 0:
                    leader.update_target_Q_parameter(tau=1)
                    follower.update_target_Q_parameter(tau=1)
                
                # check if s_new is terminal state
                if np.all(s_new == 0):
                    logger.debug(f'Episode {ep+1} ends at step {t}.')
                    break
                else:
                    s0 = s_new
            
            logger.info(f'time for Episode {ep+1}: {100*(time.time()-st1):.3f} ms.')

            # evaluate current online Q network
            ss, rll, rff = self.eval_policy(parameters, leader, follower)
            eval_step[ep] = ss
            eval_rl[ep] = rll
            eval_rf[ep] = rff

            # save completion step for each episode
            train_step[ep] = t + 1
            
            '''
            # observe learning progress
            if ep % 100 == 0:
                from .. import tmp_data_dir
                import os
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(np.mean(train_ll[ep, :train_step[ep]])[:ep], label='ll')
                ax.plot(np.mean(train_lf[ep, :train_step[ep]])[:ep], label='ff')
                ax.legend()
                ax.set_xlabel('episode')
                ax.set_ylabel('TD error')
                fig.savefig(os.path.join(tmp_data_dir, 'tmp_nash.png'), dpi=200)
                plt.close(fig)
            '''
        logger.info('Nash training completed.')
        save_flag = True
        if save_flag:
            import os
            from .. import train_data_dir
            dname = os.path.join(train_data_dir, 'nash', f'task{parameters["task_id"]}')
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
            al, af = self.compute_ne_qonline(leader, follower, s0)
            rl, rf = env_tmp.reward(s0, al, af)
            env_tmp.step(al, af)
            s_new, _ = env_tmp.get_current_state()
            rl_array[t] = rl
            rf_array[t] = rf

            if np.all(s_new == 0):
                break
            
            s0 = s_new
        total_step = t + 1
        return total_step, rl_array, rf_array
