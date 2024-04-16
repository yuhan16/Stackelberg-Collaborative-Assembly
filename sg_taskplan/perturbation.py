import os
import numpy as np
import torch
from .sg.model import LeaderQNet, FollowerQNet
from .utils import save_results


class DisturbEval:
    '''
    Implement disturbance to the learned model.
    '''
    def __init__(self, env) -> None:
        a = env.get_task_info()
        self.dims, self.dimAl, self.dimAf = a['dims'], a['dimAl'], a['dimAf']

    
    def choose_disturb_seq(self, task_id):
        '''
        Define disturbance step (robot fails, equilvalent to do nothing). [] denotes no disturbance
        '''
        if task_id == 1:
            l_disturb, f_disturb = [1,4], [6,8]
        elif task_id == 2:
            l_disturb, f_disturb = [3,7], [5,10]
        elif task_id == 3:
            l_disturb, f_disturb = [1,2,9], [2,7,11]
        elif task_id == 4:
            l_disturb, f_disturb = [9,14,15], [2,6]
        else:
            pass
        return l_disturb, f_disturb


    def compute_se(self, Qlnet, Qfnet, s):
        '''
        Compute Stackelberg equilibrium using leader and follower's online Q-networks.
        '''
        s = s[None,:].float().to('cpu') if torch.is_tensor(s) else torch.from_numpy(s[None, :]).float().to('cpu')
        with torch.no_grad():
            ql = Qlnet(s).reshape([self.dimAl, self.dimAf])
            qf = Qfnet(s).reshape([self.dimAl, self.dimAf])
            
        val = -100
        al_se, af_se = 0, 0     # index, not specific actions
        for i in range(self.dimAl):
            af = qf[i, :].argmax().item()   # compute best response index
            if val < ql[i, af]:
                val = ql[i, af]
                al_se, af_se = i, af
        return al_se-1, af_se-1     # restore action from index


    def eval(self, parameters, env, exp_id):
        '''
        At the disturb step, agent is equilivent to select nothing and get zero reward. The parterner's score does not change.
        '''
        from . import train_data_dir
        dname = os.path.join(train_data_dir, 'sg', f'task{env.task_id}/exp{exp_id}')
        Ql = LeaderQNet(parameters, self.dims, self.dimAl, self.dimAf)
        Qf = FollowerQNet(parameters, self.dims, self.dimAl, self.dimAf)
        Ql.load_state_dict(torch.load(dname+'/l_onlineQ.pt', map_location='cpu'))
        Qf.load_state_dict(torch.load(dname+'/f_onlineQ.pt', map_location='cpu'))
        
        env.reset_env()
        l_disturb, f_disturb = self.choose_disturb_seq(env.task_id)
        al_traj, af_traj = [], []
        rl_traj, rf_traj = [], []
        s0, _ = env.get_current_state()
        step = 0    # completion steps
        cnt = 1
        
        print(f'Eval sg disturbance with data from exp {exp_id}.')
        print(f'l_disturb: {l_disturb}')
        print(f'f_disturb: {f_disturb}')
        while True:
            # generate al and af using Q network.
            al, af = self.compute_se(Ql, Qf, s0)
            rl, rf = env.reward(s0, al, af)

            # perform disturbance
            if cnt in l_disturb:
                al = -1
                rl = -1
            if cnt in f_disturb:
                af = -1
                rf = -1
            
            env.step(al, af)
            s_new, _ = env.get_current_state()

            al_traj.append(al)
            af_traj.append(af)
            rl_traj.append(rl)
            rf_traj.append(rf)
            
            if np.all(s_new == 0):
                break

            s0 = s_new
            step += 1
            cnt += 1
        print(f'total step {step}.')
        print(al_traj)
        print(af_traj)
        print(rl_traj)
        print(rf_traj)

        # no disturbance evaluation
        env.reset_env()
        al_no_traj, af_no_traj = [], []
        rl_no_traj, rf_no_traj = [], []
        step_no = 0
        print(f'\nEval sg without disturbance with data from exp {exp_id}.')
        while True:
            al, af = self.compute_se(Ql, Qf, s0)
            rl, rf = env.reward(s0, al, af)
            env.step(al, af)
            s_new, _ = env.get_current_state()
            al_no_traj.append(al)
            af_no_traj.append(af)
            rl_no_traj.append(rl)
            rf_no_traj.append(rf)
            if np.all(s_new == 0):
                break
            s0 = s_new
            step_no += 1
        print(f'total step {step_no}.')
        print(al_no_traj)
        print(af_no_traj)
        print(rl_no_traj)
        print(rf_no_traj)

        # save perturbation results with txt files, or move to utils.
        save_flag = True
        if save_flag:
            from . import train_data_dir
            dname = os.path.join(train_data_dir, 'perturb', f'task{env.task_id}/exp{exp_id}')   # change to save numpy array ???
            res = {
                'l_disturb.npy': np.array(l_disturb),
                'f_disturb.npy': np.array(f_disturb),
                'step_perturb.npy': step,
                'rl_traj_perturb.npy': rl_traj,
                'rf_traj_perturb.npy': rf_traj,
                'al_traj_perturb.npy': al_traj,
                'af_traj_perturb.npy': af_traj,
                'step_normal.npy': step_no,
                'rl_traj_normal.npy': rl_no_traj,
                'rf_traj_normal.npy': rf_no_traj,
                'al_traj_normal.npy': al_no_traj,
                'af_traj_normal.npy': af_no_traj,
            }
            save_results(dname, res)
