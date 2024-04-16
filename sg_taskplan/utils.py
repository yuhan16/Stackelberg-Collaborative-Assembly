import os
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, parameters) -> None:
        #import parameters
        self.rng = np.random.default_rng(parameters['seed'])
        self.total_buffer_size = parameters['buffer_size']
        self.data = []
    
    
    def __len__(self):
        return len(self.data)
    
    
    def add(self, samp):
        '''
        Add a sample to the buffer.
        '''
        if len(self.data) == 0:
            self.data = samp[None, :]
        elif self.data.shape[0] < self.total_buffer_size:
            self.data = np.vstack((self.data, samp[None, :]))
        else:
            self.data = np.vstack((self.data[1:, :], samp[None, :]))
        return
    

    def sample_buffer(self, N):
        '''
        Randomly sample N data from buffer.
        '''
        if len(self.data) < N:
            raise Exception('Not enough buffer data for sampling.')
        else:
            idx = self.rng.choice(self.data.shape[0], N, replace=False)
            return self.data[idx, :]
    

    def empty_buffer(self):
        '''
        Empty the current buffer.
        '''
        self.data = []



def save_results(dname, res):
    '''Save the provided results.'''
    if not os.path.exists(dname):
        os.makedirs(dname)
    
    for key in res:
        if '.pt' in key:
            torch.save(res[key], f'{dname}/{key}')
        else:
            np.save(f'{dname}/{key}', res[key])


def process_data(dname, tasks=None, exps=None):
    """
    Process the learned data.
    tasks and exps specify which tasks and exps to process.
    """
    #dname = 'data/train/sg'    # dname example
    if tasks is None:
        tasks = [x+1 for x in range(8)]
    if exps is None:
        exps = [x+1 for x in range(10)]
    
    task_dict = {}  # task dictionary
    for task_id in tasks:
        train_step_list, train_rl_cum_list, train_rf_cum_list = [], [], []
        eval_step_list, eval_rl_cum_list, eval_rf_cum_list = [], [], []
        ave_rl_list, ave_rf_list = [], []
        for i in exps:
            train_step = np.load(dname + '/task' + str(task_id) + '/exp' + str(i) + '/train_step.npy')
            train_ll_cum = np.load(dname + '/task' + str(task_id) + '/exp' + str(i) + '/train_ll.npy').sum(axis=1)
            train_lf_cum = np.load(dname + '/task' + str(task_id) + '/exp' + str(i) + '/train_lf.npy').sum(axis=1)
            train_rl_cum = np.load(dname + '/task' + str(task_id) + '/exp' + str(i) + '/train_rl.npy').sum(axis=1)
            train_rf_cum = np.load(dname + '/task' + str(task_id) + '/exp' + str(i) + '/train_rf.npy').sum(axis=1)
            
            eval_step = np.load(dname + '/task' + str(task_id) + '/exp' + str(i) + '/eval_step.npy')
            eval_rl_cum = np.load(dname + '/task' + str(task_id) + '/exp' + str(i) + '/eval_rl.npy').sum(axis=1)
            eval_rf_cum = np.load(dname + '/task' + str(task_id) + '/exp' + str(i) + '/eval_rf.npy').sum(axis=1)

            train_step_list.append(train_step)
            train_rl_cum_list.append(train_rl_cum)
            train_rf_cum_list.append(train_rf_cum)
            eval_step_list.append(eval_step)
            eval_rl_cum_list.append(eval_rl_cum)
            eval_rf_cum_list.append(eval_rf_cum)
            ave_rl_list.append(train_rl_cum / train_step)
            ave_rf_list.append(train_rf_cum / train_step)
        
        train_step_list = np.vstack(train_step_list)
        train_rl_cum_list = np.vstack(train_rl_cum_list)
        train_rf_cum_list = np.vstack(train_rf_cum_list)
        
        eval_step_list = np.vstack(eval_step_list)
        eval_rl_cum_list = np.vstack(eval_rl_cum_list)
        eval_rf_cum_list = np.vstack(eval_rf_cum_list)
        
        ave_rl_list = np.vstack(ave_rl_list)
        ave_rf_list = np.vstack(ave_rf_list)
        
        train_step_mean, train_step_std = np.mean(train_step_list, axis=0), np.std(train_step_list, axis=0)
        train_rl_cum_mean, train_rl_cum_std = np.mean(train_rl_cum_list, axis=0), np.std(train_rl_cum_list, axis=0)
        train_rf_cum_mean, train_rf_cum_std = np.mean(train_rf_cum_list, axis=0), np.std(train_rf_cum_list, axis=0)
        eval_step_mean, eval_step_std = np.mean(eval_step_list, axis=0), np.std(eval_step_list, axis=0)
        eval_rl_cum_mean, eval_rl_cum_std = np.mean(eval_rl_cum_list, axis=0), np.std(eval_rl_cum_list, axis=0)
        eval_rf_cum_mean, eval_rf_cum_std = np.mean(eval_rf_cum_list, axis=0), np.std(eval_rf_cum_list, axis=0)
        ave_rl_mean, ave_rl_std = np.mean(ave_rl_list, axis=0), np.std(ave_rl_list, axis=0)
        ave_rf_mean, ave_rf_std = np.mean(ave_rf_list, axis=0), np.std(ave_rf_list, axis=0)

        a = {}
        a['train_step_mean'] = train_step_mean
        a['train_step_std'] = train_step_std
        a['train_rl_cum_mean'] = train_rl_cum_mean
        a['train_rl_cum_std'] = train_rl_cum_std
        a['train_rf_cum_mean'] = train_rf_cum_mean
        a['train_rf_cum_std'] = train_rf_cum_std
        
        a['eval_step_mean'] = eval_step_mean
        a['eval_step_std'] = eval_step_std
        a['eval_rl_cum_mean'] = eval_rl_cum_mean
        a['eval_rl_cum_std'] = eval_rl_cum_std
        a['eval_rf_cum_mean'] = eval_rf_cum_mean
        a['eval_rf_cum_std'] = eval_rf_cum_std
        
        a['ave_rl_mean'] = ave_rl_mean
        a['ave_rl_std'] = ave_rl_std
        a['ave_rf_mean'] = ave_rf_mean
        a['ave_rf_std'] = ave_rf_std
        task_dict['task'+str(task_id)] = a
    return task_dict


def process_perturb_data(dname, tasks=None, exps=None):
    """
    Process the perturbed data.
    tasks and exps specify which tasks and exps to process.
    """
    #dname = 'data/train/perturb'    # dname example
    if tasks is None:
        tasks = [x+1 for x in range(8)]
    if exps is None:
        exps = [x+1 for x in range(10)]
    
    task_dict = {}  # task dictionary
    for task_id in tasks:
        step_p, step_n = [], []
        rl_p, rf_p, rl_n, rf_n = [], [], [], []
        al_p, af_p, al_n, af_n = [], [], [], []
        for i in exps:
            step_p.append( np.load(dname+f'/task{task_id}/exp{i}/step_perturb.npy') )
            step_n.append( np.load(dname+f'/task{task_id}/exp{i}/step_normal.npy') )

            rl_p.append( np.load(dname+f'/task{task_id}/exp{i}/rl_traj_perturb.npy') )
            rf_p.append( np.load(dname+f'/task{task_id}/exp{i}/rf_traj_perturb.npy') )
            rl_n.append( np.load(dname+f'/task{task_id}/exp{i}/rl_traj_normal.npy'))
            rf_n.append( np.load(dname+f'/task{task_id}/exp{i}/rf_traj_normal.npy'))
            
            al_p.append( np.load(dname+f'/task{task_id}/exp{i}/al_traj_perturb.npy') )
            af_p.append( np.load(dname+f'/task{task_id}/exp{i}/af_traj_perturb.npy') )
            al_n.append( np.load(dname+f'/task{task_id}/exp{i}/al_traj_normal.npy'))
            af_n.append( np.load(dname+f'/task{task_id}/exp{i}/af_traj_normal.npy'))
            
        a = {
            'l_disturb': np.load(dname+f'/task{task_id}/exp1/rl_traj_perturb.npy'), # all exps have same l_disturb
            'f_disturb': np.load(dname+f'/task{task_id}/exp1/rf_traj_perturb.npy'),
            'step_perturb': step_p,
            'step_normal': step_n,
            'rl_traj_perturb': rl_p,
            'rf_traj_perturb': rf_p,
            'rl_traj_normal': rl_n,
            'rf_traj_normal': rf_n,
            'al_traj_perturb': al_p,
            'af_traj_perturb': af_p,
            'al_traj_normal': al_n,
            'af_traj_normal': af_n,
        }
        task_dict[f'task{task_id}'] = a
    return task_dict