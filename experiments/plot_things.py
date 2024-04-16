"""Plot scripts, need to install matplotlib package, see README."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sg_taskplan.utils import process_data, process_perturb_data


def plot_env_task():
    from sg_taskplan.env import Environment
    from sg_taskplan import config_data_dir

    task_id = 3
    fname = os.path.join(config_data_dir, f'task{task_id}/parameters.json')
    param = json.load(open(fname))
    env = Environment(param)
    env.task_viewer()


def plot_sg():
    '''Plot training results of Stackelberg DDQN.'''
    from sg_taskplan import train_data_dir, plot_data_dir
    dname = os.path.join(train_data_dir, 'sg')
    tasks = [1,2,3,4,5,6,7,8]
    tasks = [1]
    exps = [1,2,3,4,5,6,7,8,9,10] 
    sg_dict = process_data(dname, tasks, exps)

    for task_id in tasks:
        aa = sg_dict[f'task{task_id}']
        
        # plot accumulated reward, leader and follower
        ll_mean, ll_std = aa['train_rl_cum_mean'], aa['train_rl_cum_std']
        lf_mean, lf_std = aa['train_rf_cum_mean'], aa['train_rf_cum_std']
        idx = np.arange(0, ll_mean.shape[0], 10)
        fig, ax = plt.subplots()
        ax.plot(idx, ll_mean[idx], label='leader')
        ax.plot(idx, lf_mean[idx], label='follower')
        plt.fill_between(idx, ll_mean[idx]-ll_std[idx], ll_mean[idx]+ll_std[idx], color=(0.792, 0.909, 0.949), alpha=0.7, edgecolor=None)
        plt.fill_between(idx, lf_mean[idx]-lf_std[idx], lf_mean[idx]+lf_std[idx], color=(1.000, 0.686, 0.055), alpha=0.7, edgecolor=None)
        ax.legend()
        ax.set_xlabel('episode')
        ax.set_ylabel('TD error')
        ax.set_title('Task '+str(task_id))
        fig.savefig(os.path.join(plot_data_dir, 'sg_td_err.png'), dpi=200)
        plt.close(fig)

        # plot eval loss
        eval_rl_mean, eval_rl_std = aa['eval_rl_cum_mean'], aa['eval_rl_cum_std']
        eval_rf_mean, eval_rf_std = aa['eval_rf_cum_mean'], aa['eval_rf_cum_std']
        idx = np.arange(0, eval_rf_mean.shape[0], 10)
        fig, ax = plt.subplots()
        ax.plot(idx, eval_rl_mean[idx], label='leader')
        ax.plot(idx, eval_rf_mean[idx], label='follower')
        plt.fill_between(idx, eval_rl_mean[idx]-eval_rl_std[idx], eval_rl_mean[idx]+eval_rl_std[idx], color=(0.792, 0.909, 0.949), alpha=0.7, edgecolor=None)
        plt.fill_between(idx, eval_rf_mean[idx]-eval_rf_std[idx], eval_rf_mean[idx]+eval_rf_std[idx], color=(1.000, 0.686, 0.055), alpha=0.7, edgecolor=None)
        ax.legend()
        ax.set_xlabel('episode')
        ax.set_ylabel('accumulated episodic reward')
        ax.set_title('Task '+str(task_id))
        fig.savefig(os.path.join(plot_data_dir, 'sg_acc_reward.png'), dpi=200)
        plt.close(fig)

        # plot eval completion step
        eval_step_mean, eval_step_std = aa['train_step_mean'], aa['train_step_std']
        idx = np.arange(0, eval_step_mean.shape[0], 10)
        fig, ax = plt.subplots()
        ax.plot(idx, eval_step_mean[idx], label='sg')
        plt.fill_between(idx, eval_step_mean[idx]-eval_step_std[idx], eval_step_mean[idx]+eval_step_std[idx], color=(0.792, 0.909, 0.949), alpha=0.7, edgecolor=None)
        ax.legend()
        ax.set_xlabel('episode')
        ax.set_ylabel('Completed steps')
        ax.set_title('Task '+str(task_id))
        fig.savefig(os.path.join(plot_data_dir, 'sg_step.png'), dpi=200)
        plt.close(fig)
    return  
    

def plot_sg_perturb(self):
    '''
    Plot perturb results using Stackelberg learning results.
    Choose one exp for one task to plot.
    '''
    from sg_taskplan import train_data_dir, plot_data_dir
    dname = os.path.join(train_data_dir, 'perturb')
    tasks = [1,2,3,4]
    exps = [1,2,3,4,5,6,7,8,9,10]
    sg_perturb = process_perturb_data(dname, tasks, exps)

    task_id, exp_id = 1, 1  # plot first experiment result of task 1
    rl_p = sg_perturb[f'task{task_id}']['rl_traj_perturb'][exp_id-1]    # exp index is exp_id - 1
    rf_p = sg_perturb[f'task{task_id}']['rf_traj_perturb'][exp_id-1]
    rl_n = sg_perturb[f'task{task_id}']['rl_traj_normal'][exp_id-1]
    rf_n = sg_perturb[f'task{task_id}']['rf_traj_normal'][exp_id-1]
        
    clist = ['#1f77b4', '#2ca02c', '#ff7f03']   # blue, green, orange
    fig, ax = plt.subplots(2,1,figsize=(7, 4.8))
    ax[0].plot(np.arange(rl_n.shape[0])+1, np.cumsum(rl_n), label='normal', marker='o', color=clist[0])
    ax[0].plot(np.arange(rl_p.shape[0])+1, np.cumsum(rl_p), linestyle='--', marker='o', label='perturbed', color=clist[2])
    ax[0].legend(fontsize='x-large')
    ax[0].set_ylabel('Leader', fontsize='xx-large')
    ax[0].set_title('Task'+str(task_id), fontsize='xx-large')

    ax[1].plot(np.arange(rf_n.shape[0])+1, np.cumsum(rf_n), label='normal', marker='s', color=clist[1])
    ax[1].plot(np.arange(rf_p.shape[0])+1, np.cumsum(rf_p), linestyle='--', marker='s', label='perturbed', color=clist[2])
    ax[1].legend(fontsize='x-large')
    ax[1].set_ylabel('Follower', fontsize='xx-large')
    ax[1].set_xlabel('steps', fontsize='xx-large')
    fig.savefig(os.path.join(plot_data_dir, f'disturb_t{task_id}_e{exp_id}.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    plot_env_task()
    plot_sg()
    #plot_sg_perturb()