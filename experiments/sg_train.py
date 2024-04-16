import os
import json
import multiprocessing
import time
from sg_taskplan.env import Environment
from sg_taskplan.sg.stackelberg import StackelbergLearn
from sg_taskplan import config_data_dir


def train(task_id=1):
    '''Run a single training for a specific task.'''
    fname = os.path.join(config_data_dir, f'task{task_id}/parameters.json')
    param = json.load(open(fname))
    #param['episode_size'] = 2    # for quick test
    env = Environment(param)

    sgtrainner = StackelbergLearn(param, env)
    sgtrainner.train(param, env, exp_id=None)


def train_multiproc(task_id=1):
    '''Run multiple training (experiments) in parallel for a specific task.'''
    exps = [x+1 for x in range(10)]     # or change exps to run specific exp
    
    # pre-define some parameters for all experiments
    seed_list = [39304, 12276, 12702, 1359, 31415, 21186, 5858, 716, 63008, 29616]      
    device_list = ['cuda:0']*2 + ['cuda:1']*3 + ['cuda:2']*3 + ['cuda:3']*4
    arg_list = []

    for i, exp_id in enumerate(exps):
        fname = os.path.join(config_data_dir, f'task{task_id}/parameters.json')
        param_i = json.load(open(fname))
        param_i['seed'] = seed_list[i]
        param_i['device'] = device_list[i]
        #param_i['episode_size'] = 2    # for quick test
        env_i = Environment(param_i)
        arg_list.append( (param_i, env_i, exp_id) )
    
    with multiprocessing.Pool() as pool:
        st = time.time()
        res_mp = pool.starmap(run_train_wrapper, arg_list)
        et = time.time() - st
    print(f'total time for multi-processing: {et/60:.3f} min.')


def run_train_wrapper(param, env, exp_id):
    '''wrapper for training function.'''
    sgtrainner = StackelbergLearn(param, env)
    return sgtrainner.train(param, env, exp_id)



if __name__ == '__main__':
    task_id = 1
    #task_id = int(sys.argv[1])    # uncomment this to use bash script.
    
    #train(task_id)            # single training
    train_multiproc(task_id)   # multiple training