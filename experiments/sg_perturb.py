# test perturbation using the learned model
import os
import json
from sg_taskplan.env import Environment
from sg_taskplan.perturbation import DisturbEval
from sg_taskplan import config_data_dir


if __name__ == '__main__':
    task_id = 1
    fname = os.path.join(config_data_dir, f'task{task_id}/parameters.json')
    param = json.load(open(fname))
    env = Environment(param)

    # evaluate perturbations using different learned models
    sgeval = DisturbEval(env)
    exps = [x+1 for x in range(10)]
    for exp_id in exps:
        sgeval.eval(param, env, exp_id)
