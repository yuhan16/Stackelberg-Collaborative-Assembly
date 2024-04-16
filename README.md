# Stackelberg Learning for Collaborative Assembly Task Planning
This repo is for the project of Stackelberg learning for collaborative assembly task planning.


## Requirements
- Python 3.10 or higher
- Pytorch 2.0.1 or higher


## Running Scripts
1. Create a python virtual envrionment with Python 3.11 and source the virtual environment: 
```bash
$ python3.11 -m venv <your-virtual-env-name>
$ source /path-to-venv/bin/activate
```
2. Use `pip` to install related packages:
```bash
(your-venv)$ pip install -e .
```
To use plotting functions, install with 
```bash
(your-venv)$ pip install -e ".[visual]"
```
3. Go to `experiments/` directory and run different training scripts. e.g.,
```bash
(your-venv)$ python sg_train.py
```
**Note:** `sg_perturb.py` and `plot_things.py` should be run after all trainings are completed.


## Project Structure
- `sg_task/`: algorithm implementations
    - `data/`: environment settings and learning hyperparameters.
    - `sg/`: Stackelberg learning functions.
    - `other/`: nash and independent learning functions.
    - `env.py`: environment implementations.
    - `perturbation.py`: perturbation test with Stackelberg learning models.
    - `utils.py`: miscellaneous utilities.
- `data/`: data directory, for saving generated data and learned models.
- `experiments/`: python scripts to run the experiments.
- `tests/`: test scripts.
- `scripts/`: bash scripts to run the code.


### Comparison Algorithm
- `nash`: Nash Q-learning algorithm.
- `ind`: Independent learning algorithm.
- `maddpg`: Multi-Agent Deep Deterministic Policy Gradient, see [maddpg](https://github.com/openai/maddpg)


### Tuining Parameters
The hyperparameters and environment configurations are all in the `sg_task/data/` director. New customized tasks can be freely added by following the structure of Task 1-8.


### Parallel Running
We use `multiprocessing` package to run the training of a specific task over different experiments in parallel. See code in `experiments/` directory. The training seeds and devices can be set manually for different experiments.

We use `bash` to run the training of different tasks in parallel. 

To enable command-line options, uncomment the following statement in every training script in the `experiments/` directory:
```python
task_id = int(sys.argv[1])  # uncomment this to use bash script.
```

### Coding Specifications
- action: use list. `Al = Af = [-1, 0, 1,..., n-1]`, -1 means do nothing, the board width is n
- buffer: use 2D numpy array. `D[i, :] = [s, al, af, rl, rf, s_new]`
- Q-function in Stackelberg learning: `dims -> Al x Af`. For output vector, we use order `[(al_0,af_0), (al_0,af_1), ..., (al_0,af_m), ..., (al_m,af_m)]`