import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from . import config_data_dir, plot_data_dir


class Environment:
    def __init__(self, parameters) -> None:
        self.rng = np.random.default_rng(parameters['seed'])
        self.task_id = parameters['task_id']
        self.task_board, self.task_prop = self.task_reader(self.task_id)
        self.curr_board = np.copy(self.task_board)


    def task_reader(self, task_id):
        '''
        Read the task info given the task id.
        '''
        fboard_name = os.path.join(config_data_dir, f'task{task_id}/task_board.csv')
        #fboard_name = 'data/task' + str(task_id) + '/task_board.csv'
        task_board = np.loadtxt(fboard_name, dtype=int, delimiter=',')

        fprop_name = os.path.join(config_data_dir, f'task{task_id}/task_property.csv')
        #fprop_name = 'data/task' + str(task_id) + '/task_property.csv'
        a = np.loadtxt(fprop_name, dtype=str, delimiter=',')
        task_prop = {'type': a[1:, 1].astype(int), 'shape': a[1:, 2].astype(int), 
                     'l_succ': a[1:, 3].astype(float), 'f_succ': a[1:, 4].astype(float)}
        
        return task_board, task_prop
    

    def task_viewer(self, board=None):
        '''
        View the task board.
        '''
        task_board = self.task_board if board is None else board
        task_type = self.task_prop['type']
        task_shape = self.task_prop['shape']
        # color_task = {0: (1,1,1),
        #               1: (0.55686275, 0.62745098, 0.78039216),
        #               2: (0.50980392, 0.74901961, 0.65098039),
        #               3: (0.92156863, 0.56078431, 0.41960784),
        #               4: (0.98039216, 0.84313725, 0.3254902)}
        color_task = {0: (1,1,1),
                      1: (0.74117647, 0.84313725, 0.93333333),
                      2: (0.95686275, 0.69411765, 0.51372549),
                      3: (1.        , 0.90196078, 0.6       ),
                      4: (0.77254902, 0.87843137, 0.70588235)}
        fig, ax = plt.subplots()

        # plot task color blocks
        for i in range(task_board.shape[0]):
            for j in range(task_board.shape[1]):
                rect = Rectangle((j-0.5,i-0.5), 1, 1, color=color_task[ task_type[task_board[i,j]] ], ec=None)
                ax.add_patch(rect)
        
        # plot task separator
        for i in range(task_board.shape[0]):
            ax.plot([-0.5,task_board.shape[1]-0.5], [i+0.5, i+0.5], color='k', linestyle='-', linewidth=0.5)    # horizontal separator
            j = 0
            while j < task_board.shape[1]:
                sh = task_shape[task_board[i,j]]
                if sh == 0:     # skip empty task
                    j += 1
                    continue

                cx, cy = j + (sh-1) / 2, i
                ax.plot([j-0.5,j-0.5], [i-0.5, i+0.5], color='k', linestyle='-', linewidth=0.5)                 # vertial separator
                ax.plot([j+(sh-1)+0.5,j+(sh-1)+0.5], [i-0.5, i+0.5], color='k', linestyle='-', linewidth=0.5)   # vertial separator
                r = 'T' + str(task_board[i,j])
                ax.annotate(r, (cx,cy), color='k', weight='regular', fontsize=8, ha='center', va='center')     # task name
                j += sh

        # plot axis border
        ax.plot([-0.5,-0.5], [-0.5, task_board.shape[0]-0.5], color='k', linestyle='-', linewidth=1)    # left border
        ax.plot([task_board.shape[1]-0.5, task_board.shape[1]-0.5], [-0.5, task_board.shape[0]-0.5], color='k', linestyle='-', linewidth=1)    # right border
        ax.plot([-0.5, task_board.shape[1]-0.5], [-0.5, -0.5], color='k', linestyle='-', linewidth=1)    # lower border
        ax.plot([-0.5, task_board.shape[1]-0.5], [task_board.shape[0]-0.5, task_board.shape[0]-0.5], color='k', linestyle='-', linewidth=1)    # upper border

        # add legend
        ax.add_patch(Rectangle((task_board.shape[1]+0.5, 0), 0.5, 0.5, color=color_task[4], ec=None))
        ax.annotate('Type 4', (task_board.shape[1]+1.5,0.25), color='k', weight='regular', fontsize=6, ha='center', va='center')
        ax.add_patch(Rectangle((task_board.shape[1]+0.5, 1), 0.5, 0.5, color=color_task[3], ec=None))
        ax.annotate('Type 3', (task_board.shape[1]+1.5,1.25), color='k', weight='regular', fontsize=6, ha='center', va='center')
        ax.add_patch(Rectangle((task_board.shape[1]+0.5, 2.), 0.5, 0.5, color=color_task[2], ec=None))
        ax.annotate('Type 2', (task_board.shape[1]+1.5,2.25), color='k', weight='regular', fontsize=6, ha='center', va='center')
        ax.add_patch(Rectangle((task_board.shape[1]+0.5, 3), 0.5, 0.5, color=color_task[1], ec=None))
        ax.annotate('Type 1', (task_board.shape[1]+1.5,3.25), color='k', weight='regular', fontsize=6, ha='center', va='center')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((-0.5-0.01, task_board.shape[1]+1.5))
        #ax.set_ylim((-0.5-0.01, task_board.shape[0]-0.5+0.01))
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_title('Task chessboard for Task' + str(self.task_id))
        
        fname = os.path.join(plot_data_dir, f'task_board_{self.task_id}.png')
        fig.savefig(fname, dpi=300)
        plt.close(fig)


    def get_task_info(self): 
        '''
        Get task info. Used to initialize trainner.
        '''
        a = {}
        a['task_id'] = self.task_id
        a['dims'] = self.task_board.shape[1]
        a['dimAl'] = self.task_board.shape[1] + 1       # A = [-1, 0, 1,..., n-1], -1 means idle action
        a['dimAf'] = self.task_board.shape[1] + 1       # same action space as leader
        a['dimal'] = 1
        a['dimaf'] = 1
        a['task_prop'] = self.task_prop
        return a


    def get_current_state(self):
        '''
        Use all-zero vector to represent the completion state of the task.
        '''
        return np.copy(self.curr_board[0, :]), np.copy(self.curr_board)
    

    def set_env(self, board):
        self.curr_board = np.copy(board)


    def reset_env(self):
        '''
        Reset the current environment.
        '''
        self.curr_board = np.copy(self.task_board)


    def set_deterministic(self):
        '''
        Set deterministic transition kernel.
        '''
        self.task_prop['l_succ'][self.task_prop['l_succ'] > 0] = 1
        self.task_prop['f_succ'][self.task_prop['f_succ'] > 0] = 1

    
    def step(self, al, af):
        '''
        One-step forward simulation using the action paris (al, af).
        '''
        # simulate if task is completed
        if al == -1:
            tl, tl_done = 0, False  # select idle
        else:
            tl = self.curr_board[0, al]
            if tl == 0: 
                tl_done = False     # not idle but select 0 task (nothing)
            else:
                tl_done = True if self.rng.uniform() < self.task_prop['l_succ'][tl] else False
        
        if af == -1:
            tf, tf_done = 0, False
        else:
            tf = self.curr_board[0, af]
            if tf == 0:
                tf_done = False
            else:
                tf_done = True if self.rng.uniform() < self.task_prop['f_succ'][tf] else False
        #print(f'leader chooses T{tl}, {tl_done}; follower chooses T{tf}, {tf_done}.')

        # update the task board based on the simulated result.
        self.update_board(tl, tl_done, tf, tf_done)

        return  # no return values


    def update_board(self, tl, tl_done, tf, tf_done):
        '''
        Update current borad based on the simulated task completion results.
        '''
        # 1: update the first row (bottom row of the borad)
        if tl == tf:
            if tl != 0 and self.task_prop['type'][tl] == 4 and tl_done and tf_done:       # complete cooperative task
                idx = np.where(self.curr_board[0] == tl)[0]
                self.curr_board[0, idx] = 0
            else:
                pass    # 1. both do nothing; 2. choose the same noncooperative task; 3. both fail to complete cooperative task
        else:
            if tl != 0 and tl_done and self.task_prop['type'][tl] != 4:
                idx = np.where(self.curr_board[0] == tl)[0]   # set all tl blocks to 0
                self.curr_board[0, idx] = 0
            if tf != 0 and tf_done and self.task_prop['type'][tf] != 4:
                idx = np.where(self.curr_board[0] == tf)[0]   # set all tf blocks to 0
                self.curr_board[0, idx] = 0

        for i in range(self.task_board.shape[0]-1):
            curr_row, next_row = self.curr_board[i, :], self.curr_board[i+1, :]
            # 2: find all task id that may drop from next_row, using curr_row 0 index
            task_list = []
            idx = np.where(curr_row == 0)[0]
            for i in idx:
                task_id = next_row[i]
                if task_id !=0 and task_id not in task_list:    # task 0 does not count
                    task_list.append(task_id)

            # 3: compare curr_row and next_row, then update them
            mod_flag = False
            for ti in task_list:
                idx = np.where(next_row == ti)[0]
                if np.all(curr_row[idx] == 0):
                    curr_row[idx] = ti
                    next_row[idx] = 0
                    mod_flag = True
            if not mod_flag:    # no modification made, no need to update future rows
                break
        
        return  # no return values
    

    def reward(self, s, al, af):
        '''
        Reward function for Stackelberg game, nash game, independent learning.
        - If both cooperative, both get reward. 
        - If both idle or same task (except for cooperative task), negative reward.
        - If the one select coop alone or wrong type, get negative reward.
        - If one idle, zero reward.
        '''
        # determine task id that corresponds to the action
        tl = 0 if al == -1 else s[al]
        tf = 0 if af == -1 else s[af]

        if tl == 0 and tf == 0:     # both choose zero task
            if al == -1 and af == -1:
                rl, rf = -0.5, -0.5 # both idle
            elif al == -1 and af != -1:
                rl, rf = 0, -1      # leader idle, follower empty task
            elif al != -1 and af == -1:
                rl, rf = -1, 0      # leader empty task, follower idle
            else:
                rl, rf = -2, -2     # leader and follower both empty task
        elif tl == tf and tl != 0 and tf != 0:  # both choose the same nonzero task
            if self.task_prop['type'][tl] == 4:
                rl, rf = 2, 2       # choose cooperative task
            else:
                rl, rf = -1, -1     # choose same noncooperative task, collide
        else:   # choose different task
            # process leader
            if tl == 0:
                rl = 0 if al == -1 else -1  # leader either idle or empty task
            elif self.task_prop['type'][tl] == 1 or self.task_prop['type'][tl] == 3:
                rl = 1      # choose right type task
            else:
                rl = -1     # choose wrong task or do cooperative tak along
            # process follower
            if tf == 0:
                rf = 0 if af == -1 else -1  # follower either idle or empty task
            elif self.task_prop['type'][tf] == 2 or self.task_prop['type'][tf] == 3:
                rf = 1      # choose right type task
            else:
                rf = -1     # choose wrong task or do cooperative task along
        
        return float(rl), float(rf)
    

    def generate_random_state(self, N=50):
        '''
        Generate a set of random states to start the episodic training.
        '''
        board_list = []
        _, board = self.get_current_state()
        board_list.append(board)
        action_set = np.arange(-1, self.task_board.shape[1])
        while len(board_list) < N:
            self.reset_env()
            s0, _, = self.get_current_state()
            iter = 0
            while not np.all(s0 == 0) and iter < 100:   # aviod too many failures completing the task
                al = self.rng.choice(action_set)
                af = self.rng.choice(action_set)
                self.step(al, af)
                s0, board = self.get_current_state()
                board_list.append(board)
                #self.task_viewer(board)
        board_list = np.array(board_list)       # no shuffle, order is one-to-one correspondence
        
        # choose N feasible states
        idx = np.arange(board_list.shape[0], N, replace=False)
        return board_list[idx, :]
