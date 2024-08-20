# gwo.py 

# ================================================================
#                  ____  ____  ______      __   _    __      
#                 / ___|/ ___||  _ \ \    / /  | |  / /  _   _ 
#                | |  _ \___ \| | | \ \  / /   | | / /  | | | |
#                | |_| |___) | |_| |\ \/ /    | |/ /   | |_| |
#                 \____|____/|____/  \__/     |___/     \__,_|
#  
#                   Grey Wolf Optimizer (GWO)
#               Optimizing Filters and Learning Rate
#  
#                    Developed by sudo-xda
#                    License: GPL-3.0
# ================================================================


import numpy as np

class GWO:
    def __init__(self, obj_function, lb, ub, dim, n_agents, max_iter):
        self.obj_function = obj_function
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.n_agents = n_agents
        self.max_iter = max_iter

    def run(self):
        positions = np.random.uniform(self.lb, self.ub, (self.n_agents, self.dim))
        alpha_position = np.zeros(self.dim)
        alpha_score = float("inf")
        beta_position = np.zeros(self.dim)
        beta_score = float("inf")
        delta_position = np.zeros(self.dim)
        delta_score = float("inf")

        for iteration in range(self.max_iter):
            for i in range(self.n_agents):
                fitness = self.obj_function(positions[i])

                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_position = beta_position.copy()

                    beta_score = alpha_score
                    beta_position = alpha_position.copy()

                    alpha_score = fitness
                    alpha_position = positions[i].copy()
                elif fitness < beta_score:
                    delta_score = beta_score
                    delta_position = beta_position.copy()

                    beta_score = fitness
                    beta_position = positions[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_position = positions[i].copy()

            for i in range(self.n_agents):
                a = 2 - iteration * (2 / self.max_iter)

                for j in range(self.dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(C1 * alpha_position[j] - positions[i][j])
                    X1 = alpha_position[j] - A1 * D_alpha

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    D_beta = abs(C2 * beta_position[j] - positions[i][j])
                    X2 = beta_position[j] - A2 * D_beta

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(C3 * delta_position[j] - positions[i][j])
                    X3 = delta_position[j] - A3 * D_delta

                    positions[i][j] = (X1 + X2 + X3) / 3

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Loss: {alpha_score:.6f}, Best Position: {alpha_position}")

        return alpha_position, alpha_score
