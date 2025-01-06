import numpy as np
import concurrent.futures
from Agent import Agent
from ObjectiveFunction import ObjectiveFunction
import signal

class Supervisor:

    def __init__(self):
        self.of = ObjectiveFunction
        self.x_final : [] = []
        self.x_global : [] = []
        self.x_current : [] = []
        self.u_global : [] = []
        self.u_final : np.ndarray = np.empty((0, 3))
        self.agents : [] = []
        self.max_t : float = 3
        self.max_iter : float = 30
        self.dt : float = 0.01
        self.m = np.array([1.5, 2, 1])
        self.n_agents = len(self.m)
        self.t = 0
        self.X_feasible : np.ndarray = np.array([[-5, 5], [-2, 2]])
        self.U_feasible : np.ndarray = np.array([-1.5, 1.5])
        self.Q : np.ndarray = np.diag([2, 0.05, 2, 0.05, 2, 0.05])
        self.R: np.ndarray = np.diag([0.1, 1, 0.1])
        self.Np_agents: np.ndarray = np.array([50, 50, 50])
        self.tolerance : float = 0.01
        self.timeout : int = 60

    def initialize_global_state(self):
        r0, v0 = [], [] #np.zeros(len(self.m))
        for _ in np.arange(0, len(self.m), 1):
            r0.append(np.random.uniform(self.X_feasible[0][0], self.X_feasible[0][1]))
            v0.append(np.random.uniform(self.X_feasible[1][0], self.X_feasible[1][1]))
        self.x_global = np.column_stack((r0, v0))

    def compute_lyapunov(self, x_f: np.ndarray, u_f: np.ndarray) -> float:
        P = self.of.compute_P(x_f, u_f, self.m)
        return float(x_f.T @ P @ x_f)

    def terminal_input(self, u_f: np.ndarray, x_f: np.ndarray, P: np.ndarray) -> np.ndarray:
        f_terminal : [] = []
        for i, agent in enumerate(self.agents):
            f_terminal.append(self.of.predict_state(
                x_f[:self.n_agents], float(x_f[self.n_agents + i]), float(u_f[i]), i, float(self.m[i]), self.dt))
        return x_f.T @ self.Q @ x_f + u_f.T @ self.R @ u_f + 2 * (x_f.T @ P @ np.array(f_terminal).flatten()) * self.dt

    def run(self):
        self.initialize_global_state()
        self.u_global = [np.zeros(self.Np_agents[0]), np.zeros(self.Np_agents[1]), np.zeros(self.Np_agents[2])]
        a1 = Agent(0, float(self.m[0]), self.dt, self.t, self.X_feasible, self.U_feasible, self.n_agents, self.Q, self.R, self.tolerance)
        a2 = Agent(1, float(self.m[1]), self.dt, self.t, self.X_feasible, self.U_feasible, self.n_agents, self.Q, self.R, self.tolerance)
        a3 = Agent(2, float(self.m[2]), self.dt, self.t, self.X_feasible, self.U_feasible, self.n_agents, self.Q, self.R, self.tolerance)
        self.agents = [a1, a2, a3]
        a1.x_global_set, a2.x_global_set, a3.x_global_set = self.x_global, self.x_global, self.x_global
        with concurrent.futures.ThreadPoolExecutor() as executor:
            signal.alarm(self.timeout)
            try:
                for t in np.arange(0, self.max_t, self.dt):
                    for k in np.arange(0, self.max_iter, 1):
                        if k == 0:
                            self.x_current = np.array(self.x_global[:, :2]).flatten()
                            if t == 0:
                                print(np.zeros(int(self.Np_agents[0] / self.dt)))
                                self.u_global = [np.zeros(int(self.Np_agents[0] / self.dt)), np.zeros(int(self.Np_agents[1] / self.dt)), np.zeros(int(self.Np_agents[2] / self.dt))]
                                a1.u_current, a2.u_current, a3.u_current = self.u_global[0], self.u_global[1], self.u_global[2]
                        a1.x_global, a2.x_global, a3.x_global = [], [], []
                        a1.t, a2.t, a3.t = t, t, t
                        executor.submit(a1.predict_state_sequence(self.x_current))
                        executor.submit(a2.predict_state_sequence(self.x_current))
                        executor.submit(a3.predict_state_sequence(self.x_current))
                        self.x_global = [a1.x_global, a2.x_global, a3.x_global]
                        x_f = np.concatenate((a1.x_global[-1][:2], a2.x_global[-1][2:4], a3.x_global[-1][4:6]))
                        u_f = np.array([a1.u_current[-1], a2.u_current[-1], a3.u_current[-1]])
                        print (u_f)
                        V_f = self.compute_lyapunov(x_f, u_f)
                        executor.submit(a1.predict_input_sequence(np.array(self.u_global), np.array(self.x_global), V_f))
                        executor.submit(a2.predict_input_sequence(np.array(self.u_global), np.array(self.x_global), V_f))
                        executor.submit(a3.predict_input_sequence(np.array(self.u_global), np.array(self.x_global), V_f))
                        self.u_global = [a1.u_current, a2.u_current, a3.u_current]
                        if (np.abs(a1.J_current - a1.J_past) < a1.tolerance) and (np.abs(a2.J_current - a2.J_past) < a2.tolerance) and (np.abs(a3.J_current - a3.J_past) < a3.tolerance):
                            self.u_final = np.vstack((self.u_final, [a1.u_current[0], a2.u_current[0], a3.u_current[0]]))
                            self.x_final.append(self.x_current)
                        else:
                            print("No convergence yet, going to iteration + " + str(k))
                    signal.alarm(0)
            except TimeoutError: print("Timeout, possibly non-convergent")

                #P = self.of.compute_P(x_f, u_f, np.array(self.m))
                #bounds = [(self.U_feasible[0], self.U_feasible[1])] * len(self.m)
                #terminal_guess = minimize(self.terminal_input, np.array(u_f), args=(x_f, P), bounds=bounds, method='SLSQP').x
                #u_init[:, -1] = terminal_guess

sf = Supervisor()
sf.run()

















