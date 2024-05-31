from simulator import Simulator
import numpy as np
import argparse
from skopt import gp_minimize
from skopt.plots import plot_convergence

# This script was used to optimize the lateral PID controller with bayesian optimization

def save_gp_result(res):
    np.save('func_vals.npy', res.func_vals)
    np.save('x_iters.npy', res.x_iters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    simulator = Simulator()

    # while True:
    #     params = np.loadtxt('params.txt')
    #     params = [3.25, 1., 0., 1., 35., 50., 1., 20]
    #     score = simulator.drive_track(params)
    #     print(score)
    #     breakpoint()

    # pid
    best params [3.118357247806046, 0.9755321901954155, 1.9152884533402488, 1.3782508892109167, 24.971503202815928, 23.150102938235136, 0.6406067986034124, 6.521455880467447]
    bounds = [(1.0, 3.5), (0.5, 1.9), (0.1, 2.7), (0.1, 3.0), (15., 75.), (0., 65.), (0., 1.0), (0., 20.)]
    # best score 0.0385686512762162
    # k_p, speed_scale, speed_offset, k_d, default_lookahead, speed_threshold, k_i, n

    n_calls, n_random_starts = 300, 100
    x0, y0 = None, None

    if args.load:
        x0, y0 = np.load('x_iters.npy').tolist(), np.load('func_vals.npy').tolist()
        n_calls = max(0, n_calls - len(x0))
        n_random_starts = max(0, n_random_starts - len(x0))

    res = gp_minimize(simulator.drive_track,                
                bounds,         
                n_calls=n_calls,      
                n_random_starts=n_random_starts, 
                x0 = x0,
                y0 = y0,
                verbose=True,
                callback=save_gp_result)  

    print(f"Best params: {res['x']}")
