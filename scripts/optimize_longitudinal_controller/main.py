from simulator import Simulator
import numpy as np
import argparse
from skopt import gp_minimize
from skopt.plots import plot_convergence

# This script is used to optimize the longitudinal linear regression params with bayesian optimization

def save_gp_result(res):
    np.save('func_vals.npy', res.func_vals)
    np.save('x_iters.npy', res.x_iters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    simulator = Simulator()

    # while True:
    #     params = [1.1990342347353184, -0.8057602384167799, 1.710818710950062, 0.921890257450335, 1.556497522998393, -0.7013479734904027, 1.031266635497984]
    #     score = simulator.drive_track(params)
    #     print(score)
    #     breakpoint()

    # pid:
    # bounds = [(0., 2.5), (0., 2.5), (0., 2.5), (0., 10.), (1., 1.2)]
    # score: 0.3249
    # params: [1.0016429066823955, 1.5761818624794222, 0.2941563856687906, 0.0, 1.0324622059220139]
    # k_p, k_d, k_i, max_length_window, braking_ratio

    # polynomials
    # score: 0.1975
    best params:  [1.1990342347353184, -0.8057602384167799, 1.710818710950062, 0.921890257450335, 1.556497522998393, -0.7013479734904027, 1.031266635497984]
    bounds = [(-2, 2), (-2., 2.), (-2., 2.), (-5., 5.), (-2., 2.), (-2., 2.), (-2., 2.), (-2., 2.), (1.0, 1.1)]

    n_calls, n_random_starts = 500, 150
    x0, y0 = None, None

    if args.load:
        x0, y0 = np.load('x_iters.npy').tolist(), np.load('func_vals.npy').tolist()
        n_calls = max(0, n_calls - len(x0))
        n_random_starts = max(0, n_random_starts - len(x0))

        argmin_func_value = np.argmin(y0)
        simulator.best_score = y0[argmin_func_value]
        simulator.best_parameters = x0[argmin_func_value]

    res = gp_minimize(simulator.drive_track,                
                bounds,         
                n_calls=n_calls,      
                n_random_starts=n_random_starts, 
                x0 = x0,
                y0 = y0,
                verbose=True,
                n_jobs=-1,
                callback=save_gp_result)  

    print(f"Best params: {res['x']}")
