import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import os
from my_objective_with_log_barrier import OptimizationSolver

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    A_big = pd.read_csv('A_big.csv', header=None).values
    b_big = pd.read_csv('b_big.csv', header=None).values.flatten()
    c_big = pd.read_csv('c_big.csv', header=None).values.flatten()
    x0_big = pd.read_csv('x0_big.csv', header=None).values.flatten()

    solver_big = OptimizationSolver(A_big, b_big, c_big, x0_big)
    x_opt_big, f_opt_big, outer_iterations_big, total_newton_steps_big, outer_records_big, inner_iteration_data_big, final_t_big = solver_big.solve_unconstrained_problem()
    
    print("1.j Outer iteration records (t, f_val, inner_newton_steps, lambda_x, s):", outer_records_big)
    lambdas_big = [data[3]**2 / 2 for data in outer_records_big]
    plt.plot(lambdas_big)
    plt.xlabel('Iteration (k)')
    plt.ylabel('λ(x(k))^2 / 2')
    plt.title('Newton Decrement over Iterations')
    plt.show()
    
    print("1.l (1) Number of outer iterations:", outer_iterations_big)
    print("1.l (2) Number of inner_newton_steps:", [data[2] for data in outer_records_big])
    print("1.l (3) t and the function value:", [data[0:2] for data in outer_records_big])
    print("1.l (4) Total number of Newton steps:", total_newton_steps_big)
    print("1.l (5) Optimal point x:", x_opt_big)


    
    # Solve the large problem using CVXPY
    n = c_big.shape
    x_cvx_big = cp.Variable(n)
    objective_big = cp.Minimize(c_big.T @ x_cvx_big)
    constraints_big = [A_big @ x_cvx_big <= b_big]
    prob_big = cp.Problem(objective_big, constraints_big)
    result_big = prob_big.solve()

    print("1.m CVX optimal point for large problem x:", x_cvx_big.value)
    print("CVX optimal value for large problem f(x):", result_big)
    print("CVX optimal value f(x) after multipling final_t:", result_big * final_t_big)

    print("optimal point x difference:", x_opt_big - x_cvx_big.value)
    print("optimal value difference:", f_opt_big / final_t_big - result_big)

if __name__ == "__main__":
    main()
