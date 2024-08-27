import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import os
from my_objective_with_log_barrier import OptimizationSolver

def main():
    np.random.seed(0)
    n = 5
    m = 10
    A = np.random.randn(m, n)
    b = np.random.rand(m) + 1  # Ensure b_i > 0
    c = np.random.randn(n)
    x0 = np.random.rand(n)

    while not np.all(np.dot(A, x0) < b):
        x0 = np.random.rand(n)

    solver = OptimizationSolver(A, b, c, x0)
    x_opt, f_opt, outer_iterations, total_newton_steps, outer_records, inner_iteration_data, final_t = solver.solve_unconstrained_problem()
    print("1.j Outer iteration records (t, f_val, inner_newton_steps, lambda_x, s):", outer_records)
    lambdas = [data[3]**2 / 2 for data in outer_records]
    plt.plot(lambdas)
    plt.xlabel('Iteration (k)')
    plt.ylabel('λ(x(k))^2 / 2')
    plt.title('Newton Decrement over Iterations')
    plt.show()
    
    print("1.l (1) Number of outer iterations:", outer_iterations)
    print("1.l (2) Number of inner_newton_steps:", [data[2] for data in outer_records])
    print("1.l (3) t and the function value:", [data[0:2] for data in outer_records])
    print("1.l (4) Total number of Newton steps:", total_newton_steps)
    print("1.l (5) Optimal point x:", x_opt)

    # Solve the same problem using CVXPY
    x_cvx = cp.Variable(n)
    objective = cp.Minimize(c.T @ x_cvx)
    constraints = [A @ x_cvx <= b]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()


    print("1.m CVX optimal point x:", x_cvx.value)
    print("CVX optimal value f(x):", result)
    print("CVX optimal value f(x) after multipling final_t:", result * final_t)

    print("optimal point x difference:", x_opt - x_cvx.value)
    print("optimal value difference:", f_opt / final_t - result)


if __name__ == "__main__":
    main()
