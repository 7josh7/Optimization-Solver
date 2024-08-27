# Optimization Solver ReadMe

## Overview

This Python code provides an implementation of an optimization solver using the interior-point method and Newton's method. The solver is designed to handle constrained optimization problems by iteratively finding the optimal solution that minimizes a given objective function while satisfying certain constraints. It is also compared against a solution obtained using the `CVXPY` library for benchmarking purposes.

## Features

1. **Interior-Point Method**: The solver employs an interior-point method, which is a common technique for solving linear and nonlinear programming problems. It introduces a logarithmic barrier function to handle inequality constraints.

2. **Newton's Method**: The solver uses Newton's method for optimization within the interior-point framework to find the direction of descent and adjust the step size.

3. **Backtracking Line Search**: This is used to determine an appropriate step size that satisfies the constraints while making sufficient progress toward the minimum.

4. **Comparison with CVXPY**: The solution obtained through the custom solver is compared with a solution derived using the `CVXPY` library, a popular convex optimization package.

5. **Visualization**: The code includes functionality to visualize the Newton decrement over iterations, providing insights into the convergence behavior of the optimization process.

## Code Structure

### Class: `OptimizationSolver`

- **Initialization (`__init__`)**: 
  - Initializes the solver with matrices `A`, `b`, `c`, an initial guess `x0`, and other parameters like `t`, `mu`, `epsilon_inner`, and `epsilon_outer`.
  - Parameters:
    - `A`: Matrix of constraint coefficients.
    - `b`: Vector of constraint bounds.
    - `c`: Vector of objective function coefficients.
    - `x0`: Initial guess for the optimization variables.
    - `t`: Initial value for the barrier method.
    - `mu`: Scaling factor for updating `t`.
    - `epsilon_inner`: Tolerance for the inner Newton iteration.
    - `epsilon_outer`: Tolerance for the outer loop convergence.

- **Method: `evaluate_f_gradient_hessian`**: 
  - Evaluates the value, gradient, and Hessian of the objective function at a given point `x` for a particular `t`.
  - Computes both the barrier component and the linear objective component.

- **Method: `newton_step`**: 
  - Computes the Newton step direction using the gradient and Hessian of the function.

- **Method: `newton_decrement`**: 
  - Calculates the Newton decrement to determine the stopping criterion for the inner loop.

- **Method: `backtracking_line_search`**: 
  - Performs backtracking line search to find a suitable step size that decreases the objective function value while staying within the feasible region.

- **Method: `solve_unconstrained_problem`**: 
  - Main method to solve the optimization problem using the interior-point method with nested Newton steps.
  - Returns the optimal solution, optimal function value, number of outer and inner iterations, and detailed iteration data.

- **Method: `solve_with_cvxpy`**: 
  - Solves the same optimization problem using `CVXPY` for benchmarking against the custom solver.

## Example Usage

1. **Initialization**: The solver is initialized with randomly generated matrices `A`, `b`, `c`, and an initial guess `x0` that satisfies the constraints `A @ x0 < b`.

2. **Solving the Problem**: The problem is solved using both the custom interior-point method and the `CVXPY` library. The results are printed and compared.

3. **Visualization**: The Newton decrement is plotted over iterations to visualize the convergence of the solver.

```python
# Create an instance of OptimizationSolver
solver = OptimizationSolver(A, b, c, x0)

# Solve using the custom method
x_opt, f_opt, outer_iterations, total_newton_steps, outer_records, inner_iteration_data, final_t = solver.solve_unconstrained_problem()

# Plotting the Newton decrement
import matplotlib.pyplot as plt
lambdas = [data[3]**2 / 2 for data in outer_records]
plt.plot(lambdas)
plt.xlabel('Iteration (k)')
plt.ylabel('λ(x(k))^2 / 2')
plt.title('Newton Decrement over Iterations')
plt.show()

# Solve using CVXPY for comparison
x_cvx, result = solver.solve_with_cvxpy()
```

## Requirements

- Python 3.x
- `numpy` for numerical computations
- `cvxpy` for solving the problem using the CVXPY library
- `matplotlib` for plotting results

## Notes

- The `OptimizationSolver` class is designed to handle constrained optimization problems where the constraints are defined by a matrix inequality \(Ax < b\).
- The example usage demonstrates solving a randomly generated problem; in practice, the matrices `A`, `b`, and `c` should be defined based on specific optimization scenarios.
- The solver's performance and accuracy can be benchmarked by comparing its results with those obtained using `CVXPY`.

This implementation provides a robust framework for learning and experimenting with optimization techniques, especially in financial engineering and other fields requiring efficient numerical solutions to constrained problems.
