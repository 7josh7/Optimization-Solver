import numpy as np
import cvxpy as cp

class OptimizationSolver:
    def __init__(self, A, b, c, x0, t=1, mu=10, epsilon_inner=1e-5, epsilon_outer=1e-6):
        self.A = A
        self.b = b
        self.c = c
        self.x0 = x0
        self.t = t
        self.mu = mu
        self.epsilon_inner = epsilon_inner
        self.epsilon_outer = epsilon_outer

    def evaluate_f_gradient_hessian(self, x, t):
        m, n = self.A.shape

        f_0 = np.dot(self.c, x)
        f_i = np.dot(self.A, x) - self.b
        phi = -np.sum(np.log(-f_i))
        f_val = t * f_0 + phi

        grad_f_0 = self.c
        d = 1 / -f_i
        grad_phi = self.A.T @ d
        grad_f = t * grad_f_0 + grad_phi

        D = np.diag(d**2)
        hessian_phi = self.A.T @ D @ self.A
        hessian_f = hessian_phi

        return f_val, grad_f, hessian_f

    def newton_step(self, grad_f, hessian_f):
        delta_x_nt = -np.linalg.solve(hessian_f, grad_f)
        return delta_x_nt

    def newton_decrement(self, grad_f, delta_x_nt):
        return np.sqrt(-grad_f.T @ delta_x_nt)

    def backtracking_line_search(self, x, delta_x_nt, t, alpha=0.1, beta=0.7):
        s = 1.0
        f_x, grad_f, _ = self.evaluate_f_gradient_hessian(x, t)
        while True:
            x_new = x + s * delta_x_nt
            if np.all(self.A @ x_new < self.b):
                f_x_new, _, _ = self.evaluate_f_gradient_hessian(x_new, t)
                if f_x_new <= f_x - alpha * s * (grad_f.T @ delta_x_nt):
                    break
            s *= beta
        return s

    def solve_unconstrained_problem(self):
        x = self.x0
        outer_iterations = 0
        total_newton_steps = 0
        outer_records = []
        inner_iteration_data = []

        while True:
            outer_iterations += 1
            inner_newton_steps = 0
            
            while True:
                f_val, grad_f, hessian_f = self.evaluate_f_gradient_hessian(x, self.t)
                delta_x_nt = self.newton_step(grad_f, hessian_f)
                lambda_x = self.newton_decrement(grad_f, delta_x_nt)

                if lambda_x**2 / 2 < self.epsilon_inner:
                    break

                s = self.backtracking_line_search(x, delta_x_nt, self.t)
                x = x + s * delta_x_nt
                inner_newton_steps += 1
                total_newton_steps += 1

                inner_iteration_data.append((f_val, lambda_x, s))
            
            outer_records.append((self.t, f_val, inner_newton_steps, lambda_x, s))

            if self.A.shape[0] / self.t < self.epsilon_outer:
                break
            
            self.t *= self.mu

        return x, f_val, outer_iterations, total_newton_steps, outer_records, inner_iteration_data, self.t

    def solve_with_cvxpy(self):
        x_cvx = cp.Variable(len(self.c))
        f_0_cvx = self.c.T @ x_cvx
        phi_cvx = -cp.sum(cp.log(self.b - self.A @ x_cvx))
        objective = cp.Minimize(self.t * f_0_cvx + phi_cvx)
        constraints = [self.A @ x_cvx <= self.b]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        return x_cvx.value, result


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
import matplotlib.pyplot as plt
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