import numpy as np

def univariate_gaussian_data_generator(m, s):
    # m = mean, s = variance
    # Box-Muller transform: Z = {[-2 * ln(U)] ** 1/2} * cos(2π * V)
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)
    Z = np.sqrt(-2.0 * np.log(U)) * np.cos(2.0 * np.pi * V)

    return m + Z * np.sqrt(s)

def polynomial_basis_linear_model_data_generator(n, a, w):
    # n = basis number, a = variance of error, w = coefficients of polynomial
    # y = W_T * phi(x) + e, e ~ N(0, a)
    x = np.random.uniform(-1, 1)
    phi_x = np.array([x ** i for i in range(n)])
    e = univariate_gaussian_data_generator(0, a)
    y = np.dot(w, phi_x) + e

    return x, y

if __name__ == "__main__":
    # Example usage
    m = 0  # mean
    s = 1  # variance
    print("Univariate Gaussian data generator:", univariate_gaussian_data_generator(m, s))

    n = 3  # basis number
    a = 0.1  # variance of error
    w = np.array([1.0, -2.0, 3.0])  # coefficients of polynomial
    x, y = polynomial_basis_linear_model_data_generator(n, a, w)
    print("Polynomial basis linear model data generator:", (x, y))