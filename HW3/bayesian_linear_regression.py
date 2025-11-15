import numpy as np
import matplotlib.pyplot as plt
import random_data_generator as rdg

def visualization(ax, a, num_points, x_values, y_values, mean, variance, title, n):
    t = np.linspace(-2, 2, 500)
    mean_predict = np.zeros(500)
    var_predict = np.zeros(500)
    for i in range(500):
        X = np.asarray([t[i]**k for k in range(n)])
        mean_predict[i] = (X @ mean).item(0)  # item(0): 1x1 array to scalar
        var_predict[i] = (a + X @ variance @ X.T).item(0)

    ax.plot(x_values[0:num_points], y_values[0:num_points], 'bo')
    ax.plot(t, mean_predict, 'k-')
    ax.plot(t, mean_predict + var_predict, 'r-')
    ax.plot(t, mean_predict - var_predict, 'r-')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-20, 20)
    ax.set_title(title)


def bayesian_linear_regression(b, n, a, w):
    prior_covariance = np.identity(n) / b  # covariance matrix, initial = (b ** (-1))I
    prior_lambda = np.linalg.inv(prior_covariance)  # lambda = covariance ** (-1)
    prior_mean = np.zeros((n, 1))  # mean vector, initial = 0
    x_values, y_values = [], []
    num_points = 0
    previous_predictive_variance = None
    threshold = 1e-6
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    while num_points < 1000:
        x, y = rdg.polynomial_basis_linear_model_data_generator(n, a, w)
        x_values.append(x)
        y_values.append(y)
        print(f'Add data point: ({x}, {y})')
        print()

        phi_x = np.array([x ** i for i in range(n)]).reshape((1, n))
        posterior_lambda = prior_lambda + (1 / a) * np.dot(phi_x.T, phi_x)
        posterior_covariance = np.linalg.inv(posterior_lambda)
        posterior_mean = np.dot(posterior_covariance, np.dot(prior_lambda, prior_mean) + (1 / a) * phi_x.T * y)
        print('Posterior mean:')
        print(posterior_mean)
        print()
        print('Posterior variance:')
        print(posterior_covariance)
        print()

        # predictive distribution
        predictive_mean = np.dot(phi_x, posterior_mean).item(0)
        predictive_variance = a + np.dot(phi_x, np.dot(posterior_covariance, phi_x.T)).item(0)
        print(f'Predictive distribution ~ N({predictive_mean:.5f}, {predictive_variance:.5f})')
        print()

        if previous_predictive_variance is not None:
            variance_difference = abs(predictive_variance - previous_predictive_variance)

            if variance_difference < threshold:
                break

        previous_predictive_variance = predictive_variance

        prior_lambda = posterior_lambda
        prior_covariance = posterior_covariance
        prior_mean = posterior_mean

        if num_points == 9:
            visualization(axs[1, 0], a, num_points + 1, x_values, y_values, posterior_mean, posterior_covariance, f'After {num_points + 1} incomes', n)
        if num_points == 49:
            visualization(axs[1, 1], a, num_points + 1, x_values, y_values, posterior_mean, posterior_covariance, f'After {num_points + 1} incomes', n)

        num_points += 1
    
    visualization(axs[0, 1], a, num_points, x_values, y_values, posterior_mean, posterior_covariance, f'Predict result', n)
    visualization(axs[0, 0], a, 0, x_values, y_values, w, np.zeros((n,n)), f'Ground truth', n)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig('bayesian_linear_regression.png')

    print("Total number of iterations: ", num_points)


if __name__ == "__main__":
    b = 1
    n = 3
    a = 3
    w = [1, 2, 3]
    bayesian_linear_regression(b, n, a, w)
