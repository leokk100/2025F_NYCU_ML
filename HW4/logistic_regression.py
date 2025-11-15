import numpy as np
import matplotlib.pyplot as plt

def univariate_gaussian_data_generator(m, s):
    # m = mean, s = variance
    # Box-Muller transform: Z = {[-2 * ln(U)] ** 1/2} * cos(2π * V)
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)
    Z = np.sqrt(-2.0 * np.log(U)) * np.cos(2.0 * np.pi * V)

    return m + Z * np.sqrt(s)

def draw(D, weights, labels, axs, title):
    predicted_0 = []
    predicted_1 = []
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(D.shape[0]):
        z = D[i] @ weights
        prediction = 1 / (1 + np.exp(-z))
        if prediction < 0.5:
            predicted_0.append(D[i, 1:])
            if labels[i] == 0:
                tp += 1
            else:
                fp += 1
        else:
            predicted_1.append(D[i, 1:])
            if labels[i] == 1:
                tn += 1
            else:
                fn += 1
    axs.plot(np.array(predicted_0)[:, 0], np.array(predicted_0)[:, 1], 'ro')
    axs.plot(np.array(predicted_1)[:, 0], np.array(predicted_1)[:, 1], 'bo')
    axs.set_title(title)

    # Print results
    if title == 'Gradient descent':
        print('Gradient descent:')
    else:
        print('Newton\'s method:')    
    print()
    print('w:')
    print(weights)
    print()
    print('Confusion Matrix:')
    print('\t\t\tPredict cluster 1\tPredict cluster 2')
    print(f'Is cluster 1\t\t\t{tp}\t\t\t{fn}')
    print(f'Is cluster 2\t\t\t{fp}\t\t\t{tn}')
    print()
    print('Sensitivity (Successfully predict cluster 1): ', tp / (tp + fn))
    print('Specificity (Successfully predict cluster 2): ', tn / (tn + fp))
    print()


def gradient_descent(D, labels, learning_rate=0.01, epsilon=1e-6, num_iterations=1000):
    m, n = D.shape # m = number of data points * 2, n = number of features(3)
    weights = np.random.rand(n, 1) # shape = (3, 1)

    for _ in range(num_iterations):
        z = D @ weights # shape = (m, 1)
        predictions = 1 / (1 + np.exp(-z)) # shape = (m, 1)
        errors = predictions - labels # shape = (m, 1)
        gradient = D.T @ errors # shape = (n, 1)
        if np.linalg.norm(learning_rate * gradient) < epsilon:
            break
        weights -= learning_rate * gradient

    draw(D, weights, labels, axs[1], title='Gradient descent')

def newtons_method(D, labels, learning_rate=0.01, epsilon=1e-6, num_iterations=1000):
    m, n = D.shape # m = number of data points * 2, n = number of features(3)
    weights = np.random.rand(n, 1) # shape = (3, 1)

    for _ in range(num_iterations):
        DD = np.zeros((m, m))

        for i in range(m):
            temp = (D[i] @ weights)[0] # scalar
            DD[i, i] = (np.exp(-temp) / ((1 + np.exp(-temp)) ** 2))
        
        H = D.T @ DD @ D # Hessian: shape = (n, n)
        
        # calculate gradient (the same as in gradient descent)
        z = D @ weights # shape = (m, 1)
        predictions = 1 / (1 + np.exp(-z)) # shape = (m, 1)
        errors = predictions - labels # shape = (m, 1)
        gradient = D.T @ errors # shape = (n, 1)
        
        if np.linalg.det(H) != 0:
            H_inv = np.linalg.inv(H) # shape = (n, n)
            gradient = H_inv @ gradient # shape = (n, 1)

        if np.linalg.norm(learning_rate * gradient) < epsilon:
            break
        
        weights -= learning_rate * gradient

    draw(D, weights, labels, axs[2], title='Newton\'s method')

if __name__ == '__main__':
    n = 50 # number of data points
    mx1, my1 = 1, 1
    mx2, my2 = 3, 3
    vx1, vy1 = 2, 2
    vx2, vy2 = 4, 4
    fig, axs = plt.subplots(1, 3) # axs[0]: ground truth, axs[1]: gradient descent, axs[2]: Newton's method

    D1_x = [univariate_gaussian_data_generator(mx1, vx1) for _ in range(n)] # shape = (n, )
    D1_y = [univariate_gaussian_data_generator(my1, vy1) for _ in range(n)] # shape = (n, )
    D2_x = [univariate_gaussian_data_generator(mx2, vx2) for _ in range(n)] # shape = (n, )
    D2_y = [univariate_gaussian_data_generator(my2, vy2) for _ in range(n)] # shape = (n, )

    D1 = np.stack((D1_x, D1_y), axis=1) # shape = (n, 2)
    D2 = np.stack((D2_x, D2_y), axis=1) # shape = (n, 2)
    D = np.concatenate((D1, D2), axis=0) # shape = (2n, 2)
    ones = np.ones((2*n, 1)) # shape = (2n, 1)
    D = np.concatenate((ones, D), axis=1) # shape = (2n, 3)
    labels = np.array([0]*n + [1]*n).reshape(2*n, 1) # shape = (2n, 1)

    # Plot ground truth
    axs[0].plot(D1[:, 0], D1[:, 1], 'ro', label='cluster 1')
    axs[0].plot(D2[:, 0], D2[:, 1], 'bo', label='cluster 2')
    axs[0].set_title('Ground truth')

    # Gradient descent
    gradient_descent(D, labels)
    print('----------------------------------------------------------------------')

    # Newton's method
    newtons_method(D, labels)

    fig.savefig('logistic_regression.png')
    

    
    