import numpy as np
import matplotlib.pyplot as plt
import sys

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    return data

def get_A_matrix(data, bases):
    A = np.zeros((data.shape[0], bases)) # A is initialized as a (n * bases) zeros matrix
    for i in range(bases): 
        A[:,i] = data[:,0] ** (bases - 1 - i) 
    return A 

def transpose(mat):
    res = np.zeros((mat.shape[1], mat.shape[0])) # result is initialized as a zeros matrix
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            res[j,i] = mat[i,j]
    return res

def matmul(mat1, mat2):
    res = np.zeros((mat1.shape[0], mat2.shape[1])) # result is initialized as a zeros matrix
    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[1]):
            for k in range(mat1.shape[1]):
                res[i,j] += mat1[i,k] * mat2[k,j] 
    return res

def get_gradient(A, A_T, w, b, lambdas): # gradient = 2 * A_T * (A * w - b) + λ * sign(w)
    A_w = matmul(A, w.reshape(-1,1))
    gradient = (2 * matmul(A_T, A_w - b)).flatten() + lambdas * np.sign(w)
    return gradient

def draw(w):
    data = load_data(sys.argv[1])
    bases = int(sys.argv[2])

    # draw the real scattering data
    plt.scatter(data[:,0], data[:,1], label='Data') 

    # draw the line created by steepest descent method
    x_line = np.linspace(min(data[:,0]) - 1, max(data[:,0]) + 1, 100)
    x_matrix = np.zeros((len(x_line), 2))
    x_matrix[:,0] = x_line
    y_line = matmul(get_A_matrix(x_matrix, bases), w).flatten() # (n, 1) -> (n, )
    plt.plot(x_line, y_line, color='red', label='Fitted curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Steepest descent method with n = {sys.argv[2]}, λ = {sys.argv[3]}', fontsize=12)

    plt.savefig('plot_steepest_descent.png')

def steepest_descent(bases, lambdas, max_iter, lr, tol):
    data = load_data(sys.argv[1])
    A = get_A_matrix(data, bases) # get A
    A_T = transpose(A) # get A_T
    b = data[:,1].reshape(data.shape[0],1) # get b

    # the iteration to find the min w
    w_old = np.zeros(bases) # w_init is initialized as a (bases, ) vector
    w_new = np.zeros(bases) # w_new is initialized as a (bases, ) vector
   
    for i in range(max_iter):
        gradient = get_gradient(A, A_T, w_old, b, lambdas)
        w_new = w_old - lr * gradient

        if np.linalg.norm(abs(w_new - w_old)) < tol:
            break

        w_old = w_new

    # calculate error
    y_pred = matmul(A, w_new.reshape(-1,1)).flatten()
    error = np.sum((y_pred - data[:,1]) ** 2)
    
    # print out equation and total error
    print('Steepest descent:')
    print(f'Case: n = {bases}, λ = {lambdas}, max iteration = {max_iter}, learning rate = {lr}, tolerance = {tol}')
    sentence = 'Fitting line: '
    for i in range(len(w_new)):
        if len(w_new) - 1 - i == 1:
            sentence += f'{w_new[i]} x + '
        elif len(w_new) - 1 - i == 0:
            sentence += f'{w_new[i]}'
        else:
            sentence += f'{w_new[i]} x^{len(w_new) - 1 - i} + '
    print(sentence)
    print(f'Total error: {error}')

    return w_new

w = steepest_descent(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
draw(w.reshape(-1,1))