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

def LU_decompostion(A):
    # Initialize L and U matrices
    n = len(A)
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    # Compute L and U
    for i in range(n):
        for j in range(i, n): # U
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i, n): # L
            if i == j:
                L[i][i] = 1 # Diagonal as 1
            else:
                L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U

def forward_substitution(L, b): # Ly = b
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    return y


def backward_substitution(U, y): # Ux = y
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

def solve_via_LU(L, U, b): # Ax = b
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x

def inverse_matrix(A, L, U):
    n = A.shape[0]
    inv_A = np.zeros_like(A)
    identity = np.eye(n)

    for i in range(n):
        e_i = identity[:, i]
        inv_A[:, i] = solve_via_LU(L, U, e_i)

    return inv_A

def get_coefficient(A, A_T, w, b):
    # get gradient = 2 * A_T * (A *w - b)
    A_w = matmul(A, w.reshape(-1, 1))
    gradient = 2 * matmul(A_T, A_w - b)

    # get hessian = 2 * A_T * A
    hessian = 2 * matmul(A_T, A)

    # get coefficient = gradient * (hessian ^ -1)
    L, U = LU_decompostion(hessian)
    hessian_inv = inverse_matrix(hessian, L, U)
    coefficient = matmul(hessian_inv, gradient)

    return coefficient.flatten()

def draw(w):
    data = load_data(sys.argv[1])
    bases = int(sys.argv[2])

    # draw the real scattering data
    plt.scatter(data[:,0], data[:,1], label='Data') 

    # draw the line created by rLSE method
    x_line = np.linspace(min(data[:,0]) - 1, max(data[:,0]) + 1, 100)
    x_matrix = np.zeros((len(x_line), 2))
    x_matrix[:,0] = x_line
    y_line = matmul(get_A_matrix(x_matrix, bases), w).flatten() # (n, 1) -> (n, )
    plt.plot(x_line, y_line, color='red', label='Fitted curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Newton\'s method with n = {sys.argv[2]}', fontsize=12)

    plt.savefig('plot_Newton.png')

def Newton(bases, max_iter, tol):
    data = load_data(sys.argv[1])
    A = get_A_matrix(data, bases) # get A
    A_T = transpose(A) # get A_T
    b = data[:,1].reshape(data.shape[0],1) # get b

    # the iteration to find the min w
    w_old = np.zeros(bases) # w_init is initialized as a (bases, ) vector
    w_new = np.zeros(bases) # w_new is initialized as a (bases, ) vector
   
    for i in range(max_iter):
        coefficient = get_coefficient(A, A_T, w_old, b)
        w_new = w_old - coefficient

        if np.linalg.norm(abs(w_new - w_old)) < tol:
            break

        w_old = w_new

    # calculate error
    y_pred = matmul(A, w_new.reshape(-1,1)).flatten()
    error = np.sum((y_pred - data[:,1]) ** 2)
    
    # print out equation and total error
    print('Newton:')
    print(f'Case: n = {bases}, max iteration = {max_iter}, tolerance = {tol}')
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

w = Newton(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
draw(w.reshape(-1,1))
    