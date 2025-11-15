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
    plt.title(f'LSE method with n = {sys.argv[2]}, λ = {sys.argv[3]}', fontsize=12)

    plt.savefig('plot_LSE.png')

def rLSE(bases, lambdas): # w = ((A_T * A + lambdas * I) ** -1 ) * (A_T * b)
    data = load_data(sys.argv[1])
    A = get_A_matrix(data, bases) # get A
    A_T = transpose(A) # get A_T
    I = np.identity(bases) # get I
    b = data[:,1].reshape(data.shape[0],1) # get b

    # get g = (A_T * A + lambdas * I)
    A_A_T = matmul(A_T, A)
    g = A_A_T + lambdas * I
    
    # get L, U and get inverse matrix
    L, U = LU_decompostion(g)
    inv_g = inverse_matrix(g, L, U)

    # get w
    temp = matmul(inv_g, A_T)
    w = matmul(temp, b)

    # calculate error
    y_pred = matmul(A, w).flatten()
    error = np.sum((y_pred - data[:,1]) ** 2)
    
    # print out equation and total error
    print('LSE:')
    print(f'Case: n = {bases}, λ = {lambdas}')
    sentence = 'Fitting line: '
    for i in range(len(w)):
        if len(w) - 1 - i == 1:
            sentence += f'{w.flatten()[i]} x + '
        elif len(w) - 1 - i == 0:
            sentence += f'{w.flatten()[i]}'
        else:
            sentence += f'{w.flatten()[i]} x^{len(w) - 1 - i} + '
    print(sentence)
    print(f'Total error: {error}')

    return w


w = rLSE(int(sys.argv[2]), int(sys.argv[3]))    
draw(w)
