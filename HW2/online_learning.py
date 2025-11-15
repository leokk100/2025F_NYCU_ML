import numpy as np
import sys

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def ln_factorial(n):
    return np.sum(np.log(np.arange(1, n + 1))) 

def cal_bino(head, trial, MLE):
    return np.exp(ln_factorial(trial) - ln_factorial(trial - head) - ln_factorial(head)) * (MLE ** head) * ((1 - MLE) ** (trial - head)) # see details in HW2_online_learning.pdf

def online_learning(a, b):
    data = load_data('testcase.txt')
    for i in range(len(data)):
        d = data[i].strip() # delete '\n'
        head = d.count('1') # number of heads
        trial = len(d) # number of trials 
        MLE = head / trial # maximum likelihood

        # calculate binomial likelihood: P = {(trial!) / [(trial-head)! * (head)!]} * (MLE^head) * [(1-MLE)^(trial-head)]
        likelihood = cal_bino(head, trial, MLE)    
        print(f'case {i + 1}: {d}')
        print(f'likelihood: {likelihood}')
        print(f'Beta prior:     a = {a}   b = {b}')   
        a_prime = a + head
        b_prime = b + trial - head
        print(f'Beta posterior: a = {a_prime}   b = {b_prime}\n')
        a = a_prime
        b = b_prime
        
online_learning(int(sys.argv[1]), int(sys.argv[2]))