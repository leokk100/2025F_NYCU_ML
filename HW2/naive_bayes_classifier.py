import numpy as np
import struct
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_image_data(file_path):
    with open(file_path, 'rb') as file:
        magic_number, image_nums, row_nums, col_nums = struct.unpack(">IIII", file.read(16)) # read 16 bytes, with big endian
        images = np.fromfile(file, dtype=np.uint8).reshape(image_nums, row_nums, col_nums) # reshape images to (image_nums, row_nums, col_nums)
    
    return images

def load_label_data(file_path):
    with open(file_path, 'rb') as file:
        magic_number, label_nums = struct.unpack(">II", file.read(8)) # read 8 bytes, with big endian
        labels = np.fromfile(file, dtype=np.uint8)
    
    return labels

def calculate_posterior(test_image, prior, likelihood, mode):
    log_posteriors = np.log(prior)

    for label in range(10):
        for row in range(test_image.shape[0]):
            for column in range(test_image.shape[1]):
                if mode == 0:
                    pixel_value = test_image[row][column] // 8
                    log_posteriors[label] += np.log(likelihood[label, row, column, pixel_value])

                elif mode == 1:
                    pixel_value = test_image[row, column]
                    log_posteriors[label] += likelihood[label, row, column, pixel_value]

    log_posteriors_normalized = log_posteriors / np.sum(log_posteriors)  # normalization

    return log_posteriors_normalized

def training(train_images, train_labels, mode):
    if mode == 0: # discrete mode
        likelihood = np.zeros((10, train_images.shape[1], train_images.shape[2], 32)) # shape = (class_nums, row_nums, col_nums, 32)
        prior = np.zeros(10)
        
        for i in tqdm(range(train_images.shape[0])):
            prior[train_labels[i]] += 1
            for j in range(train_images.shape[1]):
                for k in range(train_images.shape[2]):
                    bin_index = train_images[i][j][k] // 8 # Tally the frequency of the values of each pixel into 32 bins
                    likelihood[train_labels[i]][j][k][bin_index] += 1
        
        likelihood = (likelihood + 1) / (prior[:, None, None, None] + 32)  # Laplace smoothing 
        prior /= train_images.shape[0]

        return likelihood, prior

    elif mode == 1: # continuous mode
        mean = np.zeros((10, train_images.shape[1], train_images.shape[2]), dtype=np.float32)
        variance = np.zeros((10, train_images.shape[1], train_images.shape[2]), dtype=np.float32)
        log_likelihood = np.zeros((10, train_images.shape[1], train_images.shape[2], 256))
        prior = np.zeros(10)

        for i in range(train_images.shape[0]):
            prior[train_labels[i]] += 1
            mean[train_labels[i]] += train_images[i]

        for i in range(10):
            mean[i] /= prior[i]

        for i in range(train_images.shape[0]):
            variance[train_labels[i]] += (train_images[i] - mean[train_labels[i]]) ** 2

        for i in range(10):
            variance[i] /= prior[i]

        prior /= train_images.shape[0]

        for i in tqdm(range(10)):
            for row in range(train_images.shape[1]):
                for column in range(train_images.shape[2]):
                    for j in range(256):
                        if variance[i, row, column] == 0:  # To prevent division by zero
                            variance[i, row, column] = 1e3

                        # Calculate the log of the conditional probability of each pixel value given the class.
                        log_likelihood[i, row, column, j] = (-(j - mean[i, row, column])**2) / (2 * variance[i, row, column]) - np.log(np.sqrt(variance[i, row, column])) - 0.5 * np.log(2 * np.pi)

        return log_likelihood, prior

def print_digit_imagination(likelihood, mode):
    print("Imagination of numbers in Bayesian classifier:")

    for label in range(10):
        print(f"{label}:")

        if mode == 0:
            for row in range(28):
                row = ''.join(['1 ' if np.argmax(likelihood[label, row, column]) >= 16 else '0 ' for column in range(28)])
                print(row)

        elif mode == 1:
            for row in range(28):
                row = ''.join(['1 ' if np.argmax(likelihood[label, row, column]) >= 128 else '0 ' for column in range(28)])
                print(row)

        print()

def naive_bayes_classifier():
    # get training and testing data 
    train_images = load_image_data('train-images.idx3-ubyte_')
    train_labels = load_label_data('train-labels.idx1-ubyte_')
    test_images = load_image_data('t10k-images.idx3-ubyte_')
    test_labels = load_label_data('t10k-labels.idx1-ubyte_')

    # get toggle option
    mode = int(input('Toggle option (0 for discrete mode, 1 for continuous mode): '))

    # training
    likelihood, prior = training(train_images, train_labels, mode)

    # testing
    error = 0 

    for i in range(len(test_labels)):
        log_posteriors_normalized = calculate_posterior(test_images[i], prior, likelihood, mode)

        print("Posterior (in log scale):")
        for j in range(10):
            print(f"{j}: {log_posteriors_normalized[j]}")

        predicted_label = np.argmin(log_posteriors_normalized)
        print(f"Prediction: {predicted_label}, Ans: {test_labels[i]}\n")

        if predicted_label != test_labels[i]:
            error += 1

    print_digit_imagination(likelihood, mode=mode)
    error_rate = error / len(test_labels)
    print(f"Error rate: {error_rate}")


naive_bayes_classifier()