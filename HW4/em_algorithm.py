import numpy as np
import struct

def load_image_data(file_path):
    with open(file_path, 'rb') as file:
        magic_number, image_nums, row_nums, col_nums = struct.unpack(">IIII", file.read(16)) # read 16 bytes, with big endian
        images = np.fromfile(file, dtype=np.uint8).reshape(image_nums, row_nums*col_nums) # reshape images to (image_nums, row_nums, col_nums)
    
    return images # shape = (60000, 28*28)

def load_label_data(file_path):
    with open(file_path, 'rb') as file:
        magic_number, label_nums = struct.unpack(">II", file.read(8)) # read 8 bytes, with big endian
        labels = np.fromfile(file, dtype=np.uint8)
    
    return labels

def print_imagination(p):
    for i in range(10):
        print(f'class {i}:')
        for j in range(28):
            for k in range(28):
                if p[i][j * 28 + k] > 0.5:
                    print('1', end=' ')
                else:
                    print('0', end=' ')
            print()
        print()

def print_labels(w, labels, p, iteration):
    predicted_clusters = np.argmax(w, axis=1) # shape: (60000,)
    confusion_matrix = np.zeros((10, 10), dtype=int)
    
    for true_label in range(10):
        for cluster_idx in range(10):
            # Find all images that are BOTH this true_label AND assigned to this cluster_idx
            mask = (labels == true_label) & (predicted_clusters == cluster_idx)
            confusion_matrix[true_label, cluster_idx] = np.sum(mask)

    mapping = np.argmax(confusion_matrix, axis=1) # direction from true label to cluster index
    
    for i in range(10):
        print(f'labeled class {i}:')
        for j in range(28):
            for k in range(28):
                if p[mapping[i]][j * 28 + k] > 0.5:
                    print('1', end=' ')
                else:
                    print('0', end=' ')
            print()
        print()
    
    print('------------------------------------------------------------------------------')
    print()

    print("--- Confusion Matrix (Rows: True Label, Cols: Cluster Index) ---")
    print(confusion_matrix)
    print("------------------------------------------------------------------")

    # Calculate and print sensitivity and specificity for each digit
    correct = 0
    for i in range(10):
        print(f'Confusion Matrix {i}:')
        tp = confusion_matrix[i, mapping[i]] # True Positives
        fp = np.sum(confusion_matrix[:, mapping[i]]) - tp # False Positives
        fn = np.sum(confusion_matrix[i, :]) - tp # False Negatives
        tn = np.sum(confusion_matrix) - (tp + fp + fn) # True Negatives
        correct += tp
        print(f'\t\t\tPredict number {i}\tPredict not number {i}')
        print(f'Is number {i}\t\t\t{tp}\t\t\t{fn}')
        print(f'Isn\'t number {i}\t\t\t{fp}\t\t\t{tn}')
        print()
        print(f'Sensitivity (Successfully predict number {i}): ', tp / (tp + fn))
        print(f'Specificity (Successfully predict not number {i}): ', tn / (tn + fp))
        print()
        print('------------------------------------------------------------------------------')
        print()
    
    print(f'Total iteration to converge: {iteration}')
    print(f'Total error rate: {1 - correct / len(labels)}')

def em_algorithm(images, labels):
    # initialize parameters
    image_length, pixel_length = images.shape # (60000, 784)
    lamb = [0.1 for _ in range(10)] # prior probability for each digit
    p = np.random.rand(10, pixel_length) / 3 # conditional probability for each pixel (0 ~ 0.33)
    p = np.clip(p, 0.01, 0.33) # avoid probabilities below 0.01 or above 0.33

    # make the center of the image more likely to be 1
    for i in range(10):
        center_start = 4
        center_end = center_start + 20
        for j in range(center_start, center_end):
            p[i, j * 28 + center_start: j * 28 + center_end] += 0.2

    # some parameters
    p_prev = np.zeros((10, pixel_length))
    w = np.zeros((image_length, 10))  # shape: (60000, 10)
    iteration = 0
    difference = np.inf
    tolerence = 20

    while iteration < 20 and difference > tolerence:
        # E-step: calculate w
        for i in range(image_length):
        #    for j in range(10):
        #        prob = 1.0
        #        for k in range(pixel_length):
        #            if images[i][k] == 1:
        #                prob *= p[j][k]
        #            else:
        #                prob *= (1 - p[j][k])
        #        w[i][j] = lamb[j] * prob
        #    w[i] /= np.sum(w[i])  # normalize (shape: (10,))
            w[i] = lamb * np.prod(p ** images[i], axis=1) * np.prod((1 - p) ** (1 - images[i]), axis=1)
            w[i] /= np.sum(w[i])

        # M-step: update lamb and p
        for j in range(10):
            N_j = np.sum(w[:, j])
            lamb[j] = N_j / image_length # lamb_new = sigma(w) / image_length
            for k in range(pixel_length):
                numerator = np.sum(w[:, j] * images[:, k])
                p[j][k] = numerator / N_j # p_new = sigma(w * x) / sigma(w)
            p[j] = np.clip(p[j], 0.01, 0.99)  # avoid probabilities of exactly 0 or 1

        # check convergence
        difference = np.sum(np.abs(p - p_prev))
        p_prev = p.copy()
        iteration += 1
        print_imagination(p)
        print(f'No. of iteration: {iteration}, difference: {difference}')
        print()
        print('------------------------------------------------------------------------------')
        print()

    print_labels(w, labels, p, iteration)
    
    
if __name__ == '__main__':
    train_images_path = 'train-images.idx3-ubyte__'
    train_labels_path = 'train-labels.idx1-ubyte__'
    train_images = load_image_data(train_images_path)
    train_labels = load_label_data(train_labels_path)
    train_images_binary = train_images // 128 # Binarize the images
    em_algorithm(train_images_binary, train_labels)