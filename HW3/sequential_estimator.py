import random_data_generator as rdg

def sequential_estimator(m, s):
    count = 0
    mean = 0
    M2 = 0  # Sum of squares of differences from the current mean
    previous_mean = float('inf')
    previous_variance = float('inf')
    threshold = 1e-3
    print(f'Data point source function: N({m}, {s})')
    print()

    while True:
        # Welford's online algorithm
        count += 1
        x = rdg.univariate_gaussian_data_generator(m, s)
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2
        variance = M2 / count if count > 1 else 0.0

        print(f'Add data point: {x}')
        print(f'Mean = {mean:<{20}} Variance = {variance}')

        if abs(mean - previous_mean) < threshold and abs(variance - previous_variance) < threshold:
            break

        previous_mean = mean
        previous_variance = variance

if __name__ == "__main__":
    m = 3.0  # mean
    s = 5.0  # variance
    sequential_estimator(m, s)
