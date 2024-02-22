import numpy as np
from scipy.stats import entropy
import pandas as pd

def convert_to_numpy_dataset(x_train, x_test, y_train, y_test):
    # convert to numpy arrays
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # make sure the y is one dimensional
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    print("Number of features: ", x_train.shape[1])
    print("Number of training examples: ", x_train.shape[0])
    print("Number of test examples: ", x_test.shape[0])
    return x_train, x_test, y_train, y_test

def make_categorical_into_onehot(X, y, columns_to_onehot):
    X = pd.get_dummies(X, columns=columns_to_onehot)
    return X, y


def sample_data(x_train, y_train, sample_size, seed):
    np.random.seed(seed)
    #print(x_train.shape)
    indices = np.random.choice(x_train.shape[0], sample_size, replace=False)
    smaller_x_train = x_train[indices]
    smaller_y_train = y_train[indices]

    return smaller_x_train, smaller_y_train

def sample_data_improved(x_train, y_train, sample_size):
    population_size = 100
    combined_data = np.column_stack((x_train, y_train))
    first_population = generate_first_population(population_size, x_train, sample_size)
    fitnesses = [fitness(sample, combined_data) for sample in first_population]
    first_population, fitnesses = zip(*sorted(zip(first_population, fitnesses), key=lambda x: x[1]))
    generations = 10
    mutation_chance = 0.1
    best_fitness = [fitnesses[0]]

    for _ in range(generations):
        # Repopulate
        population = first_population[:10] # keep the best 10 samples
        candidates_for_parents = first_population[:50] # only the best 50 samples can be parents
        while len(population) < population_size:
            parent1 = np.random.choice(candidates_for_parents)
            parent2 = np.random.choice(candidates_for_parents)
            son = crossover(parent1, parent2)
            if np.random.rand() < mutation_chance:
                mutate(son)
            population.append(son)

        # Calculate fitness
        fitnesses = [fitness(sample, combined_data) for sample in population]
        population, fitnesses = zip(*sorted(zip(population, fitnesses), key=lambda x: x[1]))
        best_sample = population[0]
        best_fitness.append(fitnesses[0])

    return best_sample[:x_train.shape[0]], best_sample[x_train.shape[0]:], best_fitness


    


def fitness(samples_picked, train_data):
    # make samples_picked a 1D array
    print(samples_picked.shape)
    # make into boolean array
    samples_picked = samples_picked.astype(bool)

    samples_picked = samples_picked.flatten()

    print(samples_picked.shape)


    feature_count = train_data.shape[1]

    
    distance = 0
    for i in range(feature_count):
        # get the column of the feature in the training data
        feature = train_data[:, i]
        # get the column of the feature in the training data
        print(samples_picked)
        sample_feature = train_data[samples_picked, i]
        # calculate the KL divergence
        distance += entropy(feature, sample_feature)

    return distance

def generate_first_population(population_size, train_data, sample_size):
    population = []
    for i in range(population_size):
        # pick sample_size random indices
        indices = np.random.choice(train_data.shape[0], sample_size, replace=False)
        # create a sample with the picked indices
        sample = np.zeros(train_data.shape[0])
        sample[indices] = 1
        population.append(sample)

    return population

def crossover(samples_picked1, samples_picked2):
    max_picked = np.sum(samples_picked1)
    son = np.zeros(samples_picked1.shape[0])
    indexes_where_both_are_1 = np.where(np.logical_and(samples_picked1 == 1, samples_picked2 == 1))[0]
    indexes_where_at_least_one_is_1 = np.where(np.logical_or(samples_picked1 == 1, samples_picked2 == 1))[0]
    son[indexes_where_both_are_1] = 1

    while np.sum(son) < max_picked:
        # pick a random index that is 0 in son and 1 in at least one of the parents
        indexes_where_both_are_0 = np.where(np.logical_and(samples_picked1 == 0, samples_picked2 == 0))[0]
        indexes_where_at_least_one_is_1 = np.where(np.logical_or(samples_picked1 == 1, samples_picked2 == 1))[0]
        indexes_where_both_are_0_and_at_least_one_is_1 = np.intersect1d(indexes_where_both_are_0, indexes_where_at_least_one_is_1)
        random_index = np.random.choice(indexes_where_both_are_0_and_at_least_one_is_1)
        son[random_index] = 1

    return son

def mutate(samples_picked):
    # swaps is between 1 and 5
    swaps = np.random.randint(1, 5)

    for i in range(swaps):
        # pick two random indices to swap
        index1 = np.random.randint(0, samples_picked.shape[0])
        index2 = np.random.randint(0, samples_picked.shape[0])

        # swap the two indices
        samples_picked[index1], samples_picked[index2] = samples_picked[index2], samples_picked[index1]


def get_baseline_guesses(y):
    mean_guess = np.mean(y)
    median_guess = np.median(y)
    guesses = {
        'mean': mean_guess,
        'median': median_guess
    }
    return guesses

def get_baseline_results(y, guesses):
    # for each guess calculate the mse and r2
    results = {}
    for key, value in guesses.items():
        mse = np.mean((y - value) ** 2)
        r2 = 1 - (np.sum((y - value) ** 2) / np.sum((y - np.mean(y)) ** 2))
        results[key] = {
            'MSE': mse,
            'R^2': r2
        }
    return results