from scripts.genetic import GeneticAlgorithmSampler
import numpy as np
def first_method(train, sample_size):
    # last column is the target column
    x_train = train.drop(columns=train.columns[-1]).to_numpy()
    y_train = train[train.columns[-1]].to_numpy()

    # weights are all 1 except 0 for the target column
    weights = [1] * (len(train.columns) - 1) + [0]
    weights = np.array(weights)

    genetic_sampler = GeneticAlgorithmSampler(
        fitness_function='kl_divergence',
        sample_size=sample_size,
        x_train=x_train,
        y_train=y_train,
        population_size=10, 
        max_generations=50,
        crossover_rate=0.8, 
        mutation_rate=0.6,
        mutation_cap=0.2,
        elite_size=1,
        use_same_bandwidth=True,
        verbose=False,
        df_for_ga=train.to_numpy(),
        weights_for_features=weights
    )
    x_train_ws, y_train_ws, history = genetic_sampler.sample()

    return x_train_ws, y_train_ws, history



def second_method(train, sample_size):
    # last column is the target column
    x_train = train.drop(columns=train.columns[-1]).to_numpy()
    y_train = train[train.columns[-1]].to_numpy()

    # weights are all 1 
    weights = [1] * (len(train.columns))
    weights = np.array(weights)

    genetic_sampler = GeneticAlgorithmSampler(
        fitness_function='kl_divergence',
        sample_size=sample_size,
        x_train=x_train,
        y_train=y_train,
        population_size=10, 
        max_generations=50,
        crossover_rate=0.8, 
        mutation_rate=0.6,
        mutation_cap=0.2,
        elite_size=1,
        use_same_bandwidth=True,
        verbose=False,
        df_for_ga=train.to_numpy(),
        weights_for_features=weights
    )
    x_train_ws, y_train_ws, history = genetic_sampler.sample()

    return x_train_ws, y_train_ws, history



def third_method(train, sample_size):
    # last column is the target column
    x_train = train.drop(columns=train.columns[-1]).to_numpy()
    y_train = train[train.columns[-1]].to_numpy()

    weights = [0,0,1,0,0,1,0,0,1,0,0,1,1]
    weights = np.array(weights)

    genetic_sampler = GeneticAlgorithmSampler(
        fitness_function='kl_divergence',
        sample_size=sample_size,
        x_train=x_train,
        y_train=y_train,
        population_size=10, 
        max_generations=50,
        crossover_rate=0.8, 
        mutation_rate=0.6,
        mutation_cap=0.2,
        elite_size=1,
        use_same_bandwidth=True,
        verbose=False,
        df_for_ga=train.to_numpy(),
        weights_for_features=weights
    )
    x_train_ws, y_train_ws, history = genetic_sampler.sample()

    return x_train_ws, y_train_ws, history


def fourth_method(train, sample_size):
    # last column is the target column
    x_train = train.drop(columns=train.columns[-1]).to_numpy()
    y_train = train[train.columns[-1]].to_numpy()

    weights = [0,0,1,0,0,0,1,1,0,0,0,1,1]
    weights = np.array(weights)

    genetic_sampler = GeneticAlgorithmSampler(
        fitness_function='kl_divergence',
        sample_size=sample_size,
        x_train=x_train,
        y_train=y_train,
        population_size=10, 
        max_generations=50,
        crossover_rate=0.8, 
        mutation_rate=0.6,
        mutation_cap=0.2,
        elite_size=1,
        use_same_bandwidth=True,
        verbose=False,
        df_for_ga=train.to_numpy(),
        weights_for_features=weights
    )
    x_train_ws, y_train_ws, history = genetic_sampler.sample()

    return x_train_ws, y_train_ws, history