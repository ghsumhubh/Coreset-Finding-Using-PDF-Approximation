from scripts.xgb_results import xgb_results_regression, get_all_data_and_baseline_results
from scripts.utils import *
import numpy as np
from scripts.genetic import GeneticAlgorithmSampler
from scripts.fitness_funcs import *

def sample_and_get_results(train, test,sample_sizes , redundancy):
    # y is the last column
    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_test = test[:, :-1]
    y_test = test[:, -1]

    all_data_results, baseline_results = get_all_data_and_baseline_results(x_train, x_test, y_train, y_test)


    mse_dict_random = {}
    mse_dict_ws = {}
    mse_dict_ws_short = {}
    mse_dict_kl = {}
    mse_dict_kl_short = {}
    mse_dict_js = {}


   



    for _, sample_size in enumerate(sample_sizes):
        #clear_output() 
        print('\tSample Size:', sample_size)

        mse_dict_random[sample_size] = []
        mse_dict_ws[sample_size] = []
        mse_dict_ws_short[sample_size] = []
        mse_dict_kl[sample_size] = []
        mse_dict_kl_short[sample_size] = []
        mse_dict_js[sample_size] = []

        
        for i in range(redundancy):
            print('\t\tIteration {}/{}'.format(i+1, redundancy))
            np.random.seed(i)
            
            x_train_random, y_train_random = sample_data(x_train, y_train, sample_size)

            genetic_sampler_ws = GeneticAlgorithmSampler(
                fitness_function='wasserstein_distance',
                sample_size=sample_size,
                x_train=x_train,
                y_train=y_train,
                population_size=10, 
                max_generations=30,
                crossover_rate=0.8, 
                mutation_rate=0.6,
                mutation_cap=0.2,
                elite_size=1,
                verbose=False
            )
            x_train_ws, y_train_ws, history = genetic_sampler_ws.sample()

            genetic_sampler_ws_short = GeneticAlgorithmSampler(
                fitness_function='wasserstein_distance',
                sample_size=sample_size,
                x_train=x_train,
                y_train=y_train,
                population_size=10, 
                max_generations=2,
                crossover_rate=0.8, 
                mutation_rate=0.6,
                mutation_cap=0.2,
                elite_size=1,
                verbose=False
            )
            x_train_ws_short, y_train_ws_short, history = genetic_sampler_ws_short.sample()
           
            genetic_sampler_kl = GeneticAlgorithmSampler(
                fitness_function='kl_divergence',
                sample_size=sample_size,
                x_train=x_train,
                y_train=y_train,
                population_size=10, 
                max_generations=30,
                crossover_rate=0.8, 
                mutation_rate=0.6,
                mutation_cap=0.2,
                elite_size=1,
                verbose=False
            )
            x_train_kl, y_train_kl, history = genetic_sampler_kl.sample()

            genetic_sampler_kl_short = GeneticAlgorithmSampler(
                fitness_function='kl_divergence',
                sample_size=sample_size,
                x_train=x_train,
                y_train=y_train,
                population_size=10,
                max_generations=2,
                crossover_rate=0.8,
                mutation_rate=0.6,
                mutation_cap=0.2,
                elite_size=1,
                verbose=False
            )
            x_train_kl_short, y_train_kl_short, history = genetic_sampler_kl_short.sample()

            genetic_sampler_js = GeneticAlgorithmSampler(
                fitness_function='js_divergence',
                sample_size=sample_size,
                x_train=x_train,
                y_train=y_train,
                population_size=10, 
                max_generations=30,
                crossover_rate=0.8, 
                mutation_rate=0.6,
                mutation_cap=0.2,
                elite_size=1,
                verbose=False
            )
            x_train_js, y_train_js, history = genetic_sampler_js.sample()


            
            results_random = xgb_results_regression(x_train_random, x_test, y_train_random, y_test)
            results_ws = xgb_results_regression(x_train_ws, x_test, y_train_ws, y_test)
            results_ws_short = xgb_results_regression(x_train_ws_short, x_test, y_train_ws_short, y_test)
            results_kl = xgb_results_regression(x_train_kl, x_test, y_train_kl, y_test)
            results_kl_short = xgb_results_regression(x_train_kl_short, x_test, y_train_kl_short, y_test)
            results_js = xgb_results_regression(x_train_js, x_test, y_train_js, y_test)

            
            mse_dict_random[sample_size].append(results_random['Testing Metrics']['MSE'])
            mse_dict_ws[sample_size].append(results_ws['Testing Metrics']['MSE'])
            mse_dict_ws_short[sample_size].append(results_ws_short['Testing Metrics']['MSE'])
            mse_dict_kl[sample_size].append(results_kl['Testing Metrics']['MSE'])
            mse_dict_kl_short[sample_size].append(results_kl_short['Testing Metrics']['MSE'])
            mse_dict_js[sample_size].append(results_js['Testing Metrics']['MSE'])

        print('')

    avg_dict ={}
    std_dict = {}

    avg_dict['Random'] = []
    std_dict['Random'] = []
    avg_dict['Wasserstein Distance'] = []
    std_dict['Wasserstein Distance'] = []
    avg_dict['Wasserstein Distance Short'] = []
    std_dict['Wasserstein Distance Short'] = []
    avg_dict['KL Divergence'] = []
    std_dict['KL Divergence'] = []
    avg_dict['KL Divergence Short'] = []
    std_dict['KL Divergence Short'] = []
    avg_dict['JS Divergence'] = []
    std_dict['JS Divergence'] = []


    for sample_size in sample_sizes:
        avg_dict['Random'].append(np.mean(mse_dict_random[sample_size]))
        std_dict['Random'].append(np.std(mse_dict_random[sample_size]))
        avg_dict['Wasserstein Distance'].append(np.mean(mse_dict_ws[sample_size]))
        std_dict['Wasserstein Distance'].append(np.std(mse_dict_ws[sample_size]))
        avg_dict['Wasserstein Distance Short'].append(np.mean(mse_dict_ws_short[sample_size]))
        std_dict['Wasserstein Distance Short'].append(np.std(mse_dict_ws_short[sample_size]))
        avg_dict['KL Divergence'].append(np.mean(mse_dict_kl[sample_size]))
        std_dict['KL Divergence'].append(np.std(mse_dict_kl[sample_size]))
        avg_dict['KL Divergence Short'].append(np.mean(mse_dict_kl_short[sample_size]))
        std_dict['KL Divergence Short'].append(np.std(mse_dict_kl_short[sample_size]))
        avg_dict['JS Divergence'].append(np.mean(mse_dict_js[sample_size]))
        std_dict['JS Divergence'].append(np.std(mse_dict_js[sample_size]))

    mse_dicts = [mse_dict_random, mse_dict_ws, mse_dict_ws_short, mse_dict_kl, mse_dict_kl_short, mse_dict_js]
    labels = ['Random', 'Wasserstein Distance', 'Wasserstein Distance Short', 'KL Divergence','KL Divergence Short' 'JS Divergence']

    return avg_dict, std_dict, mse_dicts, labels, all_data_results, baseline_results