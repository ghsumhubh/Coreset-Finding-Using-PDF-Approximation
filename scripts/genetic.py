import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
# To clear the output of the current cell
# measure time
import time
import concurrent.futures
from scripts.fitness_funcs import fitness_wasserstein_distance, full_train_pdf



def calculate_fitness(sample, used_training, fitness_function, pdfs, is_constant, mins, maxes):
    return fitness_function(sample, used_training, pdfs, is_constant, mins, maxes)

class GeneticAlgorithmSampler():
    def __init__(self,
                  fitness_function,
                    sample_size,
                      x_train, y_train,
                        elite_size = 3,
                          mutation_cap = 5,
                            population_size = 10,
                                mutation_rate = 0.1,
                                    crossover_rate = 0.8,
                                  max_generations = 10,
                                    features_indices_to_drop = set(), verbose = False):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.fitness_function = fitness_function
        self.sample_size = sample_size
        self.x_train = x_train
        self.y_train = y_train
        self.elite_size = elite_size
        self.history = []
        self.mutation_cap = mutation_cap
        self.features_indices_to_drop = features_indices_to_drop
        all_indices = [i for i in range(0,self.x_train.shape[1])]
        self.indices_to_use = list(set(all_indices) - set(self.features_indices_to_drop))
        self.used_training = self.x_train[:, self.indices_to_use]
        self.verbose = verbose
        self.pdfs, self.is_consant, self.mins, self.maxes = full_train_pdf(self.used_training)



    def init_first_population(self):
        self.population = []
        for _ in range(self.population_size):
            indices = np.random.choice(self.used_training.shape[0], self.sample_size, replace=False)
            sample = np.zeros(self.used_training.shape[0])
            sample[indices] = 1
            self.population.append(sample)
        
    def crossover(self, sample1, sample2):
        max_picked = np.sum(sample1)
        offspring = np.zeros(sample1.shape[0])
        indexes_where_both_are_1 = np.where(np.logical_and(sample1 == 1, sample2 == 1))[0]
        offspring[indexes_where_both_are_1] = 1

        while np.sum(offspring) < max_picked:
            # pick a random index that is 0 in son and 1 in at least one of the parents
            indexes_where_offspring_is_0 = np.where(offspring == 0)[0]
            indexes_where_offspring_is_0_and_at_least_one_is_1 = np.where(sample1[indexes_where_offspring_is_0] + sample2[indexes_where_offspring_is_0] > 0)[0]
            random_index = np.random.choice(indexes_where_offspring_is_0_and_at_least_one_is_1)
            offspring[random_index] = 1

        return offspring
    
    def mutate(self, sample):
        # get the number of 1's in the sample
        count = np.sum(sample)
        swaps = np.random.randint(1, self.mutation_cap*count+1)

        for i in range(swaps):
            # pick two random indices to swap, where one is 1 and the other is 0
            index1 = np.random.choice(np.where(sample == 1)[0])
            index2 = np.random.choice(np.where(sample == 0)[0])

            # swap the two indices
            sample[index1], sample[index2] = sample[index2], sample[index1]
    




    def calc_fitnesses(self):
        if self.fitness_function == 'wasserstein_distance':
            with concurrent.futures.ProcessPoolExecutor() as executor:
                self.fitnesses = list(executor.map(calculate_fitness, self.population, [self.used_training]*len(self.population), [fitness_wasserstein_distance]*len(self.population), [self.pdfs]*len(self.population), [self.is_consant]*len(self.population), [self.mins]*len(self.population), [self.maxes]*len(self.population)))
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                self.fitnesses = list(executor.map(calculate_fitness, self.population, [self.used_training]*len(self.population), [fitness_wasserstein_distance]*len(self.population), [self.pdfs]*len(self.population), [self.is_consant]*len(self.population), [self.mins]*len(self.population), [self.maxes]*len(self.population)))

    
        self.population, self.fitnesses = zip(*sorted(zip(self.population, self.fitnesses), key=lambda x: x[1], reverse=True))

    def store_best_fitness(self):
        self.history.append(self.fitnesses[0])


    def repopulate(self):
        # Save elite
        new_population = self.population[:self.elite_size]
        new_population = list(new_population)

        # Calculate selection probabilities based on linear ranking
        # Assuming the population size is the denominator for the probability distribution
        total_ranks = sum(range(1, self.population_size + 1))
        selection_probabilities = [rank/total_ranks for rank in range(self.population_size, 0, -1)]

        # Repopulate the rest
        while len(new_population) < self.population_size:
            # Select parents based on rank-based selection probabilities
            parents_indices = np.random.choice(self.population_size, size=2, p=selection_probabilities, replace=False)
            parent1 = self.population[parents_indices[0]]
            parent2 = self.population[parents_indices[1]]
            if np.random.rand() < self.crossover_rate:
                offspring = self.crossover(parent1, parent2)
            else:
                if np.random.rand() < 0.5:
                    offspring = parent1
                else:
                    offspring = parent2
            if np.random.rand() < self.mutation_rate:
                self.mutate(offspring)
            new_population.append(offspring)
        
        self.population = new_population

    def print_generation_start(self, generation):
        print("\tGeneration {}".format(generation))

    def print_generation_end(self, generation):
        if len(self.history) == 1:
            print("\t\tBest fitness: ", self.history[-1])
        else:
            if self.history[-1] < self.history[-2]:
                print("\t\tBest fitness: ", self.history[-1], " (improved)")

    def print_population(self):
        print("Population: ")
        for sample, fitness in zip(self.population, self.fitnesses):
            indexes = np.where(sample == 1)[0]
            print("Sample: ", indexes)
            print("Fitness: ", fitness)
    def run(self):

        self.print_generation_start(1)
        self.init_first_population()
        self.calc_fitnesses()
        self.store_best_fitness()
        if self.verbose:
            self.print_population()
        self.print_generation_end(1)
    
        for generation in range(self.max_generations-1):
            self.print_generation_start(generation+2)
            self.repopulate()
            self.calc_fitnesses()
            starting_time = time.time()
            self.store_best_fitness()
            if self.verbose:
                self.print_population()
            self.print_generation_end(generation+2)
        
        return self.population[0]
    
    def sample(self):
        best_sample = self.run()
        best_sample = best_sample.astype(bool)
        best_sample = best_sample.flatten()

        sampled_x_train = self.x_train[best_sample, :]
        sampled_y_train = self.y_train[best_sample]
        return sampled_x_train, sampled_y_train, self.history
        


