import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
# To clear the output of the current cell
# measure time
import time
import concurrent.futures
from scripts.fitness_funcs import fitness_wasserstein_distance, full_train_pdf, fitness_kl_divergence, fitness_js_divergence



def calculate_fitness(sample, used_training, fitness_function, pdfs, is_constant, mins, maxes, kde_bandwidth=None):
    return fitness_function(sample, used_training, pdfs, is_constant, mins, maxes, kde_bandwidth)

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
                                    features_indices_to_drop = set(),
                                      verbose = False,
                                      use_same_bandwidth = False):
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
        self.kde_bandwidth = self.get_kde_bandwidth()
        self.use_same_bandwidth = use_same_bandwidth
        
        if self.use_same_bandwidth == False:
            self.pdfs, self.is_consant, self.mins, self.maxes = full_train_pdf(self.used_training)
        else:
            self.pdfs, self.is_consant, self.mins, self.maxes = full_train_pdf(self.used_training, self.kde_bandwidth)


    def get_kde_bandwidth(self):
        bandwidths = []
        for i in range(self.used_training.shape[1]):
            bandwidths.append(self.silverman_bandwidth(self.used_training[:, i]))
        return bandwidths

    def silverman_bandwidth(self, data):
        # Calculate standard deviation of the data
        std_dev = np.std(data)
        # Number of data points
        n = self.sample_size
        # Calculate bandwidth using Silverman's rule of thumb
        bandwidth = (4 * std_dev**5 / (3 * n))**(1/5)
        return bandwidth



    def init_first_population(self):
        self.population = []
        for _ in range(self.population_size):
            indices = np.random.choice(self.used_training.shape[0], self.sample_size, replace=False)
            sample = np.zeros(self.used_training.shape[0])
            sample[indices] = 1
            self.population.append(sample)
        
    def crossover(self, sample1, sample2):
        offspring = np.zeros(sample1.shape[0], dtype=int)
        # Indices where both parents have 1
        indexes_where_both_are_1 = np.where(np.logical_and(sample1 == 1, sample2 == 1))[0]
        offspring[indexes_where_both_are_1] = 1

        max_picked = np.sum(sample1)
        # Calculate remaining 1s to add
        ones_needed = int(max_picked - np.sum(offspring))
        # Indices where at least one parent has 1 and offspring is 0
        valid_indices = np.where((sample1 + sample2 > 0) & (offspring == 0))[0]

        #print("Ones needed: ", ones_needed)
        #print("Valid indices: ", valid_indices) 
        
        if ones_needed > 0 and valid_indices.size > 0:
            # If more 1s needed and there are valid indices, randomly pick and update
            chosen_indices = np.random.choice(valid_indices, min(ones_needed, valid_indices.size), replace=False)
            offspring[chosen_indices] = 1

        return offspring
    
    def mutate(self, sample):
        # Number of 1's to flip to 0's
        ones_indices = np.where(sample == 1)[0]
        count = ones_indices.size
        ones_to_flip = np.random.randint(1, int(self.mutation_cap*count + 1))
        
        # Flip the chosen 1's to 0's
        flip_ones_indices = np.random.choice(ones_indices, ones_to_flip, replace=False)
        sample[flip_ones_indices] = 0
        
        # Decide the number of 0's to flip to 1's, can be the same or different
        # For this example, let's flip the same amount: ones_to_flip
        zeros_indices = np.where(sample == 0)[0]
        flip_zeros_indices = np.random.choice(zeros_indices, ones_to_flip, replace=False)
        sample[flip_zeros_indices] = 1
        
        return sample
    




    def calc_fitnesses(self):
        if self.use_same_bandwidth == False:
            if self.fitness_function == 'wasserstein_distance':
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    self.fitnesses = list(executor.map(calculate_fitness, self.population, [self.used_training]*len(self.population), [fitness_wasserstein_distance]*len(self.population), [self.pdfs]*len(self.population), [self.is_consant]*len(self.population), [self.mins]*len(self.population), [self.maxes]*len(self.population)))
            elif self.fitness_function == 'kl_divergence':
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    self.fitnesses = list(executor.map(calculate_fitness, self.population, [self.used_training]*len(self.population), [fitness_kl_divergence]*len(self.population), [self.pdfs]*len(self.population), [self.is_consant]*len(self.population), [self.mins]*len(self.population), [self.maxes]*len(self.population)))
            elif self.fitness_function == 'js_divergence':
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    self.fitnesses = list(executor.map(calculate_fitness, self.population, [self.used_training]*len(self.population), [fitness_js_divergence]*len(self.population), [self.pdfs]*len(self.population), [self.is_consant]*len(self.population), [self.mins]*len(self.population), [self.maxes]*len(self.population)))
            else:
                raise ValueError("Invalid fitness function")
        else:
            if self.fitness_function == 'wasserstein_distance':
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    self.fitnesses = list(executor.map(calculate_fitness, self.population, [self.used_training]*len(self.population), [fitness_wasserstein_distance]*len(self.population), [self.pdfs]*len(self.population), [self.is_consant]*len(self.population), [self.mins]*len(self.population), [self.maxes]*len(self.population), [self.kde_bandwidth]*len(self.population)))
            elif self.fitness_function == 'kl_divergence':
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    self.fitnesses = list(executor.map(calculate_fitness, self.population, [self.used_training]*len(self.population), [fitness_kl_divergence]*len(self.population), [self.pdfs]*len(self.population), [self.is_consant]*len(self.population), [self.mins]*len(self.population), [self.maxes]*len(self.population), [self.kde_bandwidth]*len(self.population)))
            elif self.fitness_function == 'js_divergence':
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    self.fitnesses = list(executor.map(calculate_fitness, self.population, [self.used_training]*len(self.population), [fitness_js_divergence]*len(self.population), [self.pdfs]*len(self.population), [self.is_consant]*len(self.population), [self.mins]*len(self.population), [self.maxes]*len(self.population), [self.kde_bandwidth]*len(self.population)))
            else:
                raise ValueError("Invalid fitness function")
    
        self.population, self.fitnesses = zip(*sorted(zip(self.population, self.fitnesses), key=lambda x: x[1], reverse=True))

    def store_best_fitness(self):
        self.history.append(self.fitnesses[0])


    def repopulate(self):
        # Save elite
        new_population = self.population[:self.elite_size]
        # print the indexes of elite[0] that are 1
        #print(np.where(new_population[0] == 1)[0])
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
                    offspring = parent1.copy()
                else:
                    offspring = parent2.copy()
            if np.random.rand() < self.mutation_rate:
                offspring = self.mutate(offspring)
            new_population.append(offspring)
        
        self.population = new_population

    def print_generation_start(self, generation):
        print("\tGeneration {}".format(generation))

    def print_generation_end(self, generation):
        if len(self.history) == 1:
            print("\t\tBest fitness: ", self.history[-1])
        else:
            if self.history[-1] > self.history[-2]:
                print("\t\tBest fitness: ", self.history[-1], " (improved)")
            elif self.history[-1] < self.history[-2]:
                print("\t\tBest fitness: ", self.history[-1], " (regressed)")

    def print_population(self):
        print("Population: ")
        for sample, fitness in zip(self.population, self.fitnesses):
            indexes = np.where(sample == 1)[0]
            print("Sample: ", indexes)
            print("Fitness: ", fitness)
    def run(self):

        #self.print_generation_start(1)
        self.init_first_population()
        self.calc_fitnesses()
        self.store_best_fitness()
        if self.verbose:
            self.print_population()
        #self.print_generation_end(1)
    
        for generation in range(self.max_generations-1):
            #self.print_generation_start(generation+2)
            self.repopulate()
            self.calc_fitnesses()
            self.store_best_fitness()
            if self.verbose:
                self.print_population()
            #self.print_generation_end(generation+2)
        
        return self.population[0]
    
    def sample(self):
        best_sample = self.run()
        best_sample = best_sample.astype(bool)
        best_sample = best_sample.flatten()

        sampled_x_train = self.x_train[best_sample, :]
        sampled_y_train = self.y_train[best_sample]
        return sampled_x_train, sampled_y_train, self.history
        


