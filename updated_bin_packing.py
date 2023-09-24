import time
import random
from random import randint
import numpy as np
import statistics
from Struct2 import Struct_Binpacking
from Struct2 import Bin
import statistics
import math
import matplotlib.pyplot as plt
MAX = 0.7
MIN = 0.1
SHARING_RADIUS = 120

# class for bin packing including all the information needed abou mutation rate and type ,crossover type ...
class BinPacking:
    def __init__(self, capacity, POPSIZE, items, MUTATION, mutation_type, crossover_type, selection,aging, elitism_size):
        self.ELITISM_SIZE = elitism_size * POPSIZE
        self.MUTATION_RATE = MUTATION
        self.capacity = capacity
        self.items = items
        self.num_items = len(items)
        self.POPSIZE = POPSIZE
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
        self.selection = selection
        self.aging = aging
        self.population = []
        self.new_population = []

    # for each item put it in a random bin
    def init_population(self):
        for i in range(self.POPSIZE):
            individual = Struct_Binpacking(self.num_items)
            for j in range(self.num_items):
                individual.list_bins.append(random.randint(0, self.num_items - 1))
            self.population.append(individual)

    # combination between first fit and best fit heuristics
    def fitness(self):

        for i in range(self.POPSIZE):

            sorted_items = sorted(self.items, reverse=True)
            non_empty_bins = [Bin(self.capacity) for j in range(self.num_items)]


            for item in sorted_items:
                for bin in non_empty_bins:
                    if bin.append_item(item):
                        break

                else:
                    best_bin = None
                    best_left_capacity = self.capacity
                    for bin in non_empty_bins:
                        if self.capacity - bin.capacity_left >= item and bin.capacity_left < best_left_capacity:
                            best_bin = bin
                            best_left_capacity = bin.capacity_left


                    if best_bin is not None:
                        best_bin.append_item(item)
                    else:
                        new_bin = Bin(self.capacity)
                        new_bin.append_item(item)
                        non_empty_bins.append(new_bin)

            # calculate space_left in each bin and the number of no empty bins
            space_left = sum([bin.capacity_left for bin in non_empty_bins])
            num_non_empty_bins = len([bin for bin in non_empty_bins if bin.capacity_left < self.capacity])

            self.population[i].fitness = space_left + num_non_empty_bins * self.capacity
            self.population[i].num_bins = num_non_empty_bins

    def fitness_first_fit(self):
        for i in range(self.POPSIZE):
            sorted_items = sorted(self.items, reverse=True)
            non_empty_bins = [Bin(self.capacity) for j in range(self.num_items)]

            for item in sorted_items:
                for bin in non_empty_bins:
                    if bin.append_item(item):
                        break

                else:
                    new_bin = Bin(self.capacity)
                    new_bin.append_item(item)
                    non_empty_bins.append(new_bin)

            # Calculate space_left in each bin and the number of non-empty bins
            space_left = sum([bin.capacity_left for bin in non_empty_bins])
            num_non_empty_bins = len([bin for bin in non_empty_bins if bin.capacity_left < self.capacity])

            self.population[i].fitness = space_left + num_non_empty_bins * self.capacity
            self.population[i].num_bins = num_non_empty_bins


    def genetic_algorithm(self,generation,max_generation,niching):
        self.elitism()
        if niching ==1:
            similarity_matrix = build_similarity_matrix()
            fitnesses = niching_with_shared_fitness(population, similarity_matrix, fitnesses)
        if niching == 2:
            silhouette_score = np.zeros(K - 1)
            for i in range(2, K + 1):
                clusters = Kmean(population, i)
                silhouette_score[i - 2] = Silhouette(population, clusters,i)
            optimal_k = np.argmax(silhouette_score) + 2
            clusters = Kmean(population, optimal_k)

        while len(self.new_population) < self.POPSIZE:
            parent1_index = randint(0, self.POPSIZE // 2)
            parent2_index = randint(0, self.POPSIZE // 2)
            parent1_Bins = self.population[parent1_index].list_bins
            parent2_Bins = self.population[parent2_index].list_bins

            offspring1 = [-1] * self.num_items
            offspring2 = [-1] * self.num_items

            child1 = Struct_Binpacking(self.num_items)
            child2 = Struct_Binpacking(self.num_items)

            #RWS parent seletion
            if self.selection == "1":
                parent1_Bins = self.RWS()
                parent2_Bins = self.RWS()
            #SUS parent selection
            elif self.selection == "2":
                selected_parents = self.SUS(self.num_items)
                parent1_Bins = random.choice(selected_parents)
                parent2_Bins = random.choice(selected_parents)
            # Tournament_selection
            elif self.selection == "3":
                k = random.randint(1, len(self.population) - 1)
                parent1_Bins = self.Tournament_selection(k)
                k = random.randint(1, len(self.population) - 1)
                parent2_Bins = self.Tournament_selection(k)


            # single crossover type
            if crossover_type == "1":
                child1.list_bins = self.single_crossover(parent1_Bins, parent2_Bins)
                child2.list_bins = self.single_crossover(parent1_Bins, parent2_Bins)
            # two crossover type
            elif crossover_type == "2":
                child1.list_bins = self.two_crossover(parent1_Bins, parent2_Bins)
                child2.list_bins = self.two_crossover(parent1_Bins, parent2_Bins)
            # uniform crossover type
            else:
                child1.list_bins = [parent1_Bins[i] if random.random() < 0.5 else parent2_Bins[i] for i in
                                    range(self.num_items)]
                child2.list_bins = [parent1_Bins[i] if random.random() < 0.5 else parent2_Bins[i] for i in
                                    range(self.num_items)]

            # mutation type = inversion mutation
            if self.mutation_type == "1":
                child1.list_bins = self.inversion_mutation(child1.list_bins)
                child2.list_bins = self.inversion_mutation(child2.list_bins)
            # mutation type= scramble mutation, update NQueens of each child
            if self.mutation_type == "2":
                child1.list_bins = self.scramble_mutation(child1.list_bins)
                child2.list_bins = self.scramble_mutation(child2.list_bins)
            # non_unifrom mutation -linear
            if self.mutation_type == "3":
                child1.list_bins = self.non_uniform_linear(child1.list_bins, generation, max_generation)
                child2.list_bins = self.non_uniform_linear(child2.list_bins, generation, max_generation)
            if self.mutation_type =="4":
                child1.list_bins = self.non_unifrom_non_linear(child1.list_bins, generation, 0.25)
                child2.list_bins = self.non_unifrom_non_linear(child2.list_bins, generation, 0.25)
            #THM mutation
            if self.mutation_type == "5":
                fitness_avg,_ = self.fitness_avg_std()
                fitnesses = [individual.fitness for individual in self.population]
                max_fitness = max(fitnesses)
                trigger_fitness = 18300
                trigger_avg = 18300
                child1.list_bins = self.THM(child1.list_bins,trigger_fitness ,trigger_avg , generation ,max_fitness ,fitness_avg)
                child2.list_bins = self.THM(child2.list_bins, trigger_fitness, trigger_avg, generation, max_fitness, fitness_avg)
            #adaptive mutation
            if mutation_type == "6":
                max_fitness = 18300
                self.population.append(child1)
                fitnesses = [individual.fitness for individual in self.population]
                child1.list_bins = self.adaptive_mutation(child1.list_bins,max_fitness,child1.fitness)
                self.population.remove(child1)
                self.population.append(child2)
                fitnesses = [individual.fitness for individual in self.population]
                child2.list_bins = self.adaptive_mutation(child2.list_bins,max_fitness,child2.fitness)
                self.population.remove(child2)
            #self adaptive mutation
            if mutation_type =="7":
                fitnesses = [individual.fitness for individual in self.population]
                self.population.append(child1)
                fitnesses = [individual.fitness for individual in self.population]
                child1.list_bins = self.self_adaptive_mutation(child1.list_bins,fitnesses)
                self.population.remove(child1)
                fitnesses.remove(child1.fitness)
                self.population.append(child2)
                fitnesses = [individual.fitness for individual in self.population]
                child2.list_bins = self.self_adaptive_mutation(child2.list_bins,fitnesses)
                self.population.remove(child2)
                fitnesses.remove(child2.fitness)

            if niching == 3:
                crowding(child, fitness_func(child), index1, index2, population, fitnesses)
            # add both new childs to new_population
            self.new_population.append(child1)
            self.new_population.append(child2)

    # 1.find random point in parent1
    # 2.build child:
    # 2.1: 0->random_index-1=parent1 ,random_index->last index =parent2
    def single_crossover(self, parent1, parent2):
        random_index = random.randint(1, len(parent1) - 1)
        child = parent1[:random_index] + parent2[random_index:]
        return child

    # 1.find two random points in parent1
    # 2.sort both points
    # 3.build child:
    # 3.1: 0->random_index-1=parent1, random_index->random_index2=parent2 , random_index2->last index =parent1
    def two_crossover(self, parent1, parent2):
        random_index1 = random.randint(0, len(parent1) - 1)
        random_index2 = random.randint(0, len(parent1) - 1)
        if random_index1 > random_index2:
            random_index1, random_index2 = random_index2, random_index1
        child = parent1[:random_index1] + parent2[random_index1:random_index2] + parent1[random_index2:]
        return child

    def sum_fitness(self):
        total_fitness = 0
        for i in range(len(self.population)):
            total_fitness += self.population[i].fitness
        return total_fitness

    def RWS(self):
        total_fitness = self.sum_fitness()
        pop_probability = []  # probablity list

        if total_fitness == 0:
            individual = random.choice(self.population)
            return individual.list_bins
        pop_fitness = [individual.fitness for individual in self.population]

        scaled_pop_fitness = pop_fitness
        # find the probabilty of each individual in population by individual's fitness/total_fitness
        pop_probability = [fitness / total_fitness for fitness in scaled_pop_fitness]
        # create slices :each slice corresponds to an individual's probability of selection(like example we did in class)
        slices = [sum(pop_probability[:i + 1]) for i in range(len(pop_probability))]
        # spin the roulette by generating random number between 0 and 1
        random_spin = random.random()
        # if random number falls into slice[j] -> return individual in population[j]
        for j in range(len(slices)):
            if random_spin <= slices[j]:
                return self.population[j].list_bins

        individual = random.choice(self.population)
        return individual.list_bins

    def SUS(self, num):
        selected_parents = []
        pop_fitness = [individual.fitness for individual in self.population]

        scaled_pop_fitness = pop_fitness

        total_fitness = self.sum_fitness()
        # calculate equal step size
        step_size = total_fitness / num
        # choose random start to begin
        random_start = random.uniform(0, step_size)

        # walk equal step size and add to selected_parents according to indvidual's fitness
        for j in range(num):
            pointer = random_start
            fitness_sum = 0
            for i, individual in enumerate(self.population):
                fitness_sum += scaled_pop_fitness[i]
                if fitness_sum >= pointer:
                    selected_parents.append(individual.list_bins)
                    break
            random_start += step_size

        return selected_parents

    # scaling function(given)
    def winsorize(self, data, percentile=5):
        lower_bound = np.percentile(data, percentile)
        upper_bound = np.percentile(data, 100 - percentile)
        data = np.where(data < lower_bound, lower_bound, data)
        data = np.where(data > upper_bound, upper_bound, data)
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std
        return data

    def Tournament_selection(self, k):
        selected_k = random.sample(self.population, k)
        k_fitnesses = [individual.fitness for individual in selected_k]
        best_individual = selected_k[0].list_bins
        best_individual_fitness = k_fitnesses[0]

        for i in range(len(selected_k)):
            if k_fitnesses[i] < best_individual_fitness:
                best_individual_fitness = k_fitnesses[i]
                best_individual = selected_k[i].list_bins

        return best_individual

    # pick 3 positions randomly and then inverse between two indexes and then change place according to third index
    def inversion_mutation(self, individual):
        if random.random() < self.MUTATION_RATE:
            pos1, pos2 = sorted([randint(0, self.num_items - 1), randint(0, self.num_items - 1)])
            mov_pos = randint(pos2, self.num_items - 1)
            individual = individual[0:pos1] + individual[pos2:mov_pos] + individual[pos1:pos2][::-1] + individual[
                                                                                                       mov_pos:]
        return individual

    # pick both indexes randomly and then shuffle individual[first index:second index]
    def scramble_mutation(self, individual):
        if random.random() < self.MUTATION_RATE:
            pos1, pos2 = random.sample(range(len(individual)), 2)
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            sublist = individual[pos1:pos2]
            random.shuffle(sublist)
            individual[pos1:pos2] = sublist
        return individual

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def fitness_sort(self, x):
        return x.get_fitness()

    def elitism(self):
        for i in range(int(self.ELITISM_SIZE)):
            self.new_population.append(self.population[i])

    def swap(self):
        self.population = self.new_population
        self.new_population = []

    def print_best_solution(self):
        self.sort_by_fitness()
        print("Best Solution for Bin Packing : ", end=" ")
        print("number of bins: ", self.population[0].num_bins, end=" ")
        print(" ,and for fitness :", self.population[0].fitness)

    # parameters 5+6
    def fitness_avg_std(self):
        fitnesses = [individual.fitness for individual in self.population]
        fitness_average = sum(fitnesses) / len(fitnesses)
        fitness_std = statistics.stdev(fitnesses)
        return fitness_average, fitness_std

    # Top-Average Selection Probability Ratio
    # calculated by dividing the probability of selecting the best individual by the average probability of selecting an individual from the population.
    def Top_average(self):
        fitnesses = sum([individual.fitness for individual in self.population])
        probabilities = [fitness / fitnesses for fitness in [gene.fitness for gene in self.population]]
        avg_probability = sum(probabilities) / len(probabilities)
        top_probability = max(probabilities)
        top_avg = top_probability / avg_probability
        return top_avg

    # count how many permutation are different
    def count_unique_bins(self):
        unique_permutations = set()
        for individual in self.population:
            permutation = tuple(individual.list_bins)
            unique_permutations.add(permutation)
        return len(unique_permutations)

    # calcultae difeerence between two strings
    def bins_difference(self, individual1, individual2):
        count = 0
        for string1, string2 in zip(individual1, individual2):
            if string1 != string2:
                count += 1
        return count

    # sum all the differences between all string = total
    def sum_bins_difference(self):
        sum_difference = 0
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                individual1 = self.population[i]
                individual2 = self.population[j]
                diff = self.bins_difference(individual1.list_bins, individual2.list_bins)
                sum_difference += diff
        return sum_difference

    # LAB2 EXTRA: Non-unifrom mutation - linear
    def non_uniform_linear(self, child, generation, total_generations):
        mutate_child = child
        mutation_rate = MAX - (MAX - MIN) * (generation / total_generations)
        if random.random() < mutation_rate:
            mutate_child = self.inversion_mutation(mutate_child)
            # i, j = random.sample(range(len(child)), 2)
            # mutate_child[i], mutate_child[j] = mutate_child[j], mutate_child[i]
            return mutate_child
        return child
 # LAB2 EXTRA: Non-unifrom mutation - non_linear
    def non_unifrom_non_linear(self,child, generation, r):
        mutate_child = list(child)
        if generation != 0:
            mutation_rate = (2 * (self.MUTATION_RATE ** 2) * math.exp(generation * r)) / (
                        generation + generation * math.exp(generation * r))
        else:
            mutation_rate = self.MUTATION_RATE
        if random.random() < mutation_rate:
            mutate_child = self.inversion_mutation(mutate_child)
            # i, j = random.sample(range(len(child)), 2)
            # mutate_child[i], mutate_child[j] = mutate_child[j], mutate_child[i]
            return mutate_child
        return child

    def THM(self,child, trigger_fitness, trigger_avg, trigger_generation, fitness, avg):
        mutate_child = list(child)
        if fitness < trigger_fitness or avg < trigger_avg:
            mutation_rate = self.MUTATION_RATE * 2
        else:
            mutation_rate = self.MUTATION_RATE
        if random.random() < mutation_rate:
            mutate_child = self.inversion_mutation(mutate_child)
            # i, j = random.sample(range(len(child)), 2)
            # mutate_child[i], mutate_child[j] = mutate_child[j], mutate_child[i]
            return mutate_child
        return child

    def adaptive_mutation(self,child, max_fitness,child_fitness):
        mutate_child = list(child)
        mutation_rate = self.MUTATION_RATE * (1 - child_fitness / max_fitness)
        if random.random() < mutation_rate:
            mutate_child = self.inversion_mutation(mutate_child)
            # i, j = random.sample(range(len(child)), 2)
            # mutate_child[i], mutate_child[j] = mutate_child[j], mutate_child[i]
            return mutate_child
        return child

    def relative_fitness(self,fitnesses):
        # Compute population mean fitness
        mean_fitness = statistics.mean(fitnesses)
        # Compute relative fitness for each individual
        relative_fitness = [f / mean_fitness for f in fitnesses]
        # Normalize relative fitness
        std_dev = statistics.stdev(relative_fitness)
        if std_dev != 0:
            normalized_relative_fitness = [r / std_dev for r in relative_fitness]
        else:
            std_dev += 2.5
            normalized_relative_fitness = [r / std_dev for r in relative_fitness]
        return normalized_relative_fitness

    def self_adaptive_mutation(self,child, fitnesses):
        mutate_child = list(child)
        relative_fitnesses = self.relative_fitness(fitnesses)
        mutation_rate = self.MUTATION_RATE * (1 - relative_fitnesses[-1])
        if random.random() < mutation_rate:
            mutate_child = self.inversion_mutation(mutate_child)
            # i, j = random.sample(range(len(child)), 2)
            # mutate_child[i], mutate_child[j] = mutate_child[j], mutate_child[i]
            return mutate_child
        return child

    def mutation(self,child):
        mutate_child = list(child)
        if random.random() < self.MUTATION_RATE:
            mutate_child = self.inversion_mutation(mutate_child)
            # i, j = random.sample(range(len(child)), 2)
            # mutate_child[i], mutate_child[j] = mutate_child[j], mutate_child[i]
            return mutate_child
        return child

    # LAB2-EXTRA:function for niching
    def build_similarity_matrix(self):
        similarity_matrix = np.zeros(self.POPSIZE, self.POPSIZE)
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix[i])):
                similarity_matrix[i][j] = self.bins_difference(self.population[i], self.population[j][0])
        return similarity_matrix

    # LAB2-EXTRA:function for niching
    def niching_with_shared_fitness(self, similarity_matrix, fitnesses):
        niches = []
        for i in range(self.POPSIZE):
            flag_found = 0
            for niche in niches:
                for j in range(len(niche)):
                    matrix = similarity_matrix[i][j]
                    if matrix < SHARING_RADIUS:
                        niche.append(self.population[i])
                        flag_found = 1
                        break
                if flag_found == 1: break
            if flag_found == 0:
                niches.append([self.population[i]])
        shared_fitness = calculate_shared_fitness(niches,similarity_matrix,fitnesses)
        return shared_fitness

    # LAB2-EXTRA:function for niching
    def calculate_shared_fitness(self,niches,similarity_matrix,fitnesses):
        shared_fitness = []
        for nich in niches:
            for i, individual in enumerate(nich):
                sh = 0
                for j in range(len(nich)):
                    if i != j:
                        sh += similarity_matrix[i][j]
                shared_fitness.append(fitnesses[i]/sh)
        return shared_fitness

    def crowding(self,child, child_fitness, parent1_index, parent2_index, fitnesses):
        F1 = fitnesses[parent1_index]
        F2 = fitnesses[parent2_index]
        delta_P1_child = abs(F1 - child_fitness)
        delta_P2_child = abs(F2 - child_fitness)
        T = 400
        replace_prob1 = math.exp(-delta_P1_child / T) / (1 + math.exp(-delta_P1_child / T))
        if random.random() < replace_prob1:
            self.population[parent1_index] = child

        replace_prob2 = math.exp(-delta_P2_child / T) / (1 + math.exp(-delta_P2_child / T))
        if random.random() < replace_prob2:
            self.population[parent2_index] = child

    def Kmean(self, K):
        n = self.POPSIZE
        # this array is to save the distance between individual and all centroids
        distances = np.zeros(K)
        # this array is to save for each individual the clusters it belogns to
        clusters = np.zeros(n)
        # choose random centroids as an intial state  for all k clusters
        centroids = [''.join(random.choices(string.printable, k=13)) for i in range(K)]
        # array to check when there is no change in the clusters to stop the algorithm
        last_cluster = np.zeros(n)
        while 1:
            for i in range(n):
                for j in range(K):
                    # claculate distance between each individual and all centroids
                    distances[j] = self.bins_difference(self.population[i], self.centroids[j])
                # put individual i in cluster that has the minimum distance with its centroid
                clusters[i] = np.argmin(distances)
            # update centroid
            sum = [0] * n
            for i in range(K):
                count = 0
                for j in range(n):
                    if clusters[j] == i:
                        count += 1
                        for y in range(13):
                            str = self.population[j]
                            ascii_val = ord(str[y])
                            sum[y] += ascii_val
                if count != 0:
                    sum = [int(x / count) for x in sum]
                char_list = []
                for ascii_value in sum:
                    if ascii_value != 0:
                        my_char = chr(ascii_value)
                        char_list.append(my_char)
            str_ = ''.join(char_list)
            centroids[i] = str_

            if np.array_equal(clusters, last_cluster): break

            last_cluster = clusters
        return clusters

    def Silhouette(self, clusters, K):
        n = self.POPSIZE

        score_cluster = np.zeros(K)
        sum_cluster = 0

        for i in range(n):
            cluster_indices = np.where(clusters == i)[0]
            a = np.zeros(len(cluster_indices))
            for j in range(len(cluster_indices)):
                sum = 0
                for y in range(len(cluster_indices)):
                    sum += hamming_distance(self.population[cluster_indices[j]], self.population[cluster_indices[y]])
                if len(cluster_indices) > 1:
                    a[j] = sum / (len(cluster_indices) - 1)

            b = np.zeros(len(cluster_indices))
            temp = np.zeros(K)
            for j in range(len(cluster_indices)):  # for every point in cluster i
                for y in range(K):  # sum difference between point j in cluster i and all other clusters
                    sum = 0
                    if y != i:
                        non_cluster_indices = np.where(clusters == y)[0]
                        for z in range(len(non_cluster_indices)):
                            sum += hamming_distance(self.population[cluster_indices[j]],
                                                    self.population[non_cluster_indices[z]])
                        if len(non_cluster_indices) > 0:
                            temp[y] = sum / (
                                len(non_cluster_indices))  # b3d ma qarent point balb cluster i m3 kul points cluster m3yn y
                temp[i] = 1000000000
                b[j] = min(temp)
                if a[j] == 0 and b[j] == 0:
                    score_j = 0
                else:
                    score_j = (b[j] - a[j]) / max(a[j], b[j])
                    sum_cluster += score_j

            if len(cluster_indices) > 0:
                # print(sum_cluster/len(cluster_indices))
                score_cluster[i] = sum_cluster / len(cluster_indices)

        final_score = np.sum(score_cluster) / K
        return final_score

    def create_histogram(self,string_diff, generations):
        plt.plot(generations, string_diff)
        plt.xlabel('Generation')
        plt.ylabel('difference/distance')
        plt.show()


if __name__ == '__main__':

    print("Welcome to BinPacking Solution")
    population_size = input("Please enter population size(must be integer): ")
    max_generation = input("Please enter number of generations(must be integer): ")
    capacity = input("Please enter capacity of bins(must be integer): ")
    items_weights = input("Enter the weights of the items (separated by spaces): ")
    items_weights = [int(w) for w in items_weights.split()]
    crossover_type = input("Enter the number of which type of crossover you want: 1.Single  2.Two  3.Unifrom: ")
    selection = input(
        "Enter the number of which type of parent selection you want: 0.Random  Selection 1.RWS  2.SUS  3.Tournament Selection:  ")
    mutation_type = input(
        "Enter the number of which type of mutation you want: 1.inversion_mutation  2.scramble_mutation: ")
    niching = input("Enter the number of the option you want: \n1.niching with shared fitness\n 2.clustering \n3.crowding \noption number: ")
    aging = input("Enter the number of the option you want: 0.without aging 1.with aging: ")

    # filename = "example.txt"
    #
    # with open(filename, "r") as file:
    #      items_weights = [int(line.strip()) for line in file if line.strip()]
    items_weights = [38, 100, 60, 42, 20, 69, 24, 23, 92, 32, 84, 36, 65, 84, 34, 68, 64, 33, 69, 27, 47, 21, 85, 88, 59, 61, 50,
           53, 37, 75, 64, 84, 74, 57, 83, 28, 31, 97, 61, 36, 46, 37, 96, 80, 53, 51, 68, 90, 64, 81, 66, 67, 80, 37,
           92, 67, 64, 31, 94, 45, 80, 28, 76, 29, 64, 38, 48, 40, 29, 44, 81, 35, 51, 48, 67, 24, 46, 38, 76, 22, 30,
           67, 45, 41, 29, 41, 79, 21, 25, 90, 62, 34, 73, 50, 79, 66, 59, 42, 90, 79, 70, 66, 80, 35, 62, 98, 97, 37,
           32, 75, 91, 91, 48, 26, 23, 32, 100, 46, 29, 26]

    MUTATIONRATE = 0.25
    MUTATION = random.randint(0, 32767) * MUTATIONRATE
    BINPACKING = BinPacking(int(capacity), int(population_size), items_weights, MUTATION, mutation_type, crossover_type,
                            selection,aging, 0.5)
    BINPACKING.init_population()
    bin_diff = []
    number_generation = []
    start_absolute_best = time.time()
    for i in range(int(max_generation)):
        start_clockticks = time.perf_counter()  # Measure clock ticks
        start_absolute = time.time()  # Measure absolute time
        #BINPACKING.fitness_first_fit()
        BINPACKING.fitness()
        BINPACKING.genetic_algorithm(i,int(max_generation),niching)
        print("for generation number " + str(i) + " : ")
        print("Fitness average and Fitness std: " + str(BINPACKING.fitness_avg_std()))
        print("allele: " + str(BINPACKING.count_unique_bins()))
        print("Top_average: " + str(BINPACKING.Top_average()))
        print("bins differences: " + str(BINPACKING.sum_bins_difference()))
        bin_diff.append(BINPACKING.sum_bins_difference())
        number_generation.append(i)
        end_absolute = time.time()
        end_clockticks = time.perf_counter()
        print("CLOCK TICKS: " + str(end_clockticks - start_clockticks) + " ,absolute Time: " + str(
            end_absolute - start_absolute))
        BINPACKING.print_best_solution()
        BINPACKING.swap()

    BINPACKING.create_histogram(bin_diff,number_generation)
    end_absolute_best = time.time()
    print("absolute Time: " + str(end_absolute_best - start_absolute_best))

