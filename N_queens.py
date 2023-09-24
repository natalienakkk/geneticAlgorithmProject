import numpy as np
from Struct import Struct
import random
from random import randint
import statistics
import time
import math
import matplotlib.pyplot as plt
MAX = 0.7
MIN = 0.1

class N_Queens:
    def __init__(self, N, POPSIZE,new_POPSIZE,MUTATION,mutation_type,crossover_type,selection,elitism_size):
        self.N = N
        self.ELITISM_SIZE=elitism_size*POPSIZE
        self.population = []
        self.new_population= []
        self.POPSIZE = POPSIZE
        self.MUTATION_RATE=MUTATION
        self.mutation_type=mutation_type
        self.crossover_type=crossover_type
        self.selection=selection
        self.new_POPSIZE=new_POPSIZE


#1.build a pemutation from 0->N-1
#2.create individual using Struct
#3.add individual to population list
    def init_population(self):
        for i in range(self.POPSIZE):
            #randomString = np.random.permutation(self.N)
            random_permutation = [i for i in range(0, self.N)]
            random.shuffle(random_permutation)
            individual = Struct(self.N,random_permutation)
            self.population.append(individual)

#for each individvual calculate the fitness by suming up #of conflicts for each queen
    def fitness(self):
        for i in range(self.new_POPSIZE):
            fitness =0
            for queen_num in range(self.N):
                fitness += self.calc_conflicts(queen_num,self.population[i].NQueens)
            self.population[i].fitness=fitness


# calculate the conflicts only in diagnol because with permutation there is no conflicts in rows/cols
    def calc_conflicts(self,queen_num,indivivual):
        conflicts = 0
        row_i = int(indivivual[queen_num])
        for j in range(queen_num + 1, self.N):
            row_j = int(indivivual[j])
            if abs(row_i - row_j) == abs(queen_num - j):
                conflicts += 1
        return conflicts

#in case the user wants with age for every individual we call fitness_age function and update its new fitness
    def aging(self):
        self.fitness()
        max_age = max(individual.age for individual in self.population)
        for i in range(self.new_POPSIZE):
            new_fitness=self.fitness_age(self.population[i].fitness, self.population[i].age , max_age)
            self.population[i].fitness = new_fitness


    def fitness_age(self, candidate_fitness, age, max_age, alpha=0.5):
        # calculate the fitness score for the candidate solution
        original_score = candidate_fitness
        if max_age == 0:
            age_score = 1
        else:
            # normalize the age component
            normalized_age = age / max_age
            # calculate the age component of the fitness score
            age_score = 1 - normalized_age  # reverse the age score so that younger candidates get higher scores
        # combine the two scores with a weighted sum
        total_score = (1 - alpha) * original_score + alpha * age_score
        return total_score

#pick 3 positions randomly and then inverse between two indexes and then change place according to third index
    def inversion_mutation(self,individual,mutation_rate):
        if random.random() < mutation_rate:
            pos1,pos2 = sorted([randint(0,self.N-1),randint(0,self.N-1)])
            mov_pos = randint(pos2,self.N-1)
            individual=individual[0:pos1] + individual[pos2:mov_pos] + individual[pos1:pos2][::-1] + individual[mov_pos:]
        return individual



#pick both indexes randomly and then shuffle individual[first index:second index]
    def scramble_mutation(self,individual,mutation_rate):
        if random.random() < mutation_rate:
            pos1, pos2 = random.sample(range(len(individual)), 2)
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            sublist = individual[pos1:pos2]
            random.shuffle(sublist)
            individual[pos1:pos2] = sublist
            #print(pos1,pos2,individual)
        return individual


#1.we get both parents and pick a random index
#2.creating 2 children:
#1.2.for this random index in both parent we switch the values
#2.2.then we go in a loop in each parent and  make sure to have a permutation
#2.3 we create each child and return it
    def PMX(self,parent1,parent2):
        offspring1 = [-1] * self.N
        offspring2 = [-1] * self.N
        random_index=random.randint(0,self.N-1)
        value1=parent1[random_index]
        value2=parent2[random_index]
        parent1[random_index],parent2[random_index]=parent2[random_index],parent1[random_index]
        for i in range(self.N):
            if parent1[i]==value2 and i!=random_index:parent1[i]=value1
            elif parent2[i]==value1 and i!=random_index:parent2[i]=value2
        offspring1=parent1
        offspring2=parent2
        #print("offsprings:")
        #print(offspring1,offspring1)
        child1=Struct(self.N,offspring1)
        child2=Struct(self.N,offspring2)
        return child1,child2


    def CX(self,parent1_NQueen,parent2_NQueen):
            nqueen = [-1] * self.N
            # Find cycles
            cycle = 0
            while -1 in nqueen:
                #add each cycle
                if cycle % 2 == 0:
                    start = nqueen.index(-1)
                    index = start
                    while nqueen[index] == -1:
                        nqueen[index] = parent1_NQueen[index]
                        if parent1_NQueen[index]==-1:
                            index = parent2_NQueen.index(parent1_NQueen[index])
                else:
                    #add the rest
                    indices = [i for i, x in enumerate(nqueen) if x == -1]
                    for index in indices:
                        nqueen[index] = parent2_NQueen[index]
                cycle += 1
            child = Struct(self.N, nqueen)
            return child



    def genetic_algorithm(self,aging,generation,max_generation,mutation_control):
        self.elitism()
        while len(self.new_population) < self.POPSIZE:
            parent1_index = randint(0, self.new_POPSIZE // 2)
            parent2_index = randint(0, self.new_POPSIZE // 2)
            parent1_NQueens=self.population[parent1_index].NQueens
            parent2_NQueens=self.population[parent2_index].NQueens
            parent1_age=self.population[parent1_index].age
            parent2_age=self.population[parent2_index].age


            if aging == "1":
                self.population = [individual for individual in self.population if individual.age <= MAX_AGE]
                self.new_POPSIZE = len(self.population)
                #print(self.new_POPSIZE)


            offspring1 = [-1] * self.N
            offspring2 = [-1] * self.N
            child1 = Struct(self.N, offspring1)
            child2 = Struct(self.N, offspring2)
            # RWS parent seletion
            if self.selection == "1":
                parent1_NQueens=self.RWS()
                parent2_NQueens=self.RWS()
            #SUS parent selection
            elif self.selection=="2":
                selected_parents = self.SUS(self.new_POPSIZE)
                parent1_NQueens = random.choice(selected_parents)
                parent2_NQueens = random.choice(selected_parents)
            #Tournament_selection
            elif self.selection == "3":
                k = random.randint(1, len(self.population) - 1)
                parent1_NQueens=self.Tournament_selection(k)
                k = random.randint(1, len(self.population) - 1)
                parent2_NQueens = self.Tournament_selection(k)
            #crossover_type==PMX,send NQueens of each parent
            if self.crossover_type == "1":
                child1,child2 = self.PMX(parent1_NQueens,parent1_NQueens)
            #crossover type==CX
            elif self.crossover_type == "2":
                child1 = self.CX(parent1_NQueens,parent2_NQueens)
            # non_unifrom mutation -linear
            if mutation_control == "3":
                mutation_rate1 = self.non_uniform_linear(child1.NQueens, generation, max_generation)
                mutation_rate2 = self.non_uniform_linear(child2.NQueens, generation, max_generation)
            # non_unifrom mutation -non-linear
            if mutation_control =="4":
                mutation_rate1 = self.non_unifrom_non_linear(child1.NQueens, generation, 0.25)
                mutation_rate2 = self.non_unifrom_non_linear(child2.NQueens, generation, 0.25)
            #THM mutation
            if mutation_control == "5":
                fitness_avg = self.fitness_avg_std()
                fitnesses = [individual.fitness for individual in self.population]
                max_fitness = max(fitnesses)
                trigger_fitness = 3
                trigger_avg = 5
                mutation_rate1 = self.THM(child1.NQueens,trigger_fitness ,trigger_avg , generation ,max_fitness ,fitness_avg)
                mutation_rate2 = self.THM(child2.NQueens, trigger_fitness, trigger_avg, generation, max_fitness, fitness_avg)
            #adaptive mutation
            if mutation_control == "6":
                max_fitness = 7
                mutation_rate1 = self.adaptive_mutation(child1.NQueens,max_fitness,child1.fitness)
                mutation_rate2 = self.adaptive_mutation(child2.NQueens,max_fitness,child2.fitness)
            #self adaptive mutation
            if mutation_control =="7":
                fitnesses = [individual.fitness for individual in self.population]
                fitnesses.append(child1.fitness)
                mutation_rate1 = self.self_adaptive_mutation(child1.NQueens,fitnesses)
                fitnesses.remove(child1.fitness)
                fitnesses.append(child2.fitness)
                mutation_rate2 = self.self_adaptive_mutation(child2.NQueens,fitnesses)
                fitnesses.remove(child2.fitness)
            if mutation_control =="8":
                mutation_rate1 = self.MUTATION_RATE
                mutation_rate2 = self.MUTATION_RATE
            #mutation type = inversion mutation
            if self.mutation_type == "1":
                child1.NQueens = self.inversion_mutation(child1.NQueens,mutation_rate1)
                child2.NQueens = self.inversion_mutation(child2.NQueens,mutation_rate2)
            #mutation type= scramble mutation, update NQueens of each child
            if self.mutation_type == "2":
                child1.NQueens = self.scramble_mutation(child1.NQueens,mutation_rate1)
                child2.NQueens = self.scramble_mutation(child2.NQueens,mutation_rate2)


            #add both new childs to new_population
            self.new_population.append(child1)
            if child2.NQueens[0] != -1:
                self.new_population.append(child2)

        for individual in self.population:
            individual.age += 1


    def create_histogram(self,string_diff, generations):
        plt.plot(generations, string_diff)
        plt.xlabel('Generation')
        plt.ylabel('string difference=distance')
        plt.show()

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
            return individual.NQueens
        pop_fitness = [individual.fitness for individual in self.population]

        # scale fitnesses
        scaled_pop_fitness = abs(self.winsorize(pop_fitness))
        # find the probabilty of each individual in population by individual's fitness/total_fitness
        pop_probability = [fitness / total_fitness for fitness in scaled_pop_fitness]
        # create slices :each slice corresponds to an individual's probability of selection(like example we did in class)
        slices = [sum(pop_probability[:i + 1]) for i in range(len(pop_probability))]
        # spin the roulette by generating random number between 0 and 1
        random_spin = random.random()
        # if random number falls into slice[j] -> return individual in population[j]
        for j in range(len(slices)):
            if random_spin <= slices[j]:
                return self.population[j].NQueens

        individual = random.choice(self.population)
        return individual.NQueens

    def SUS(self, num):
        selected_parents = []
        pop_fitness = [individual.fitness for individual in self.population]
        # scale fitnesses
        scaled_pop_fitness = abs(self.winsorize(pop_fitness))

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
                    selected_parents.append(individual.NQueens)
                    break
            random_start += step_size

        return selected_parents

    # scaling function(given)
    def winsorize(self,data, percentile=5):
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
        best_individual = selected_k[0].NQueens
        best_individual_fitness = k_fitnesses[0]

        for i in range(len(selected_k)):
            if k_fitnesses[i] < best_individual_fitness:
                best_individual_fitness = k_fitnesses[i]
                best_individual = selected_k[i].NQueens

        return best_individual

    def elitism(self):
        for i in range(int(self.ELITISM_SIZE)):
            self.new_population.append(self.population[i])

    def swap(self):
        self.population = self.new_population
        self.new_population=[]

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def fitness_sort(self, x):
        return x.get_fitness()

    def print_best_solution(self):
        self.sort_by_fitness()
        print("Best Solution for N-Queens : ", end=" ")
        for i in range(self.N):
            print(self.population[0].NQueens[i], end=" ")
        print(",and for fitness :", self.population[0].fitness)

#parameters 5+6
    def fitness_avg_std(self):
        fitnesses = [individual.fitness for individual in self.population]
        fitness_average = sum(fitnesses) / len(fitnesses)
        fitness_std = statistics.stdev(fitnesses)
        return fitness_average, fitness_std

    #Top-Average Selection Probability Ratio
    # calculated by dividing the probability of selecting the best individual by the average probability of selecting an individual from the population.
    def Top_average(self):
        fitnesses = sum([individual.fitness for individual in self.population])
        probabilities = [fitness / fitnesses for fitness in [gene.fitness for gene in self.population]]
        avg_probability = sum(probabilities) / len(probabilities)
        top_probability = max(probabilities)
        top_avg = top_probability / avg_probability
        return top_avg

    # count how many permutation are different
    def count_unique_permutations(self):
        unique_permutations = set()
        for individual in self.population:
            permutation = tuple(individual.NQueens)
            unique_permutations.add(permutation)
        return len(unique_permutations)

    def count_changes(self, transform):
        changes = 0
        for i in range(len(transform)):
            for j in range(i + 1, len(transform)):
                if transform[i] > transform[j]:
                    changes += 1
        return changes

#check the difference between two permutation
    def differences(self, permutation1, permutation2):
        assert len(permutation1) == len(permutation2)
        index_map = {value: index for index, value in enumerate(permutation1)}
        #tranform permutation2 to permutation1
        transform = [index_map[value] for value in permutation2]
        return self.count_changes(transform)

#sum the difference between all genes to total differences
    def sum_permutation_differences(self):
        total_differences = 0
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                individual = self.population[i]
                individual1 = individual.NQueens
                individual = self.population[j]
                individual2 = individual.NQueens
                total_differences += self.differences(individual1, individual2)
        return total_differences


#----------------------------------------code for Lab2----------------------------------------#

#kendell tau distance function
#1. build all possible pairs in permutation1
#2. check if each pair is in the same order in permutation2
#3. if not: counter++
    def Kendall_Tau_distance(self,permutation1,permutation2):
        pairs = []
        n = len(permutation1)
        for i in range(n):
            for j in range(i+1,n):
                pairs += [(permutation1[i],permutation1[j])]

        n = len(pairs)
        counter=0
        for i in range(n):
            x,y = pairs[i]
            if ((permutation2.index(x)>permutation2.index(y)) and (x>y)) \
                    or ((permutation2.index(x)>permutation2.index(y)) and (x<y)):
                counter += 1

        return counter

#this function takes each pair of permutation and calls Kendall_Tau_distance function and sums up
    def pairing_Kendall_Tau(self):
        total_difference=0
        for i in range (self.N):
            for j in range(i+1,self.N):
                total_difference += self.Kendall_Tau_distance(self.population[i].NQueens, self.population[j].NQueens)

        return total_difference

    # LAB2 EXTRA: Non-unifrom mutation - linear
    def non_uniform_linear(self, child, generation, total_generations):
        mutate_child = child
        mutation_rate = MAX - (MAX - MIN) * (generation / total_generations)
        return mutation_rate

    # LAB2 EXTRA: Non-unifrom mutation - non_linear
    def non_unifrom_non_linear(self,child, generation, r):
        mutate_child = list(child)
        if generation != 0:
            mutation_rate = (2 * (self.MUTATION_RATE ** 2) * math.exp(generation * r)) / (
                        generation + generation * math.exp(generation * r))
        else:
            mutation_rate = self.MUTATION_RATE
        return mutation_rate


    def THM(self,child, trigger_fitness, trigger_avg, trigger_generation, fitness, avg):
        mutate_child = list(child)
        if fitness > trigger_fitness or avg > trigger_avg:
            mutation_rate = self.MUTATION_RATE * 2
        else:
            mutation_rate = self.MUTATION_RATE
        return mutation_rate

    def adaptive_mutation(self,child, max_fitness,child_fitness):
        mutate_child = list(child)
        mutation_rate = self.MUTATION_RATE * (1 - child_fitness / max_fitness)
        return mutation_rate


    def relative_fitness(self,fitnesses):
        # Compute population mean fitness
        mean_fitness = statistics.mean(fitnesses)
        # Compute relative fitness for each individual
        relative_fitness = [f / mean_fitness for f in fitnesses]
        # Normalize relative fitness
        std_dev = statistics.stdev(relative_fitness)
        normalized_relative_fitness = [r / std_dev for r in relative_fitness]
        return normalized_relative_fitness

    def self_adaptive_mutation(self,child, fitnesses):
        mutate_child = list(child)
        relative_fitnesses = self.relative_fitness(fitnesses)
        mutation_rate = self.MUTATION_RATE * (1 - relative_fitnesses[-1])
        return mutation_rate


if __name__ == '__main__':
    print("Welcome to NQueen Solution")
    max_generation=input("Please enter number of generations(must be integer): ")
    population_size = input("Please enter population size(must be integer): ")
    number_queens = input("Please enter number of queens: ")
    crossover_type = input("Enter the number of which type of crossover you want: 1.PMX  2.CX: ")
    selection = input("Enter the number of which type of parent selection you want: 0.Random  Selection 1.RWS  2.SUS  3.Tournament Selection:  ")
    mutation_type = input("Enter the number of which type of mutation you want: 1.inversion_mutation \n 2.scramble_mutation:")
    mutation_control = input("Enter the number of which type of mutation control you want: \n3.Non uniform mutation-linear \n4.Non uniform mutation-non_linear \n5.Triggered Hyper Mutation \n6.Adaptive mutation \n7.Self Adaptive \n8.constant mutation rate \noption number: ")
    aging = input("Enter the number of the option you want: 0.without aging 1.with aging: ")
    MUTATIONRATE = 0.25
    MAX_AGE=10
    MUTATION = random.randint(0, 32767) * MUTATIONRATE
    NQUEEN=N_Queens(int(number_queens), int(population_size) , int(population_size), MUTATION, mutation_type, crossover_type,selection,0.1)
    NQUEEN.init_population()
    start_absolute_best = time.time()
    permutation_diff = []
    number_generation = []
    for i in range(int(max_generation)):
        start_clockticks = time.perf_counter()  # Measure clock ticks
        start_absolute = time.time()  # Measure absolute time
        if aging == "0":
            NQUEEN.fitness()
        else:
            NQUEEN.aging()

        NQUEEN.genetic_algorithm(aging,i,int(max_generation),mutation_control)
        print("for generation number "+str(i)+" : ")
        print("Fitness average and Fitness std: "+str(NQUEEN.fitness_avg_std()))
        print("allele: "+str(NQUEEN.count_unique_permutations()))
        print("Top_average: " +str(NQUEEN.Top_average()))
        print("Total kendall tau distnace: " + str(NQUEEN.pairing_Kendall_Tau()))
        if crossover_type == "1":
            print("permutation differences: " + str(NQUEEN.sum_permutation_differences()))
        permutation_diff.append(NQUEEN.sum_permutation_differences())
        number_generation.append(i)
        end_absolute = time.time()
        end_clockticks = time.perf_counter()
        print("CLOCK TICKS: " + str(end_clockticks - start_clockticks) + " ,absolute Time: " + str(end_absolute - start_absolute))
        NQUEEN.print_best_solution()



        if NQUEEN.population[0].fitness == 0: break
        NQUEEN.swap()
    NQUEEN.create_histogram(permutation_diff,number_generation)
    end_absolute_best = time.time()
    # print("BEST FOR ALL GENERATIONS: ")
    # NQUEEN.print_best_solution()
    print("absolute Time: " + str(end_absolute_best - start_absolute_best))



