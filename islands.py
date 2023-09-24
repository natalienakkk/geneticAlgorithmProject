from random import randint
import random
import numpy as np

MUTATIONRATE = 0.25
MIGRATION_RATE = 10
MIGRATION_NUM = 10

class data:
    def __init__(self, vector):
        self.vector = vector # vector of 2 value (x,y)
        self.fitness = -1

class islands:
    def __init__(self,POPSIZE, MUTATION, elitism_size, func_type):
        self.POPSIZE = POPSIZE
        self.ELITISM_SIZE = elitism_size*POPSIZE
        self.population = []
        self.new_population = []
        self.MUTATION_RATE = MUTATION
        self.func_type = func_type


    def init_population(self):
        for i in range(self.POPSIZE):
            if self.func_type == "function_f":
                vector = np.random.uniform(low=-3, high=3, size=(2,))
            elif self.func_type == "function_g":
                vector = np.random.uniform(low=-3, high=4, size=(2,))
            individual = data(vector)
            self.population.append(individual)

    def fitness(self):
        for individual in self.population:
            x, y = individual.vector[0] , individual.vector[1]
            if self.func_type == "function_f":
                if (x ** 2 + y ** 2 <= 9) == True:
                    individual.fitness = (x ** 2 + y ** 2)
                else:
                    individual.fitness = -1
            elif self.func_type == "function_g":
                if ((x - 5) ** 2 + (y - 5) ** 2 <= 4) ==True:
                    individual.fitness = ((x - 5) ** 2 + (y - 5) ** 2)
                else:
                    individual.fitness = -1

    def fitness_sort(self, x):
        return x.fitness

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def elitism(self):
        for i in range(int(self.ELITISM_SIZE)):
            self.new_population.append(self.population[self.POPSIZE - 1 - i])

    def swap(self):
        self.population = self.new_population
        self.new_population=[]

    def mutation(self, child):
        mutate_child = list(child)
        if random.random() < self.MUTATION_RATE:
            if self.func_type == "function_f":
                mutate_child[0]=random.uniform(-3,3)
                mutate_child[1]=random.uniform(-3,3)
            else:
                mutate_child[0]=random.uniform(-3,4)
                mutate_child[1]=random.uniform(-3,4)
            return mutate_child
        return child



    def genetic_algorithm(self):
        self.elitism()
        while len(self.new_population) < self.POPSIZE:
            parent1_index = randint(self.POPSIZE//2, self.POPSIZE-1)
            parent2_index = randint(self.POPSIZE//2, self.POPSIZE-1)
            parent1_vector = self.population[parent1_index].vector
            parent2_vector = self.population[parent2_index].vector

            #crossover
            offspring1 = (parent1_vector[0], parent2_vector[1])
            offspring2 = (parent2_vector[0], parent1_vector[1])
            #mutation
            offspring1 = self.mutation(offspring1)
            offspring2 = self.mutation(offspring2)
            # build children after crossover and mutation
            child1 = data(offspring1)
            child2 = data(offspring2)
            # add them to the population
            self.new_population.append(child1)
            self.new_population.append(child2)

    def print_best_solution(self, gen):
        print("for generation", gen ,"Best Solution:", self.population[self.POPSIZE-1].vector, end=" ")
        print(" ,with fitness :", self.population[self.POPSIZE-1].fitness)


def migration(f, g, selection_type, immg_num, check_valid):
    if selection_type == "Random":
        migrants = Random(g.population, immg_num)
    elif selection_type == "RWS":
        migrants = RWS(g.population, immg_num)
    updated_population = replacement(f.population, migrants, check_valid)
    f.population = updated_population


def replacement(population, immigrants, check_valid):
    population.sort(key=lambda x: x.fitness)
    j = 0
    for i in range(len(population)):
        if j >= len(immigrants):break
        if check_valid == "f_valid":
            x, y = immigrants[j].vector[0], immigrants[j].vector[1]
            if (x ** 2 + y ** 2 <= 9) == True:
                population[i] = immigrants[j]
                j += 1
        elif check_valid == "g_valid":
            x, y = immigrants[j].vector[0], immigrants[j].vector[1]
            if (x - 5) ** 2 + (y - 5) ** 2 <= 4 == True:
                population[i] = immigrants[j]
                j += 1

    return population

def Random(population, immg_num):
    return random.sample(population, immg_num)



def RWS(population, immg_num):
    fitnesses = sum(individual.fitness for individual in population)
    probabilities = [individual.fitness / fitnesses for individual in population]
    migrants = random.choices(population, weights=probabilities, k=immg_num)
    return migrants


if __name__ == '__main__':
    max_generation = input("Please enter number of generations(must be integer): ")
    selection_type = "Random" # change selection type to rws,tournment ot random

    F = islands(100, MUTATIONRATE, 0.02, "function_f")
    G = islands(100, MUTATIONRATE, 0.02, "function_g")

    F.init_population()
    G.init_population()

    for i in range(int(max_generation)):
        F.fitness()
        F.sort_by_fitness()
        F.print_best_solution(i)
        F.genetic_algorithm()
        F.swap()

        G.fitness()
        G.sort_by_fitness()
        G.print_best_solution(i)
        G.genetic_algorithm()
        G.swap()

        if i % MIGRATION_RATE == 0:
            F.fitness()
            G.fitness()
            migration(F,G, selection_type,MIGRATION_NUM ,"f_valid" )
            F.fitness()
            G.fitness()
            migration(G, F, selection_type,MIGRATION_NUM, "g_valid" )



