import random
import statistics
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import string
import matplotlib.pyplot as plt
SHARING_RADIUS = 120


GA_MUTATIONRATE = 0.4
MAX_AGE = 10
MAX = 0.7
MIN = 0.1
K = 10


def binary_represent(individual):
    binary_representation = ''
    for i in range(len(individual)):
        char = individual[i]
        unicode_code_point = ord(char)
        binary_representation += bin(unicode_code_point)[2:].zfill(8)
    return binary_representation

def char_represent(individual):
    char_representation = []
    for i in range(0, len(individual), 8):
        # Convert the binary representation to a Unicode code point (integer)
        unicode_code_point = int(individual[i:i+8], 2)
        # Convert the Unicode code point to its corresponding character
        char_representation += chr(unicode_code_point)
    return char_representation


# Define the fitness function
def fitness(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:  # how many ch
            score += 1
    return score

def fitness_age(candidate, age, max_age, fitness_type ,alpha=0.5):
    # calculate the fitness score for the candidate solution
    if fitness_type == "0":
        original_score = heuristic(candidate)
    if fitness_type == "1":
        original_score = fitness(candidate)
    if max_age == 0:
        age_score = 1
    else:
        # normalize the age component
        normalized_age = age / max_age
        # calculate the age component of the fitness score
        age_score = 1 - normalized_age # reverse the age score so that younger candidates get higher scores
    # combine the two scores with a weighted sum
    total_score = (1 - alpha) * original_score + alpha * age_score
    return total_score

# 1. if same char in the same place ,add 15 to score
# 2. if char is found but not in the right place ,add 10 to score
# 3. else don't add anything
def heuristic(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 15
        elif np.isin(individual[i], target) == True:
            score += 10
        else:
            score += 0
    return score



def histogram(pop_fitnesses):
    plt.hist(pop_fitnesses, bins=10, rwidth=0.6, color='pink')  # You can adjust the number of bins as needed
    plt.title("Fitness distribution among population")
    plt.xlabel("Fitness")
    plt.ylabel("Amount of individuals")
    plt.show()

def fitness_avg_std(fitnesses):
    fitness_average = sum(fitnesses) / len(fitnesses)
    fitness_std = statistics.stdev(fitnesses)
    return fitness_average, fitness_std


# Define the genetic algorithm
def genetic_algorithm(crossover_type, parentSelection_type, mutation_, aging, fitness_type, trigger_fitness,
                      trigger_avg,mutation_control,niching, pop_size, num_genes, fitness_func, max_generations):
    # Initialize the population with random individuals
    population = []
    for i in range(pop_size):
        individual = [[chr(random.randint(32, 126)) for j in range(num_genes)], 0]
        population.append(individual)

    # p=binary_represent(population)
    # char_represent(p)

    start_clockticks_ = time.perf_counter()  # Measure clock ticks
    start_absolute_ = time.time()  # Measure absolute time

    string_diff = []
    number_generation = []
    # Evolve the population for a fixed number of generations
    for generation in range(max_generations):

        start_clockticks = time.perf_counter()  # Measure clock ticks
        start_absolute = time.time()  # Measure absolute time
        # take only the individuals with age less than max only if the user choose aging
        if aging == "1":
            population = [individual for individual in population if individual[1] <= MAX_AGE]

        # Evaluate the fitness of each individual
        max_age = max(individual[1] for individual in population)

        # fitnesses with age
        if aging == "1":
            fitnesses = [fitness_age(individual[0], individual[1], max_age, fitness_type) for individual in population]
        else:
            fitnesses = [fitness_func(individual[0]) for individual in population]

        if generation / max_generations == 0 or generation / max_generations == 0.25 or generation / max_generations == 0.5:
            # print histogram for fitness distribution
            histogram(fitnesses)

        # calculate avg+std
        fitness_average, fitness_std = fitness_avg_std(fitnesses)

        # Select the best individuals for reproduction
        popsize = len(population)
        elite_size = int(popsize * 0.1)
        elite_indices = sorted(range(popsize), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]
        index = sorted(range(popsize), key=lambda i: fitnesses[i], reverse=True)[:popsize]

        if niching ==1:
            similarity_matrix = build_similarity_matrix(population)
            fitnesses = niching_with_shared_fitness(population, similarity_matrix, fitnesses)

        if niching == 2:
            silhouette_score = np.zeros(K - 1)
            for i in range(2, K + 1):
                clusters = Kmean(population, i)
                silhouette_score[i - 2] = Silhouette(population, clusters,i)
            optimal_k = np.argmax(silhouette_score) + 2
            clusters = Kmean(population, optimal_k)


        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            # Random parent selection
            parent1 = random.choice([population[i][0] for i in index[:popsize // 2]])
            parent2 = random.choice([population[i][0] for i in index[:popsize // 2]])

            # RWS parent selection
            if parentSelection_type == "1":
                parent1 = RWS([individual[0] for individual in population], fitnesses)
                parent2 = RWS([individual[0] for individual in population], fitnesses)
            # SUS parent selection
            elif parentSelection_type == "2":
                selected_parents = SUS([individual[0] for individual in population], fitnesses, popsize - elite_size)
                parent1 = random.choice(selected_parents)
                parent2 = random.choice(selected_parents)
            # Tournament parent selection
            elif parentSelection_type == "3":
                k = random.randint(1, len(population) - 1)
                parent1 = Tournament_selection([individual[0] for individual in population], k)
                k = random.randint(1, len(population) - 1)
                parent2 = Tournament_selection([individual[0] for individual in population], k)

            # save the index of both parents before mutations and crossover
            index1 = index2 = -1
            for i in range(len(population)):
                if population[i][0] == parent1:
                    index1 = i
                if population[i][0] == parent2:
                    index2 = i
                if index1 != -1 and index2 != -1: break

            # no crossover
            if crossover_type == "0":
                child = parent1
            # single crossover type
            elif crossover_type == "1":
                child = single_crossover(parent1, parent2)
            # two crossover type
            elif crossover_type == "2":
                child = two_crossover(parent1, parent2)
            # uniform crossover type
            else:
                parent1 = list(binary_represent(parent1))
                parent2 = list(binary_represent(parent2))
                child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]
                child = char_represent(''.join(child))

            mutated_child = child
            # no mutation
            if mutation_ == "0":
                mutated_child = child
            # basic mutation operator
            elif mutation_ == "1" and mutation_control == "1":
                mutated_child = mutation(child)
            # Non-uniform mutation - linear
            elif mutation_control == "2" and mutation_ == "1":
                # calculate the mutation rate for non-uniform mutation
                # mutation_rate = GA_MUTATIONRATE * ((1 - generation / max_generations)**4)
                mutated_child = non_uniform_linear(child, generation, max_generations)
            # Non-unifrom mutation - non_linear
            elif mutation_control == "3" and mutation_ == "1":
                mutated_child = non_unifrom_non_linear(child, generation, 0.25)
            # THM mutation
            elif mutation_control == "4" and mutation_ == "1":
                max_fitness = max(fitnesses)
                mutated_child = THM(child, trigger_fitness, trigger_avg, max_fitness, fitness_average)
            elif mutation_control == "5" and mutation_ == "1":
                max_fitness = 0
                if fitness_type == "0":
                    max_fitness = 195
                elif fitness_type == "1":
                    max_fitness = 13
                mutated_child = adaptive_mutation(child, max_fitness)
            # self adaptive mutation
            elif mutation_control == "6" and mutation_ == "1":
                if aging == "0":
                    child_fitness = fitness_func(child)
                else:
                    child_fitness = fitness_func(child, 0, max_age, fitness_type)
                fitnesses.append(child_fitness)
                mutated_child = self_adaptive_mutation(child, fitnesses)

            offspring.append([mutated_child, 0])
        population = elites + offspring

        if niching == 3:
            crowding(child, fitness_func(child), index1, index2, population, fitnesses)

        for individual in population:
            individual[1] += 1

        end_clockticks = time.perf_counter()  # Measure clock ticks
        end_absolute = time.time()  # Measure absolute time
        print("for generation number " + str(generation + 1) + " : ")
        print("fitness average is: " + str(fitness_average) + " ,standard deviation is: " + str(fitness_std) +
              " ,Top-Average Selection Probability Ratio: " + str(Top_average(population, fitness_type, aging)))
        print("CLOCK TICKS: " + str(end_clockticks - start_clockticks) + " ,absolute Time: " + str(
            end_absolute - start_absolute))
        print("allele: ", str(count_unique_letters(population)) + " ,strings difference: " + str(
            sum_strings_difference(population)))

        string_diff.append(count_unique_letters(population))
        number_generation.append(generation)
        print("Hamming distance: " + str(total_hamming_distance(population)))
        # print("Edit distance: "+str(total_edit_distance(population)))
        # print("Edit distance: "+str(edit_distance(population[0][0],population[1][0])))
        if max(fitnesses) == 195: break
        if max(fitnesses) == 13: break


    #histogram for lab2 question 4
    create_histogram(string_diff,number_generation)

    # Find the individual with the highest fitness
    if aging == "1":
        best_individual = max(population,
                              key=lambda individual: fitness_func(individual[0], individual[1], max_age, fitness_type))
        best_fitness = fitness_func(best_individual[0], best_individual[1], max_age, fitness_type)
    else:
        best_individual = max(population, key=lambda individual: fitness_func(individual[0]))
        best_fitness = fitness_func(best_individual[0])


    end_absolute_ = time.time()
    end_clockticks_ = time.perf_counter()
    print("CLOCK TICKS and Absolute time for maximum: " + str(end_clockticks_ - start_clockticks_) + " , " + str(
        end_absolute_ - start_absolute_))
    histogram(fitnesses)  # print last generation
    return best_individual, best_fitness

def create_histogram(string_diff, generations):
    plt.plot(generations, string_diff)
    plt.xlabel('Generation')
    plt.ylabel('string difference=distance')
    plt.show()

def create_population(population,clusters,cluster_num):
    new_population =[]
    for i in range(len(clusters)):
        if clusters[i] == cluster_num:
            new_population.append(population[i])
            #new_population[i][1]=population[i][1]
    return new_population


# 1.find random point in parent1
# 2.build child:
# 2.1: 0->random_index-1=parent1 ,random_index->last index =parent2
#LAB2 EXTRA: parents are binary represented
def single_crossover(parent1, parent2):
    parent1 = binary_represent(parent1)
    parent2 = binary_represent(parent2)
    random_index = random.randint(1, (len(parent1) - 1)//8)
    random_index = random_index * 8
    child = parent1[:random_index] + parent2[random_index:]
    child = char_represent(child)
    return child


# 1.find two random points in parent1
# 2.sort both points
# 3.build child:
# 3.1: 0->random_index-1=parent1, random_index->random_index2=parent2 , random_index2->last index =parent1
#LAB2 EXTRA: parents are binary represented
def two_crossover(parent1, parent2):
    parent1 = binary_represent(parent1)
    parent2 = binary_represent(parent2)
    random_index1 = random.randint(0, len(parent1) - 1)
    random_index2 = random.randint(0, len(parent1) - 1)
    if random_index1 > random_index2:
        random_index1, random_index2 = random_index2, random_index1
    child = parent1[:random_index1] + parent2[random_index1:random_index2] + parent1[random_index2:]
    child = char_represent(child)
    return child


def mutation(child):
    mutate_child = list(child)
    if random.random() < GA_MUTATIONRATE:
        index = random.randint(0, len("Hello, world!") - 1)
        delta = random.randint(32, 121)
        mutate_child[index] = chr((ord(child[index]) + delta) % 122)
        return mutate_child
    return child


#LAB2 EXTRA: Non-unifrom mutation - linear
def non_uniform_linear(child,generation,total_generations):
     mutate_child = list(child)
     mutation_rate = MAX - (MAX - MIN) * (generation / total_generations)
     if random.random() < mutation_rate:
        index = random.randint(0, len("Hello, world!") - 1)
        delta = random.randint(32, 121)
        mutate_child[index] = chr((ord(child[index]) + delta) % 122)
        return mutate_child
     return child

#LAB2 EXTRA: Non-unifrom mutation - non_linear
def non_unifrom_non_linear(child, generation,r):
    mutate_child = list(child)
    if generation!=0:
        mutation_rate = (2*(GA_MUTATIONRATE**2)*math.exp(generation*r))/ (generation+generation*math.exp(generation*r))
    else :
        mutation_rate = GA_MUTATIONRATE
    if random.random() < mutation_rate:
        index = random.randint(0, len("Hello, world!") - 1)
        delta = random.randint(32, 121)
        mutate_child[index] = chr((ord(child[index]) + delta) % 122)
        return mutate_child
    return child



def THM(child, trigger_fitness, trigger_avg,fitness, avg):
    mutate_child = list(child)
    if fitness < trigger_fitness or avg < trigger_avg:
        mutation_rate = GA_MUTATIONRATE*2
    else:
        mutation_rate = GA_MUTATIONRATE
    if random.random() < mutation_rate:
        index = random.randint(0, len("Hello, world!") - 1)
        delta = random.randint(32, 121)
        mutate_child[index] = chr((ord(child[index]) + delta) % 122)
        return mutate_child
    return child

def adaptive_mutation(child,max_fitness):
    mutate_child = list(child)
    child_fitness = fitness(child)
    mutation_rate = GA_MUTATIONRATE*(1-child_fitness/max_fitness)
    if random.random() < mutation_rate:
        index = random.randint(0, len("Hello, world!") - 1)
        delta = random.randint(32, 121)
        mutate_child[index] = chr((ord(child[index]) + delta) % 122)
        return mutate_child
    return child



def relative_fitness(fitnesses):
    # Compute population mean fitness
    mean_fitness = statistics.mean(fitnesses)
    # Compute relative fitness for each individual
    relative_fitness = [f / mean_fitness for f in fitnesses]
    # Normalize relative fitness
    std_dev = statistics.stdev(relative_fitness)
    if std_dev != 0:
        normalized_relative_fitness = [r / std_dev for r in relative_fitness]
    else:
        std_dev+=2.5
        normalized_relative_fitness = [r / std_dev for r in relative_fitness]
    return normalized_relative_fitness

def self_adaptive_mutation(child,fitnesses):
    mutate_child = list(child)
    relative_fitnesses = relative_fitness(fitnesses)
    mutation_rate = GA_MUTATIONRATE * (1 - relative_fitnesses[-1])
    if random.random() < mutation_rate:
        index = random.randint(0, len("Hello, world!") - 1)
        delta = random.randint(32, 121)
        mutate_child[index] = chr((ord(child[index]) + delta) % 122)
        return mutate_child
    return child



def RWS(population, pop_fitness):
    total_fitness = sum(pop_fitness)
    pop_probability = []  # probablity list

    if total_fitness == 0:
        return random.choice(population)

    #scale fitnesses
    scaled_pop_fitness = abs(winsorize(pop_fitness))
    # find the probabilty of each individual in population by individual's fitness/total_fitness
    pop_probability = [fitness / total_fitness for fitness in scaled_pop_fitness]
    # create slices :each slice corresponds to an individual's probability of selection(like example we did in class)
    slices = [sum(pop_probability[:i + 1]) for i in range(len(pop_probability))]
    # spin the roulette by generating random number between 0 and 1
    random_spin = random.random()
    # if random number falls into slice[j] -> return individual in population[j]
    for j in range(len(slices)):
        if random_spin <= slices[j]:
            return population[j]

    return random.choice(population)


def SUS(population, pop_fitness, N ):
    selected_parents = []
    #scale fitnesses
    scaled_pop_fitness = abs(winsorize(pop_fitness))
    total_fitness = sum(scaled_pop_fitness)

    #calculate equal step size
    step_size = total_fitness / N
    #choose random start to begin
    random_start = random.uniform(0, step_size)

    #walk equal step size and add to selected_parents according to indvidual's fitness
    for j in range(N):
        pointer = random_start
        fitness_sum = 0
        for i, individual in enumerate(population):
            fitness_sum += scaled_pop_fitness[i]
            if fitness_sum >= pointer:
                selected_parents.append(individual)
                break
        random_start += step_size

    return selected_parents

# scaling function(given)
def winsorize(data, percentile=5):
     lower_bound = np.percentile(data, percentile)
     upper_bound = np.percentile(data, 100 - percentile)
     data = np.where(data < lower_bound, lower_bound, data)
     data = np.where(data > upper_bound, upper_bound, data)
     mean = np.mean(data)
     std = np.std(data)
     data = (data - mean) / std
     return data


# pick k individuals from population randomly
# check which one among them with the best fitness and return it
def Tournament_selection(population, k):
    selected_k = random.sample(population, k)
    k_fitnesses = [fitness(individual) for individual in selected_k]
    best_individual = selected_k[0]
    best_individual_fitness = k_fitnesses[0]

    for i in range(len(selected_k)):
        if k_fitnesses[i] < best_individual_fitness:
            best_individual_fitness = k_fitnesses[i]
            best_individual = selected_k[i]

    return best_individual

# Top-Average Selection Probability Ratio
# calculated by dividing the probability of selecting the best individual by the average probability of selecting an individual from the population.
def Top_average(population,fitness_type,aging):
    if aging == "1":
        fitnesses = [fitness_age(individual[0], individual[1], max(individual[1] for individual in population),fitness_type) for
                     individual in population]

    elif fitness_type == "0":
        fitnesses = [heuristic(individual[0]) for individual in population]
    else:
        fitnesses = [fitness(individual[0]) for individual in population]
    total_fitness = sum(fitnesses)
    probabilities = [fitness / total_fitness for fitness in fitnesses]
    avg_probability = sum(probabilities) / len(probabilities)
    top_probability = max(probabilities)
    top_avg = top_probability / avg_probability
    return top_avg

#count how many letters are different
def count_unique_letters(population):
    unique_letters = set()
    for individual in population:
        for letter in individual[0]:
            unique_letters.add(letter)
    return len(unique_letters)

#calcultae difeerence between two strings
def gene_difference(individual1, individual2):
    count = 0
    for string1, string2 in zip(individual1, individual2):
        if string1 != string2:
            count += 1
    return count

# sum all the differences between all string = total
def sum_strings_difference(population):
    sum_difference = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            individual1 = population[i][0]
            individual2 = population[j][0]
            diff = gene_difference(individual1, individual2)
            sum_difference += diff
    return sum_difference


#----------------------------------------code for Lab2----------------------------------------#

#take two string and compare each char in string 1 with char in string 2 (same index)
def hamming_distance(individual1, individual2):
    difference = 0
    for i in range(len(individual1)):
        if individual1[i]!=individual2[i]:
            difference += 1
    return difference

#calculate total hamming distance between all individuals in the population
def total_hamming_distance(population):
    total_hamming=0
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            total_hamming += hamming_distance(population[i][0], population[j][0])
    return total_hamming


#want to find the minimum of single char edit (insertions,deletions,substitutions) to transform string to another string
#we use dynamic programming to store the distances between each pair of substring
def edit_distance(individual1, individual2):
    len1 = len(individual1)
    len2 = len(individual2)
    # build a matrix with size :  (len1+1) * (len2+1)
    arr = np.zeros((len1 + 1, len2 + 1))
    # initialize first row and column as the distance between empty string and each substring for both given strings
    for i in range(len1):
        arr[i][0] = i
    for j in range(len2):
        arr[0][j] = j

    # update the minimum distance in every iteration
    for i in range(1, len1+1):
        for j in range(1, len2+1):
        #if it's not the same char in both string then take the minimum between 3 substrings and add 1 for substitution
            if individual1[i-1] != individual2[j-1]:
                min_distance = min(arr[i-1][j], arr[i][j-1], arr[i-1][j-1])
                arr[i][j] = 1+min_distance
            else:#if we have the same char in both strings then we take the distance from the previous substring
                arr[i][j] = arr[i-1][j-1]
    return arr[len1][len2]

def total_edit_distance(population):
    total_edit=0
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            total_edit += edit_distance(population[i][0], population[j][0])
    return total_edit


def crowding(child,child_fitness,parent1_index,parent2_index,population,fitnesses):
    F1 = fitnesses[parent1_index]
    F2 = fitnesses[parent2_index]
    delta_P1_child = abs(F1-child_fitness)
    delta_P2_child = abs(F2-child_fitness)
    T = 400
    replace_prob1 = math.exp(-delta_P1_child / T) / (1 + math.exp(-delta_P1_child / T))
    if random.random() < replace_prob1:
        population[parent1_index][0] = child

    replace_prob2 = math.exp(-delta_P2_child / T) / (1 + math.exp(-delta_P2_child / T))
    if random.random() < replace_prob2:
        population[parent2_index][0] = child


# LAB2-EXTRA:function for niching
def build_similarity_matrix(population):
    similarity_matrix = np.zeros((len(population), len(population)))
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            similarity_matrix[i][j] = hamming_distance(population[i][0], population[j][0])
    return similarity_matrix

# LAB2-EXTRA:function for niching
def niching_with_shared_fitness(population, similarity_matrix, fitnesses):
    niches = []

    for i in range(len(population)):
        flag_found = 0
        for niche in niches:
            for j in range(len(niche)):
                matrix = similarity_matrix[i][j]
                if matrix < SHARING_RADIUS:
                    niche.append(population[i])
                    flag_found = 1
                    break
            if flag_found == 1: break
        if flag_found == 0:
            niches.append([population[i]])
    shared_fitness = calculate_shared_fitness(niches,similarity_matrix,fitnesses)
    return shared_fitness

# LAB2-EXTRA:function for niching
def calculate_shared_fitness(niches,similarity_matrix,fitnesses):
    shared_fitness = []
    for nich in niches:
        for i, individual in enumerate(nich):
            sh = 0
            for j in range(len(nich)):
                if i != j:
                    sh += similarity_matrix[i][j]
            shared_fitness.append(fitnesses[i]/sh)
    return shared_fitness

def Kmean(population,K):
    n = len(population)
    #this array is to save the distance between individual and all centroids
    distances = np.zeros(K)
    #this array is to save for each individual the clusters it belogns to
    clusters = np.zeros(n)
    #choose random centroids as an intial state  for all k clusters
    centroids = [''.join(random.choices(string.printable, k=13)) for i in range(K)]
    #array to check when there is no change in the clusters to stop the algorithm
    last_cluster = np.zeros(n)
    while 1:
        for i in range(n):
            for j in range(K):
                #claculate distance between each individual and all centroids
                distances[j] = hamming_distance(population[i][0], centroids[j])
            #put individual i in cluster that has the minimum distance with its centroid
            clusters[i] = np.argmin(distances)
        #update centroid
        sum = [0] * n
        for i in range(K):
            count = 0
            for j in range(n):
                if clusters[j] == i:
                    count += 1
                    for y in range(13):
                        str = population[j][0]
                        ascii_val = ord(str[y])
                        sum[y] += ascii_val
            if count!=0:
                sum = [int(x / count) for x in sum]
            char_list = []
            for ascii_value in sum:
                if ascii_value != 0 :
                    my_char = chr(ascii_value)
                    char_list.append(my_char)
        str_ = ''.join(char_list)
        centroids[i]= str_

        if np.array_equal(clusters, last_cluster): break

        last_cluster = clusters
    return clusters



def Silhouette(population, clusters, K):
    n = len(population)

    score_cluster = np.zeros(K)
    sum_cluster = 0

    for i in range(n):
        cluster_indices = np.where(clusters == i)[0]
        a = np.zeros(len(cluster_indices))
        for j in range(len(cluster_indices)):
            sum = 0
            for y in range(len(cluster_indices)):
                sum += hamming_distance(population[cluster_indices[j]][0],population[cluster_indices[y]][0])
            if len(cluster_indices) > 1:
                a[j] = sum/(len(cluster_indices)-1)

        b = np.zeros(len(cluster_indices))
        temp = np.zeros(K)
        for j in range(len(cluster_indices)): #for every point in cluster i
            for y in range(K):#sum difference between point j in cluster i and all other clusters
                sum = 0
                if y != i:
                    non_cluster_indices = np.where(clusters == y)[0]
                    for z in range(len(non_cluster_indices)):
                        sum += hamming_distance(population[cluster_indices[j]][0], population[non_cluster_indices[z]][0])
                    if len(non_cluster_indices) > 0:
                        temp[y] = sum/(len(non_cluster_indices)) # b3d ma qarent point balb cluster i m3 kul points cluster m3yn y
            temp[i]=1000000000
            b[j] = min(temp)
            if a[j] == 0 and b[j] == 0:
                score_j = 0
            else:
                score_j = (b[j]-a[j])/max(a[j], b[j])
                sum_cluster += score_j

        if len(cluster_indices) > 0:
            #print(sum_cluster/len(cluster_indices))
            score_cluster[i] = sum_cluster/len(cluster_indices)

    final_score = np.sum(score_cluster)/K
    return final_score


# Run the genetic algorithm and print the result
if __name__=="__main__":
    crossover = input("Enter the number of which type of crossover you want: 0.without crossover 1.Single  2.Two  3.Unifrom: ")
    selection = input("Enter the number of which type of parent selection you want: 0.Random  Selection 1.RWS  2.SUS  3.Tournament Selection:  ")
    aging = input("Enter the number of the option you want: 0.without aging 1.with aging: ")
    mutation_ = input("Enter the number of the option you want: \n0.without mutation \n1.Basic mutation "
                      "\noption number:  ")
    if mutation_ =="1":
        mutation_control = input("Enter the number of the option you want: \n1.None \n2.Non uniform mutation-linear"
                                 " \n3.Non uniform mutation-non_linear \n4.Triggered Hyper Mutation "
                                 "\n5.Adaptive mutation \n6.Self Adaptive \noption number:  ")
    else : mutation_control ="1"
    fitness_type = input("Enter the number of the option you want: 0.bulls-eye heuristic 1.fitness: ")
    niching = input("Enter the number of the option you want: \n1.niching with shared fitness\n 2.clustering \n3.crowding \noption number: ")

    if aging == "1":
        func_type = fitness_age
    elif fitness_type == "0":
        func_type = heuristic
        trigger_fitness = 120
        trigger_avg = 100
    else:
        func_type = fitness
        trigger_fitness = 11
        trigger_avg = 10
    cluster=1
    best_individual, best_fitness = genetic_algorithm(crossover, selection,mutation_,aging,fitness_type,trigger_fitness,trigger_avg,mutation_control,int(niching),pop_size=150, num_genes=13, fitness_func=func_type, max_generations=100)

    print("Best individual:", ''.join(best_individual[0]))
    print("Best fitness:", best_fitness)


