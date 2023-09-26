# geneticAlgorithmProject
![python](https://img.shields.io/badge/Language-Python-pink)

**Introduction:**

This repository contains implementations of genetic algorithms to solve three distinct optimization problems: N-Queens, String Matching, and Bin Packing. It also represents the island model.

**1. N-Queens**

Background:

The N-Queens puzzle is a classic problem where the task is to place N chess queens on an N×N chessboard so that no two queens threaten each other. This means that no two queens can be in the same row, column, or diagonal. The goal is to find all such arrangements for a given N.

**2. String Matching**

Background:

The String Matching problem involves finding a target string within a larger population of strings. The objective is to evolve a population of random strings over successive generations until the target string is generated or a string closely resembling the target is found.

**3. Bin Packing**
   
Background:

The Bin Packing problem is a combinatorial optimization problem where the goal is to pack a set of items with varying sizes into bins with a fixed capacity, minimizing the number of bins used. It is a NP-hard problem with applications in resource allocation, logistics, and task scheduling.

Genetic Algorithm Components:
1. Crossovers, Mutations, and Parent Selection: The genetic algorithm includes various methods for crossovers, mutations, and parent selection to explore the solution space effectively.

2. Aging: Aging is a technique used to simulate natural selection more closely by giving preference to newer solutions. Older individuals in the population have a higher chance of being replaced, promoting genetic diversity and preventing premature convergence.
 
3. Niching: Niching is a method used to maintain diversity within the population by identifying and preserving subpopulations (niches) that are fit but genetically different. This approach helps in exploring multiple peaks in the fitness landscape.

4. Crowding: Crowding is a technique used to maintain population diversity by comparing a new individual primarily with similar individuals (in terms of genotype) in the population. This method ensures that individuals in less crowded regions have a higher chance of survival.

5. Clustering: Clustering involves grouping similar individuals together. This approach helps in maintaining diversity and allows the algorithm to explore different regions of the solution space by focusing on different clusters in the population.

**Island model:**

-islands.py file implements a genetic algorithm to solve two specific optimization problems, represented by function_f and function_g. The algorithm operates on two populations (islands) and employs various genetic operations such as selection, crossover, mutation, and migration to evolve solutions.

-Function F represents an optimization problem where the goal is to find a vector (x, y) such that x^2 + y^2 is minimized, subject to the constraint x^2 + y^2 <= 9.

-Function G represents another optimization problem aiming to find a vector (x, y) that minimizes (x - 5)^2 + (y - 5)^2, subject to the constraint (x - 5)^2 + (y - 5)^2 <= 4.

-Migration occurs between the two islands (populations) at specified intervals. Different selection types (Random, RWS) can be used for selecting migrants. The migration function handles the migration process, ensuring that the immigrants satisfy the problem constraints.

**Running the project:**
 3 files included :
 1. main.py : for string matching problem.
 2. N_queens.py : for N-Queens problem.
 3. updated_bin_packing.py : for bin packing problem.
 4. islands.py : for island model.
    
To run the project run the code then answer the given questions and wait for the results.

