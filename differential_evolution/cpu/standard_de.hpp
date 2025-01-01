#ifndef STANDARD_DE_H
#define STANDARD_DE_H

#include <stdlib.h>
#include <algorithm>

template <class T>
class DifferentialEvolution 
{
private: 
    int NP, D; 
    float CR, F, lower_bound, upper_bound;
    T *population, *fitness, *offspring, *offspring_fitness; 
    T rnd_uni();

public: 
    DifferentialEvolution(int NP, int D, float CR, float F, float lower_bound, float upper_bound);
    ~DifferentialEvolution();
    void initialise_population();
    T* get_population_ptr() { return population; }
    T* get_offspring_ptr() { return offspring; }
    T* get_fitness_ptr() { return fitness; }
    T* get_offspring_fitness_ptr() { return offspring_fitness; }
    void generate_offspring();
    void replacement();
    T get_best();
};

template <class T>
DifferentialEvolution<T>::DifferentialEvolution(int NP, int D, float CR, float F, float lower_bound, float upper_bound) :
    NP(NP), D(D), CR(CR), F(F), lower_bound(lower_bound), upper_bound(upper_bound)
{
    population = new T[NP * D]; 
    fitness = new T[NP]; 
    offspring = new T[NP * D]; 
    offspring_fitness = new T[NP]; 
}

template <class T>
DifferentialEvolution<T>::~DifferentialEvolution()
{
    delete[] population; 
    delete[] fitness; 
    delete[] offspring; 
    delete[] offspring_fitness; 
}

template <class T>
void DifferentialEvolution<T>::initialise_population()
{
    for (int i = 0; i < NP; i++)
        for (int j = 0; j < D; j++)
            population[i * D + j] = rnd_uni() * (upper_bound - lower_bound) + lower_bound; 
}

template <class T>
void DifferentialEvolution<T>::generate_offspring()
{
    for (int i = 0; i < NP; i++)
    {
        int r1, r2, r3; 
        do { r1 = rand() % NP; } while (r1 == i); 
        do { r2 = rand() % NP; } while (r2 == i || r2 == r1); 
        do { r3 = rand() % NP; } while (r3 == i || r3 == r1 || r3 == r2); 
        for (int j = 0; j < D; j++)
            if (rnd_uni() < CR)
            {
                offspring[i * D + j] = population[r1 * D + j] + F * (population[r2 * D + j] - population[r3 * D + j]); 
                // if the new value is out of bounds, then reinitialise
                if (offspring[i * D + j] < lower_bound || offspring[i * D + j] > upper_bound)
                    offspring[i * D + j] = rnd_uni() * (upper_bound - lower_bound) + lower_bound;
            }
            else
                offspring[i * D + j] = population[i * D + j]; 
    }
}

template <class T>
void DifferentialEvolution<T>::replacement()
{
    for (int i = 0; i < NP; i++)
        if (offspring_fitness[i] < fitness[i])
        {
            for (int j = 0; j < D; j++)
                population[i * D + j] = offspring[i * D + j]; 
            fitness[i] = offspring_fitness[i]; 
        }
}

template <class T>
T DifferentialEvolution<T>::get_best()
{
    return *std::min_element(fitness, fitness + NP);
}


template <class T> 
T DifferentialEvolution<T>::rnd_uni()
{
    return ((T)rand()) / RAND_MAX;
}

#endif 