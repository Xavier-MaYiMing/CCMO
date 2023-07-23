#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/19 22:38
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : CCMO.py
# @Statement : A coevolutionary constrained multiobjective optimization (CCMO) framework
# @Reference : Tian Y, Zhang T, Xiao J, et al. A coevolutionary framework for constrained multiobjective optimization problems[J]. IEEE Transactions on Evolutionary Computation, 2020, 25(1): 102-116.
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def g3(x, nobj, nvar):
    return 1 + sum([2 * (x[:, i] + (x[:, i - 1] - 0.5) ** 2 - 1) ** 2 for i in range(nobj - 1, nvar)])


def cal_obj(x, npop, nvar, nobj=2):
    # MW11
    # Reference: Ma Z, Wang Y. Evolutionary constrained multiobjective optimization: Test suite construction and performance comparisons[J]. IEEE Transactions on Evolutionary Computation, 2019, 23(6): 972-986.
    temp = g3(x, nobj, nvar)
    f1 = temp * x[:, 0] * np.sqrt(1.9999)
    f2 = temp * np.sqrt(2 - (f1 / temp) ** 2)
    flag1 = (3 - f1 ** 2 - f2) * (3 - 2 * f1 ** 2 - f2) >= 0
    flag2 = (3 - 0.625 * f1 ** 2 - f2) * (3 - 7 * f1 ** 2 - f2) <= 0
    flag3 = (1.62 - 0.18 * f1 ** 2 - f2) * (1.125 - 0.125 * f1 ** 2 - f2) >= 0
    flag4 = (2.07 - 0.23 * f1 ** 2 - f2) * (0.63 - 0.07 * f1 ** 2 - f2) <= 0
    CV = np.sum((~flag1, ~flag2, ~flag3, ~flag4), axis=0)
    return np.concatenate((f1.reshape(npop, 1), f2.reshape(npop, 1)), axis=1), CV


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def fitness(objs, CV):
    # calculate the fitness value
    npop = objs.shape[0]
    S = np.zeros(npop, dtype=int)  # the strength value
    R = np.zeros(npop, dtype=int)  # the raw fitness
    dom = np.full((npop, npop), False)  # domination matrix
    for i in range(npop - 1):
        for j in range(i, npop):
            if CV[i] < CV[j]:
                S[i] += 1
                dom[i, j] = True
            elif CV[i] > CV[j]:
                S[j] += 1
                dom[j, i] = True
            elif dominates(objs[i], objs[j]):
                S[i] += 1
                dom[i, j] = True
            elif dominates(objs[j], objs[i]):
                S[j] += 1
                dom[j, i] = True
    for i in range(npop):
        R[i] = np.sum(S[dom[:, i]])
    sigma = squareform(pdist(objs, metric='seuclidean'), force='no', checks=True)
    sigma_K = np.sort(sigma)[:, int(np.sqrt(npop))]  # the K-th shortest distance
    D = 1 / (sigma_K + 2)  # density
    F = R + D  # fitness
    return sigma, F


def selection(pop, F, pc, k=2):
    # binary tournament selection
    (npop, dim) = pop.shape
    nm = int(npop * pc)
    nm = nm if nm % 2 == 0 else nm + 1
    mating_pool = np.zeros((nm, dim))
    for i in range(nm):
        selections = np.random.choice(npop, k, replace=True)
        ind = selections[np.argmin(F[selections])]
        mating_pool[i] = pop[ind]
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, dim) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, dim))
    mu = np.random.random((nm, dim))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, pm, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, dim)) < pm / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def environmental_selection(pop, npop, objs, CV):
    # environmental selection
    sigma, F = fitness(objs, CV)
    index = np.where(F < 1)[0]
    if len(index) <= npop:
        rank = np.argsort(F)[: npop]
        return pop[rank], objs[rank], CV[rank], F[rank]
    pop = pop[index]
    objs = objs[index]
    CV = CV[index]
    sigma = sigma[index][:, index]
    F = F[index]
    eye = np.arange(len(sigma))
    sigma[eye, eye] = np.inf
    delete = np.full(len(index), False)
    while np.sum(delete) < len(index) - npop:
        remain = np.where(~delete)[0]
        temp = np.sort(sigma[remain][:, remain])
        delete[remain[np.argmin(temp[:, 0])]] = True
    remain = np.where(~delete)[0]
    return pop[remain], objs[remain], CV[remain], F[remain]


def main(npop, iter, lb, ub, pc=1, pm=1, eta_c=20, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: upper bound
    :param ub: lower bound
    :param pc: crossover probability (default = 1)
    :param pm: mutation probability (default = 1)
    :param eta_c: spread factor distribution index (default = 20)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop1 = np.random.uniform(lb, ub, (npop, nvar))  # population1
    pop2 = np.random.uniform(lb, ub, (npop, nvar))  # population2
    objs1, CV1 = cal_obj(pop1, npop, nvar)  # objectives1, constraint violation1
    objs2, CV2 = cal_obj(pop2, npop, nvar)  # objectives2, constraint violation2
    F1 = fitness(objs1, CV1)[1]  # fitness1
    F2 = fitness(objs2, np.zeros(npop))[1]  # fitness2

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 20 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Selection + crossover + mutation
        mating_pool1 = selection(pop1, F1, pc)
        offspring1 = crossover(mating_pool1, lb, ub, eta_c)
        offspring1 = mutation(offspring1, lb, ub, pm, eta_m)
        mating_pool2 = selection(pop2, F2, pc)
        offspring2 = crossover(mating_pool2, lb, ub, eta_c)
        offspring2 = mutation(offspring2, lb, ub, pm, eta_m)
        off_objs1, off_CV1 = cal_obj(offspring1, npop, nvar)
        off_objs2, off_CV2 = cal_obj(offspring2, npop, nvar)
        pop1 = np.concatenate((pop1, offspring1, offspring2), axis=0)
        pop2 = np.concatenate((pop2, offspring1, offspring2), axis=0)
        objs1 = np.concatenate((objs1, off_objs1, off_objs2), axis=0)
        objs2 = np.concatenate((objs2, off_objs1, off_objs2), axis=0)
        off_CV1 = np.concatenate((CV1, off_CV1, off_CV2), axis=0)
        off_CV2 = np.zeros(pop1.shape[0])

        # Step 2.2. Environmental selection
        pop1, objs1, CV1, F1 = environmental_selection(pop1, npop, objs1, off_CV1)
        pop2, objs2, CV2, F2 = environmental_selection(pop2, npop, objs2, off_CV2)

    # Step 3. Sort the results
    CV2 = cal_obj(pop2, npop, nvar)[1]
    objs1 = objs1[np.where(CV1 == 0)[0]]
    objs2 = objs2[np.where(CV2 == 0)[0]]
    objs = np.concatenate((objs1, objs2), axis=0)
    F = fitness(objs, np.zeros(objs.shape[0]))[1]
    pf = objs[np.where(F < 1)[0]]
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    plt.scatter(x, y)
    plt.xlabel('objective 1')
    plt.ylabel('objective 2')
    plt.title('The Pareto front of MW11')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 500, np.array([0] * 15), np.array([1] * 15))
