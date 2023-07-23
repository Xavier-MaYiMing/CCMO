### CCMO: A coevolutionary constrained multiobjective optimization (CCMO) framework

##### Reference: Tian Y, Zhang T, Xiao J, et al. A coevolutionary framework for constrained multiobjective optimization problems[J]. IEEE Transactions on Evolutionary Computation, 2020, 25(1): 102-116.

##### The CCMO belongs to the category of multi-objective evolutionary algorithms (MOEAs). CCMO is a powerful algorithm to solve the constrained multiobjective optimization (CMO) problems.

| Variables   | Meaning                                              |
| ----------- | ---------------------------------------------------- |
| npop        | Population size                                      |
| iter        | Iteration number                                     |
| lb          | Lower bound                                          |
| ub          | Upper bound                                          |
| pc          | Crossover probability (default = 1)                  |
| pm          | Mutation probability (default = 1)                   |
| eta_c       | Spread factor distribution index (default = 20)      |
| eta_m       | Perturbance factor distribution index (default = 20) |
| nvar        | The dimension of decision space                      |
| pop1        | Population1                                          |
| pop2        | Population2                                          |
| objs1       | Objectives1                                          |
| objs2       | Objective2                                           |
| CV1         | Constraint violation1                                |
| CV2         | Constraint violation2                                |
| F1          | Fitness1                                             |
| F2          | Fitness2                                             |
| mating_pool | Mating pool                                          |
| offspring   | Offspring                                            |
| pf          | Pareto front                                         |



#### Test problem: MW11

##### Reference: Ma Z, Wang Y. Evolutionary constrained multiobjective optimization: Test suite construction and performance comparisons[J]. IEEE Transactions on Evolutionary Computation, 2019, 23(6): 972-986.

Define $m$ is the dimension of objective space, and $n$ is the dimension of decistion space.



$$
\begin{aligned}
&g3(x)=1+\sum_{i=m}^n2(x_i+(x_{i-1}-0.5)^2-1)^2\\
&\left\{
\begin{aligned}
&f_3(x)=\sqrt{2}g_3x_1\\
&f_2(x)=g_3\sqrt{2-(f_1/g_3)^2}\\
&\text{subject to}\\
&c_1(x)=(3-f_1^2-f_2)(3-2f_1^2-f_2)\geq0\\
&c_2(x)=(3-0.625f_1^2-f_2)(3-7f_1^2-f_2)\leq0\\
&c_3(x)=(1.62-0.18f_1^2-f_2)(1.125-0.125f_1^2-f_2)\geq0\\
&c_4(x)=(2.07-0.23f_1^2-f_2)(0.63-0.07f_1^2-f_2)\leq0\\
&0\leq x_i\leq 1, \quad i=1,\cdots,n
\end{aligned}
\right.
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 500, np.array([0] * 15), np.array([1] * 15))
```

##### Output:

![Pareto front](/Users/xavier/Desktop/Xavier Ma/个人算法主页/CCMO/Pareto front.png)

```python
Iteration 20 completed.
Iteration 40 completed.
Iteration 60 completed.
Iteration 80 completed.
Iteration 100 completed.
Iteration 120 completed.
Iteration 140 completed.
Iteration 160 completed.
Iteration 180 completed.
Iteration 200 completed.
Iteration 220 completed.
Iteration 240 completed.
Iteration 260 completed.
Iteration 280 completed.
Iteration 300 completed.
Iteration 320 completed.
Iteration 340 completed.
Iteration 360 completed.
Iteration 380 completed.
Iteration 400 completed.
Iteration 420 completed.
Iteration 440 completed.
Iteration 460 completed.
Iteration 480 completed.
Iteration 500 completed.
```

