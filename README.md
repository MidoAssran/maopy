# Multi-Agent Optimization in Python (maopy)
#### *Library of distributed convex optimization algorithms implemented in Python using MPI.

## Introduction
Welcome to **_maopy_**, the multi_agent_optimization package. The module contains several distributed optimization algorithms and distributed averaging (consensus) algorithms.

## Examples
Examples (demos) of each algorithm are provided in the main method at the bottom of each respective class. To run any of the demos, depending on your MPI distribution, execute one of the following command line instructions:
```bash
mpirun -np $(num_processes) python $(algorithm_name)
```
or
```bash
mipexec -n $(num_processes_variable) python $(algorithm_name)
```
For example, to run the push-sum consensus demo in the *push_sum_gossip_averaging* file on 5-processors, one would type
```bash
mipexec -n 5 python push_sum_gossip_averaging
```
Notice that we **did not include the file suffix** ('.py') in the command, since the path is already defined relative to the module.

# Enjoy!
