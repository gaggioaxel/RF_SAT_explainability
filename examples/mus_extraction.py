"""
Example of MUS extraction using the algorithm presented in "On Explaining Random Forests with SAT" by Yacine Izza
and Joao Marques-Silva
"""

from pysat.solvers import Solver

if __name__ == '__main__':
    print('simple infrastructure (add_clause, solve):')
    s1 = Solver(name='g3')
    s1.add_clause([-1, 2, -5])
    s1.add_clause([-2, 3, -6])
    s1.add_clause([-3, 4, -7])
    s1.add_clause([1, -8])
    s1.add_clause([-2, -9])

    if s1.solve():
        print("SAT")
        print(s1.get_model())
    else:
        print("UNSAT")

    asmt = [5, 6, 7, 8, 9]

    i = 0
    while i < len(asmt):
        ts = asmt[:i] + asmt[(i + 1):]
        if s1.solve(assumptions=ts):
            i += 1
        else:
            asmt = ts

    print(asmt)
