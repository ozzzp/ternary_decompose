from sympy import *

N = 5
K = 3

def run(i, j):
    if i == j:
        return 1
    if i + 1 == j:
        return -1
    return 0
M = Matrix([[run(i, j) for j in range(N)] for i in range(N)])
d = diag(*[symbols('x_{}'.format(i+1)) for i in range(N)])

def run(i, j):
    if j == K-1:
        a = -1
    else:
        a = 0

    if  i == j:
        b = 1
    else:
        b = 0
    return a + b

T = Matrix([[run(i, j) for j in range(N)] for i in range(N)])
B = T * (d ** -1) * (M.T ** -1)
base = Matrix([[symbols('x_{}'.format(i+1)) for i in range(N)]]).T
x = M.T * base
a = Matrix([[1 if i + 1 <= K else 0 for i in range(N)]]).T / symbols('x_{}'.format(K))
#B(B.T * u + a)
u = d * (M * (M.T * base))
u = simplify(u)
x2 = B.T * u / ((x.T * x)[0, 0]) + a
