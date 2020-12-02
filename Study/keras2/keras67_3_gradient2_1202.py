import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6

gradient = lambda x: 2*x -4

x0            = 0.0
MaxIter       = 31
learning_rate = 0.1

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0,x0,f(x0)))

for i in range(MaxIter):
    x1 = x0 - learning_rate * gradient(x0) # 0 - 0.25 * -4 = 1
    x0 = x1

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1,x0,f(x0)))
# l_r이 0.25일 경우 10
# l_r이 0.1일 경우 31
