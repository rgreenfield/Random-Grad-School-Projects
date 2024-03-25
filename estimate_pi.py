import random
import numpy as np

# global variables
n_trials = int(100_000)


def pi_calculation_random():
    
    circle_points = 0
    square_points = 0

    for _ in range(n_trials):

        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x**2 + y**2 <= 1:
            circle_points += 1
        else:
            square_points += 1

    return circle_points / square_points
def main():
    
    pi = pi_calculation_random()


    print("Final estimation of pi = ", pi)

if __name__ == '__main__':
    main()

