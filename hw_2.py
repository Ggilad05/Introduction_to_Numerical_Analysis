from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


from Numerucal_Analysis_Tools import bisection, newton_raphson, synthetic_division,\
    newton_method_for_solving_equatuion_system
import random
import matplotlib.pyplot as plt
import numpy as np


# This function returns the value of f(x) and the order of the polynom
# This function reduce polynomial order after each root find (for finding all the roots)
def f_task1(x, roots_list):
    if not roots_list:
        return [(x ** 4) + 2 * (x ** 3) - 7 * (x ** 2) + 3, 4]

    f = (x ** 4) + 2 * (x ** 3) - 7 * (x ** 2) + 3
    for i in range(len(roots_list)):
        f = f / (x - roots_list[i])
    return [f, 4]


# Derivative of function: 'f_task1'
def df_task1(x):
    return 4 * (x ** 3) + 6 * (x ** 2) - 14 * x



def f(x, y):
    return 4 * (y ** 2) + 4 * y - 52 * x - 19


def g(x, y):
    return 169 * (x ** 2) + 3 * (y ** 2) - 111 * x - 10 * y


def task_1():
    # Solution using bisection method as initial 'guess' for later, final solution using Newton - Raphson method
    a_0 = -5.0
    b_0 = 3.0
    epsilon_bisection = 0.2
    bisection_solution = bisection(a_0, b_0, epsilon_bisection, f_task1)
    epsilon_newton_raphson = 1e-4
    newton_raphson_solution = newton_raphson(bisection_solution[0], f_task1, df_task1, epsilon_newton_raphson)
    print(newton_raphson_solution)

    # Solution using synthetic division
    order_of_polynom = f_task1(1, [])[1]
    epsilon_synthetic_division = 1e-07
    f_coefficents = [3, 0, -7, 2, 1]
    [root, h_coefficents] = synthetic_division(f_coefficents, order_of_polynom, random.uniform(-5, 3),
                                               epsilon_synthetic_division)
    list_of_roots = [root]
    order_of_polynom -= 1

    while len(list_of_roots) != 3:
        [root, h_coefficents] = synthetic_division(h_coefficents, order_of_polynom, root,
                                                   epsilon_synthetic_division)
        list_of_roots.append(root)
        order_of_polynom -= 1
    list_of_roots.append(-h_coefficents[0])
    print(list_of_roots)

def task_2():
    initial_point = [-0.01, -0.01]
    # solution_1 = newton_method_for_solving_equatuion_system(f, g, initial_point)
    initial_point = [0.5, 0.5]
    solution_2 = newton_method_for_solving_equatuion_system(f, g, initial_point)

    print(solution_2)

def task_3():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Make data.
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-5, 5, 0.25)
    x, y = np.meshgrid(x, y)
    z = np.sin(4 * y) * np.cos(0.5 * x)

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()



if __name__ == '__main__':
    # task_1()
    # task_2()
    task_3()
