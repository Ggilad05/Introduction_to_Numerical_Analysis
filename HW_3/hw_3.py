import numpy as np
from tools import gauss, lu_decomposition, gauss_seidel, mat_vec_mul, lu_decomposition_lu,\
    lu_decomposition_sol
import timeit


def task_1():
    # matrix = np.array([[3.0, -3.0, 2.0, -4.0],
    #                    [-2.0, -1.0, 3.0, -1.0],
    #                    [5.0, -2.0, -3.0, 2.0],
    #                    [-2.0, 4.0, 1.0, 2.0]])
    matrix = np.array([[-9.0, 9.0, -7.0],
                       [17.0, -4.0, -5.0],
                       [4.0, -7.0, 23.0]])

    # solution_vector = np.array([7.9, -12.5, 18.0, -8.1])
    solution_vector = np.array([1500.0, 2560.0, 890.0])

    print("Hello," + "\n" + "You have three options for calculating the results of the given system of equations: "
          + "\n" + "Enter 1 for Solution through Gauss  " + "\n" + "Enter 2 for Solution through LU decomposition  "
          + "\n" + "Enter 3 for Solution through Gauss-seidel  " + "\n")
    choice = int(input("Answer :"))
    if choice == 1:
        print("Solutions Gauss: " + "\n" + str(gauss(matrix, solution_vector)) + "\n")

        gauss_efficiency = timeit.timeit(lambda: gauss(matrix, solution_vector), number=1000)
        print("gauss efficiency [sec]: "+ str(gauss_efficiency))
    if choice == 2:
        print("Solutions through LU decomposition: " + "\n" + str(lu_decomposition(matrix, solution_vector)) + "\n")
        lu_decomposition_efficiency = timeit.timeit(lambda: lu_decomposition(matrix, solution_vector), number=1000)
        print("gauss efficiency [sec]: " + str(lu_decomposition_efficiency))
    if choice == 3:
        print("Solutions Gauss-seidel: " + "\n" + str(gauss_seidel(matrix, solution_vector)))
        gauss_seidel_efficiency = timeit.timeit(lambda: gauss_seidel(matrix, solution_vector), number=1000)
        print("gauss efficiency [sec]: " + str(gauss_seidel_efficiency))



def task_2():
    matrix = np.array([[4.0, 8.0, 4.0, 0.0],
                       [1.0, 4.0, 7.0, 2.0],
                       [1.0, 5.0, 4.0, -3.0],
                       [1.0, 3.0, 0.0, -2.0]])

    inverse_matrix = []


    # Create identity matrix
    identity_matrix = np.identity(len(matrix))

    l, u = lu_decomposition_lu(matrix, print_lu=True)
    for row in identity_matrix:
        inverse_matrix.append(lu_decomposition_sol(l, u, row))

    inverse_matrix = np.array(np.transpose(inverse_matrix))

    print("inverse_matrix")
    print(inverse_matrix)
    print(" ")

    identity = []
    inverse_matrix = np.transpose(inverse_matrix)
    for row in inverse_matrix:
        identity.append(mat_vec_mul(matrix, row))

    identity = np.array(np.transpose(identity))

    print("Identity matrix")
    print(identity)


if __name__ == '__main__':
    task_1()
    # task_2()
