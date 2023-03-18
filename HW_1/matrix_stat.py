import numpy as np


def matrix_stat():
    # Values that needs to be calculated

    determinant = 0
    trace = 0

    # Create the matrix
    matrix = []
    matrix_input = open('input.txt', 'r')
    inp = matrix_input.readlines()
    for i in range(4):
        line = inp[i].split(" ")
        matrix.append(line)
    matrix = np.array(matrix)

    determinant_coefficients = [float(matrix[0, 0]), -float(matrix[0, 1]), float(matrix[0, 2]), -float(matrix[0, 3])]

    matrices_for_coefficients = [matrix[1:4, 1:4],
                                 np.append(matrix[1:4, 0:1], matrix[1:4, 2:4].transpose()).reshape((3, 3)).transpose(),
                                 np.append(matrix[1:4, 0:2].transpose(), matrix[1:4, 3:4]).reshape((3, 3)).transpose(),
                                 matrix[1:4, 0:3]]

    # Calculate the determinant: coefficient(1) * a - coefficient(2) * b + coefficient(3) * c
    for i in range(4):
        a = ((float(matrices_for_coefficients[i][1, 1]) * float(matrices_for_coefficients[i][2, 2])) -
             (float(matrices_for_coefficients[i][1, 2]) * float(matrices_for_coefficients[i][2, 1])))

        b = ((float(matrices_for_coefficients[i][1, 0]) * float(matrices_for_coefficients[i][2, 2])) -
             (float(matrices_for_coefficients[i][1, 2]) * float(matrices_for_coefficients[i][2, 0])))

        c = ((float(matrices_for_coefficients[i][1, 0]) * float(matrices_for_coefficients[i][2, 1])) -
             (float(matrices_for_coefficients[i][1, 1]) * float(matrices_for_coefficients[i][2, 0])))

        calculation = float(determinant_coefficients[i]) * ((float(matrices_for_coefficients[i][0, 0]) * a)
                                                            - (float(matrices_for_coefficients[i][0, 1]) * b)
                                                            + (float(matrices_for_coefficients[i][0, 2]) * c))

        determinant += calculation

    print(round(determinant, 2))
    print(sum([float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2]), float(matrix[3, 3])]))

    matrix_input.close()

if __name__ == '__main__':
    matrix_stat()
