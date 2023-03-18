import numpy as np
import random


# This function will get as an input matrix in .txt file and returns the matrix in np.array type.
# The numbers inside the matrix will be floats.
def create_matrix(matrix_input):
    matrix = []
    matrix_rl = matrix_input.readlines()
    for i in range(len(matrix_rl)):
        line = matrix_rl[i].split(" ")
        line = [float(x) for x in line]
        matrix.append(line)
    matrix = np.array(matrix)

    return matrix


# This function will get as an input vector in .txt file and returns the vector in np.array type.
# The numbers inside the vector will be floats.
def create_vector(vector_input):
    vec = vector_input.readlines()
    vector = []
    for i in range(len(vec)):
        vector.append(float(vec[i]))
    vector = np.array(vector)

    return vector


# This function know how to calculate the determinant and the trace of 4X4 matrix.
# The function gets as an input: matrix is np.array type
# The function output: determinant [np.float()], trace [np.float()], matrix [(np.array())]
# The function gets as an input 'command' and it will return:
# "d" for determinant, "t" for trace, "b" for both (determinant and trace)
def matrix_stat_4x4(matrix, command):
    determinant = 0

    determinant_coefficients = [matrix[0, 0], -matrix[0, 1], matrix[0, 2], -matrix[0, 3]]

    matrices_for_coefficients = [matrix[1:4, 1:4],
                                 np.append(matrix[1:4, 0:1], matrix[1:4, 2:4].transpose()).reshape(
                                     (3, 3)).transpose(),
                                 np.append(matrix[1:4, 0:2].transpose(), matrix[1:4, 3:4]).reshape(
                                     (3, 3)).transpose(),
                                 matrix[1:4, 0:3]]

    # Calculate the determinant: coefficient(1) * a - coefficient(2) * b + coefficient(3) * c
    # a, b and c are 2X2 matrices
    for i in range(4):
        a = ((matrices_for_coefficients[i][1, 1] * matrices_for_coefficients[i][2, 2]) -
             (matrices_for_coefficients[i][1, 2] * matrices_for_coefficients[i][2, 1]))

        b = ((matrices_for_coefficients[i][1, 0] * matrices_for_coefficients[i][2, 2]) -
             (matrices_for_coefficients[i][1, 2] * matrices_for_coefficients[i][2, 0]))

        c = ((matrices_for_coefficients[i][1, 0] * matrices_for_coefficients[i][2, 1]) -
             (matrices_for_coefficients[i][1, 1] * matrices_for_coefficients[i][2, 0]))

        calculation = determinant_coefficients[i] * ((matrices_for_coefficients[i][0, 0] * a)
                                                     - (matrices_for_coefficients[i][0, 1] * b) + (
                                                             matrices_for_coefficients[i][0, 2] * c))

        determinant += calculation

    trace = sum([matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[3, 3]])

    # "d" for determinant, "t" for trace, "b" for both (determinant and trace)

    if command == "d":
        return determinant

    if command == "t":
        return trace

    if command == "b":
        return determinant, trace


# This function know how to calculate the product between matrix and a vector
# This function gets as an input matrix and vector, both are np.array type.
# The output of this function is 'solution' np.array type, round by 2 numbers.
def mat_vec_mul(matrix, vector):
    if len(matrix[0]) != len(vector):
        print("The numbers of columns in the matrix must be the same as the vector dimension")
        exit()
    solution = []
    for i in range(len(matrix)):
        row_sum = 0
        for j in range(len(vector)):
            row_sum += matrix[i, j] * vector[j]
        solution.append(round(row_sum, 2))
    return np.array(solution)


# This function finds the root of a polynomial 'f' between range: (a, b)
# This function will get as an input epsilon (as a threshold)
# This function will return the 'closest' value to the root: c and the number of loops: n
def bisection(a_0, b_0, epsilon, f):
    roots = []
    loops = []
    order_of_poly = f(a_0, roots)[1]
    while len(roots) != order_of_poly:
        a = a_0
        b = b_0
        n = 0
        u = f(a, roots)[0]
        v = f(b, roots)[0]
        c = 0.5 * (a + b)
        w = f(c, roots)[0]

        while abs(w) > epsilon:
            while w * u > 0 and w * v > 0:
                c = random.uniform(-5, 3)
                w = f(c, roots)[0]
            if w * u < 0:
                b = c
                v = w
            elif w * v < 0:
                a = c
                u = w
            n += 1
            c = 0.5 * (a + b)
            w = f(c, roots)[0]
        roots.append(c)
        loops.append(n)
    return [roots, loops]


def newton_raphson(roots, f, df, epsilon):
    roots_list = []
    for root in roots:
        x = root
        while abs(f(x, [])[0]) > epsilon:
            x = x - f(x, [])[0] / df(x)
        roots_list.append(x)
    return roots_list


# The "synthetic_division()" function given list of "Polynomial coefficients", the value of Polynomial order
# root guess and epsilon[=1e-07] and return new root (that it founds).
def synthetic_division(b, n, x_0, epsilon):
    n += 1
    x_1 = x_0 + epsilon * 2
    check_if_first = False
    while abs(x_1 - x_0) > epsilon:
        if check_if_first:
            x_0 = x_1
        c = [None] * (n - 1)
        c[n - 2] = b[n - 1]

        for j in range(n - 3, -1, -1):
            c[j] = b[j + 1] + c[j + 1] * x_0

        d = [None] * (n - 2)
        d[n - 3] = c[n - 2]

        for j in range(n - 4, -1, -1):
            d[j] = c[j + 1] + d[j + 1] * x_0

        r_0 = b[0] + x_0 * c[0]
        r_1 = c[0] + x_0 * d[0]

        x_1 = x_0 - r_0 / r_1
        check_if_first = True
    return [x_1, c]


def secant_method(f, variable, point):
    delta = 1e-07
    if variable == "x":
        return (f(point[0] + delta, point[1]) - f(point[0], point[1])) / delta
    if variable == "y":
        return (f(point[0], point[1] + delta) - f(point[0], point[1])) / delta


def newton_method_for_solving_equatuion_system(f, g, p_0):
    x_0 = p_0[0]
    y_0 = p_0[1]

    epsilon = 1e-07
    x_1 = x_0 + epsilon * 2
    y_1 = y_0 + epsilon * 2
    check_if_first = False
    while abs(x_1 - x_0) > epsilon and abs(y_1 - y_0) > epsilon:
        if check_if_first:
            x_0 = x_1
            y_0 = y_1
        df_dx = secant_method(f, "x", [x_0, y_0])
        df_dy = secant_method(f, "y", [x_0, y_0])
        dg_dx = secant_method(g, "x", [x_0, y_0])
        dg_dy = secant_method(g, "y", [x_0, y_0])

        x_1 = x_0 - (((f(x_0, y_0) * dg_dy) - g(x_0, y_0) * df_dy) / (df_dx * dg_dy - dg_dx * df_dy))
        y_1 = y_0 - (((g(x_0, y_0) * df_dx) - f(x_0, y_0) * dg_dx) / (df_dx * dg_dy - dg_dx * df_dy))
        check_if_first = True
    return [x_1, y_1]
