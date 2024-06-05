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
    return roots


def newton_raphson(roots, f, df, epsilon):
    roots_list = []
    for root in roots:
        x = root
        while abs(f(x, [])[0]) > epsilon:
            x -= f(x, [])[0] / df(x)
        roots_list.append(round(x, 4))
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
    return x_1, c


def secant_method(f, variable, point):
    delta = 1e-07
    if variable == "x":
        return (f(point[0] + delta, point[1]) - f(point[0], point[1])) / delta
    if variable == "y":
        return (f(point[0], point[1] + delta) - f(point[0], point[1])) / delta


def newton_method_for_solving_equation_system(f, g, p_0):
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
    return [round(x_1, 4), round(y_1, 4)]


# The "gauss()" function gets as an input a matrix "matrix" and a vector "c" and finds the solutions of the given
# equation systems
def gauss(matrix, c):
    matrix = pivoting(matrix, c)
    n = np.shape(matrix)[0]
    x = c
    for i in range(0, len(matrix[0])):  # Loop over the rows (=i)
        for j in range(i + 1, n):  # Loop over the columns (=j)
            matrix[i][j] = matrix[i][j] / matrix[i][i]  # Step 1: A[i][j] = A[i][j] / A[i][i]
        c[i] = c[i] / matrix[i][i]  # Step 2: C[i] = C[i] / A[i][i]
        matrix[i][i] = 1.0  # Step 3: A[i][i] = 1.0
        for k in range(i + 1, n):  # Loop over the rows (i+1 to n)
            for j in range(i, n):  # Loop over the columns (i to n)
                if j == i:
                    save_aki = matrix[k][i]
                    save_k = k
                matrix[k][j] = matrix[k][j] - save_aki * matrix[i][j]  # Step 4: A[k][j] = A[k][j] - A[k][j] * A[i][j]
            c[save_k] = c[save_k] - save_aki * c[i]  # Step 5: C[k] = C[k] - A[k][i] * C[i]
    x[n - 1] = c[n - 1]
    for i in range(n - 1, -1, -1):  # Swap back
        s = 0
        for j in range(i + 1, n):
            s = s + matrix[i][j] * x[j]  # == SUM (A[i][j] * X[j])
        x[i] = c[i] - s  # Step 6: X[i] = C[i] - SUM (A[i][j] * X[j])

    # x = [round(i, 2) for i in x]  # round for printing

    return x


# The "LU_decomposition()" function gets as an input a matrix "matrix" and a vector "c" and finds the solutions of the given
# # equation systems
def lu_decomposition(matrix, c, print_lu=False):
    solution = []
    u = np.zeros([len(matrix[0]), len(matrix[0])])
    l = np.identity(len(matrix[0]))

    # "Crout" algorithm
    # "Forward elimination"
    for j in range(0, len(matrix[0])):
        for i in range(0, j + 1):
            s = 0
            for k in range(0, i):
                s += l[i][k] * u[k][j]
            u[i][j] = matrix[i][j] - s  # step 1: U[i][j] = A[i][j] - sum( L[i][k] * U[k][j] )
        for i in range(j + 1, len(matrix[0])):
            s = 0
            for k in range(0, j):
                s += l[i][k] * u[k][j]
            l[i][j] = (1 / u[j][j]) * (
                    matrix[i][j] - s)  # step 2: L[i][j] = ( 1/U[j][j] ) *  (A[i][j]- sum( L[i][k] * U[k][j] ) )

    # Now I'm filling "y" vector/ "Backward elimination"

    y = [c[0] / l[0][0]]  # step 1: y_1 = C_1 / L_11
    for i in range(1, len(l[0])):
        s = 0
        for j in range(0, i):
            s += l[i][j] * y[j]
        y.append((1 / l[i][i]) * (c[i] - s))  # step 2: y_i = C_i / L_ii , i=2,3....,n

    # Now I'm filling "X" (=solution) vector:

    for i in range(0, (len(l[0]))):
        solution.append(0.0)
    solution[len(l[0]) - 1] = y[len(l[0]) - 1] / u[len(l[0]) - 1][len(l[0]) - 1]  # step 1: X_n = y_n / U_nn
    for i in range(len(l[0]) - 2, -1, -1):
        s = 0
        for j in range(i, len(l[0])):
            s += u[i][j] * solution[j]
        solution[i] = (1 / u[i][i]) * (y[i] - s)  # step 2: X_i = y_i / U_ii , i=n-1, n-2,.....,1

    solution = [round(x, 4) for x in solution]  # round for printing

    if print_lu:
        print("L:" + "\n" + str(l) + "\n"
                                     "U:" + "\n" + str(u) + "\n")

    return solution


# The "lu_decomposition_lu()" function gets as an input a matrix "matrix" and returns the l and u parts of it.
def lu_decomposition_lu(matrix, print_lu=False):
    u = np.zeros([len(matrix[0]), len(matrix[0])])
    l = np.identity(len(matrix[0]))

    # "Crout" algorithm
    # "Forward elimination"
    for j in range(0, len(matrix[0])):
        for i in range(0, j + 1):
            s = 0
            for k in range(0, i):
                s += l[i][k] * u[k][j]
            u[i][j] = matrix[i][j] - s  # step 1: U[i][j] = A[i][j] - sum( L[i][k] * U[k][j] )
        for i in range(j + 1, len(matrix[0])):
            s = 0
            for k in range(0, j):
                s += l[i][k] * u[k][j]
            l[i][j] = (1 / u[j][j]) * (
                    matrix[i][j] - s)  # step 2: L[i][j] = ( 1/U[j][j] ) *  (A[i][j]- sum( L[i][k] * U[k][j] ) )

    if print_lu:
        print("L:" + "\n" + str(l) + "\n" +"U:" + "\n" + str(u) + "\n")

    return l, u

# The "lu_decomposition_sol()" function gets as an input the l and u parts of a "matrix" and returns the solution x
# AX=C
def lu_decomposition_sol(l, u, c):
    solution = []
    # Now I'm filling "y" vector/ "Backward elimination"

    y = [c[0] / l[0][0]]  # step 1: y_1 = C_1 / L_11
    for i in range(1, len(l[0])):
        s = 0
        for j in range(0, i):
            s += l[i][j] * y[j]
        y.append((1 / l[i][i]) * (c[i] - s))  # step 2: y_i = C_i / L_ii , i=2,3....,n

    # Now I'm filling "X" (=solution) vector:

    for i in range(0, (len(l[0]))):
        solution.append(0.0)
    solution[len(l[0]) - 1] = y[len(l[0]) - 1] / u[len(l[0]) - 1][len(l[0]) - 1]  # step 1: X_n = y_n / U_nn
    for i in range(len(l[0]) - 2, -1, -1):
        s = 0
        for j in range(i, len(l[0])):
            s += u[i][j] * solution[j]
        solution[i] = (1 / u[i][i]) * (y[i] - s)  # step 2: X_i = y_i / U_ii , i=n-1, n-2,.....,1

    solution = [round(x, 4) for x in solution]  # round for printing


    return solution

# The "gauss_seidel()" function gets as an input a matrix "matrix" and a vector "c" and finds the solutions of the given
# # equation systems
def gauss_seidel(matrix, c):
    matrix = pivoting(matrix, c)
    x = np.random.rand(len(c))
    save_previous = np.zeros(len(x))
    check_if_first = False
    iterations = 0
    while (abs((save_previous - x)) > 1e-6).all():
        if check_if_first:
            for i in range(len(x)):
                save_previous[i] = x[i]

        for i in range(len(x)):
            # x[i] = c[i] / matrix[i][i]
            x[i] = c[i]
            for j in range(len(x)):
                if i == j:
                    continue
                # x[i] -= matrix[i][j] * x[j] / matrix[i][i]
                x[i] -= matrix[i][j] * x[j]
            x[i] = x[i] / matrix[i][i]
        if not check_if_first:
            check_if_first = True

        iterations += 1

        if iterations > 50:
            print("Equations are Divergent")
            exit()

    x = [round(i, 2) for i in x]
    return x


def pivoting(matrix, c):
    save_row = 0
    for col in range(len(matrix[0]) - 1):
        save_large = matrix[col][col]
        enter = False
        for row in range(col, len(np.transpose(matrix)[0])):
            if matrix[row][col] > save_large:
                enter = True
                save_large = matrix[row][col]
                save_row = row
        if enter:
            save_row1 = [x for x in matrix[save_row]]
            save_row2 = [x for x in matrix[col]]

            matrix[col] = save_row1
            matrix[save_row] = save_row2

            save_c1 = c[save_row]
            save_c2 = c[col]
            c[save_row] = save_c2
            c[col] = save_c1

    return matrix

