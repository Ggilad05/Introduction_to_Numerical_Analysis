import numpy as np

def mat_vec_mul():
    # Create the matrix
    matrix = []
    mat_input = open('mat_input.txt', 'r')
    inp = mat_input.readlines()
    for i in range(4):
        line = inp[i].split(" ")
        line = [float(x) for x in line]
        matrix.append(line)
    matrix = np.array(matrix)

    # Create the vector
    vector = []
    vec = open('vec_input.txt', 'r')
    vec = vec.readlines()
    for i in range(len(vec)):
        vector.append(float(vec[i]))
    vector = np.array(vector)

    # Multiplying a matrix by a vector
    solution = [round(matrix[0, 0] * vector[0] + matrix[0, 1] * vector[1] + matrix[0, 2] * vector[2], 2),
                round(matrix[1, 0] * vector[0] + matrix[1, 1] * vector[1] + matrix[1, 2] * vector[2], 2),
                round(matrix[2, 0] * vector[0] + matrix[2, 1] * vector[1] + matrix[2, 2] * vector[2], 2),
                round(matrix[3, 0] * vector[0] + matrix[3, 1] * vector[1] + matrix[3, 2] * vector[2], 2)]

    solution = np.array(solution)

    # Write the solution into mvm_out.txt
    mvm_out = open("mvm_out.txt", 'w')
    str_solution = [str(x) for x in solution]
    for i in range(3):
        mvm_out.write(str_solution[i] + " ")
    print(solution)
    mat_input.close()
    mvm_out.close()

if __name__ == '__main__':
    mat_vec_mul()
