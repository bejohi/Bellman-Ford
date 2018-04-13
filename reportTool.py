import matplotlib.pyplot as plt
import numpy as np

SEQU_KEYWORD = "sequ"


def read_csv(path: str):
    with open(path) as f:
        return f.readlines()


def plot_matrix_task1and2(csv_arr: list):
    sequ_x = []
    sequ_y = []
    parallel1_x = []
    parallel1_y = []
    parallel2_x = []
    parallel2_y = []
    parallel4_x = []
    parallel4_y = []
    parallel8_x = []
    parallel8_y = []
    parallel16_x = []
    parallel16_y = []
    parallel32_x = []
    parallel32_y = []
    parallel64_x = []
    parallel64_y = []

    for line in csv_arr:
        splited_string = line.split(";")
        if splited_string[0] == SEQU_KEYWORD:
            sequ_y.append(splited_string[1])
            sequ_x.append(splited_string[2])
        else:
            if splited_string[3] == "1":
                parallel1_y.append(splited_string[1])
                parallel1_x.append(splited_string[2] * 100)
            if splited_string[3] == "2":
                parallel2_y.append(splited_string[1])
                parallel2_x.append(splited_string[2] * 100)
            if splited_string[3] == "4":
                parallel4_y.append(splited_string[1])
                parallel4_x.append(splited_string[2] * 100)
            if splited_string[3] == "8":
                parallel8_y.append(splited_string[1])
                parallel8_x.append(splited_string[2] * 100)
            if splited_string[3] == "16":
                parallel16_y.append(splited_string[1])
                parallel16_x.append(splited_string[2] * 100)
            if splited_string[3] == "32":
                parallel32_y.append(splited_string[1])
                parallel32_x.append(splited_string[2] * 100)
            if splited_string[3] == "36":
                parallel64_y.append(splited_string[1])
                parallel64_x.append(splited_string[2] * 100)

    plt.plot(sequ_x, sequ_y, label="Sequential")
    plt.plot(parallel1_x, parallel1_y, label="Parallel T=1")
    '''plt.plot(parallel2_x, parallel2_y, label="Parallel T=2")
    plt.plot(parallel4_x, parallel4_y, label="Parallel T=4")
    plt.plot(parallel8_x, parallel8_y, label="Parallel T=8")
    plt.plot(parallel16_x, parallel16_y, label="Parallel T=16")
    plt.plot(parallel32_x, parallel32_y, label="Parallel T=32")'''
    plt.plot(parallel64_x, parallel64_y, label="Parallel T=36")

    plt.legend()
    plt.show()


def manual_plot_gpu():
    # Sequential:

    plt.plot([50, 100, 500, 1000],
             [0.150957, 0.078337, 0.054403, 0.054730],
             label="Threads per Block: 128")
    plt.plot([50, 100, 500, 1000],
             [0.139761, 0.089167, 0.056434, 0.047035],
             label="Threads per Block: 256")
    plt.plot([50, 100, 500, 1000],
             [0.115838, 0.072363, 0.047787, 0.049288],
             label="Threads per Block: 512")
    plt.plot([50, 100, 500, 1000],
             [0.098102, 0.068852, 0.050519, 0.048924],
             label="Threads per Block: 1024")



    plt.legend()
    plt.show()

def manual_plot_cpu():
    plt.plot([10, 100, 1000, 2000, 4000, 8000, 10000],
             [0.000001, 0.000130, 0.010822, 0.038453, 0.155150, 0.576866, 1.062690],
             label="Sequential")

    plt.plot([10, 100, 1000, 2000, 4000, 8000, 10000],
             [0.000016, 0.000127, 0.010044, 0.035438, 0.144681, 0.535903, 1.016088],
             label="Parallel T=1")

    plt.plot([10, 100, 1000, 2000, 4000, 8000, 10000],
             [0.000091, 0.003038, 0.007924, 0.023048, 0.116962, 0.304267, 0.433818],
             label="Parallel T=4")

    plt.plot([10, 100, 1000, 2000, 4000, 8000, 10000],
             [0.003561, 0.000517, 0.009587, 0.011617, 0.038583, 0.094509, 0.132887],
             label="Parallel T=36")

    plt.legend()
    plt.show()


plt.show()

if __name__ == "__main__":
    manual_plot_cpu()
    # manual_plot_gpu()
