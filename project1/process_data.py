import numpy as np

def process_data(dir):

    data = np.loadtxt(dir, delimiter=",", dtype=str)
    var_names = data[0]
    data = data[1:]
    data = data.astype(float)
    # vars= [(var_name, i, np.amax(data[:, i])) for i, var_name in enumerate(variable_names)]
    # print(data)
    var_to_indx = {var_name:indx for indx, var_name in enumerate(var_names)}
    var_to_r = {var_name:int(np.amax(data[:, i])) for i, var_name in enumerate(var_names)}

    # print(var_to_indx)
    # print()
    # print(var_to_r)
    return (var_names, var_to_indx, var_to_r), data


if __name__ == "__main__":
    process_data("example/example_trial.csv")
                