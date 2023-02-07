import numpy as np

def process_data(dir):

    data = np.loadtxt(dir, delimiter=",", dtype=str)
    var_names = data[0]
    var_names = [var_name.replace('"', "") for var_name in var_names]
    # print("numpy var_names: ", var_names)
    data = data[1:]
    data = data.astype(float)

    var_to_indx = {var_name:indx for indx, var_name in enumerate(var_names)}
    var_to_r = {var_name:int(np.amax(data[:, i])) for i, var_name in enumerate(var_names)}

    return (var_names, var_to_indx, var_to_r), data


if __name__ == "__main__":
    process_data("example/example_trial.csv")
                