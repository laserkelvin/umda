import numpy as np

with open("properties.txt", "r") as read_file:
    contents = read_file.readlines()

is_opt = list()
contents = filter(lambda x: "structure" in x, contents)
is_opt = list(map(lambda x: "MM" in x, contents))

np.save("optimized_mask.npy", np.asarray(is_opt))
