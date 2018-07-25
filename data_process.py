from tqdm import tqdm 
import pickle
import numpy as np

x_data = []; y_data = []
dictionary = []
longest_str = 40

with open("data/training.txt") as f:
    for i in tqdm(f.readlines()):
        line = i.lower().replace(".", "").replace(",", "").split()
        x = line[1:]; y = int(line[0])
        x_vec = []
        for j in x:
            if j not in dictionary:
                dictionary.append(j)
        for k in x:
            x_vec.append(dictionary.index(k))
        while len(x_vec) < longest_str:
            x_vec.append(0)
        x_data.append(x_vec); y_data.append(y)

np.save('data/x.npy', x_data)
np.save('data/y.npy', y_data)
print(len(dictionary))
print(len(x_data))