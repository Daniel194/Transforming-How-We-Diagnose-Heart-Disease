import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('../../result/segmenter/train_result/v1/loss.pickle', 'rb') as f:
    loss_array = pickle.load(f)

    plt.plot(np.arange(0, 10000, 100), loss_array, 'r')
    plt.show()
