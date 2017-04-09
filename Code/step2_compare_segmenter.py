import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('../../result/segmenter/train_result/v1/loss.pickle', 'rb') as f:
    loss_array1 = pickle.load(f)

with open('../../result/segmenter/train_result/v2/loss.pickle', 'rb') as f:
    loss_array2 = pickle.load(f)

print('Min Loss in V1 : ', min(loss_array1))
print('Min Loss in V2 : ', min(loss_array2))

ox = np.arange(0, 10000, 100)

plt.plot(ox, loss_array1, 'b', ox, loss_array2, 'r')
plt.show()
