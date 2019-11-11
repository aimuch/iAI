import pandas as pd
import matplotlib.pyplot as plt

train_log = pd.read_csv("caffe.INFO.train")
test_log = pd.read_csv("caffe.INFO.test")

_, ax1 = plt.subplots()
ax1.set_title("train loss and test loss")
ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.5)
ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
plt.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(test_log["NumIters"], test_log["acc/top-1"], 'r')
ax2.plot(test_log["NumIters"], test_log["acc/top-5"], 'm')
ax2.set_ylabel('test accuracy')
plt.legend(loc='upper right')

plt.show()

print 'Done.'
