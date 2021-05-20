import numpy as np
import matplotlib.pyplot as plt
import neural_network_from_scratch as nnfs

iter = 1500
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])
nn = nnfs.NeuralNetwork(X, y)
Loss = np.zeros((iter))
for i in range(iter):
    nn.feedforward()
    nn.backprop()
    Loss[i] = abs(nn.output[0] - nn.y[0])

yp = np.round(nn.output, 4)
xyyp = np.concatenate((X, y, yp), axis=1)
columns = ('X1', 'X2', 'X3', 'Y', 'Yp')
xy = np.concatenate((X, y, yp), axis=1)
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 5.4))
tprw = ["#0000ff", "#0000ff", "#0000ff", "#009933"]
drw = ["#6699ff", "#6699ff", "#6699ff", "#66ff33", "#66ff33"]
lrw = ["#6699ff", "#6699ff", "#6699ff", "#99ff66", "#99ff66"]
colors = [lrw, drw, lrw, drw]

the_table = ax1.table(cellText=xy, cellColours=colors,
                      colLabels=columns, loc='center')
the_table.set_fontsize(16)
the_table.scale(1, 2)
ax2.plot(Loss)
ax2.set_title('Loss after '+str(iter)+' iterations')
plt.tight_layout()
plt.savefig(str(iter)+'iter.png')
