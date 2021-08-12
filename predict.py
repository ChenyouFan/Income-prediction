import numpy as np

with open('./X_test') as f:
    next(f)
    x = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

w = np.load('./weights.npy')
b = np.load('./bias.npy')
Z = np.dot(x, w) + b
A = 1 / (1 + np.exp(-Z))
y = np.round(A).astype(np.int)

with open('./output_{}.csv'.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(y):
        f.write('{},{}\n'.format(i, int(label)))

# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open('./X_test') as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])
