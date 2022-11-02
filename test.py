import numpy as np

a = np.array([[1,2,3], [2,3,4], [5,6,7]])
b = np.array([[3,2,1], [4,3,2], [7,6,5]])

print(a.T.dot(b).dot(a))
print(a.T.dot(b))