import numpy as np

for i in range(10):
    for j in range(20):
        for k in range(30):
            print(i, j, k)
            if i == 1 and j == 2 and k == 3:
                break

print("Broke")