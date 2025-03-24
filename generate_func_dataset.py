import numpy as np
import pandas as pd
import math


num_samples = 10000
x_values = np.random.uniform(-1.0, 1.0, num_samples)
y_values = np.random.uniform(-1.0, 1.0, num_samples)

data = []
for i in range(num_samples):
    x = 4*x_values[i]
    y = 4*y_values[i]
    distance = math.sqrt(x**2 + y**2)
    data.append([x, y, x*x+y*y-4*(math.cos(x*math.pi) + math.cos(y*math.pi))])

df = pd.DataFrame(data)
df.to_csv("func_dataset.csv", index=False, header=False)
