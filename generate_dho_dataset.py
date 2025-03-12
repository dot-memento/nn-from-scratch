import numpy as np
import scipy.integrate
import pandas as pd

# Define the DHO system
def dho_system(t, y, m, c, k):
    x, v = y  # y = [x, v]
    dxdt = v
    dvdt = - (c/m) * v - (k/m) * x
    return [dxdt, dvdt]

# Define parameter ranges
num_samples = 1000
m_values = np.random.uniform(0.5, 5.0, num_samples)
c_values = np.random.uniform(0.1, 3.0, num_samples)
k_values = np.random.uniform(0.5, 10.0, num_samples)
x0_values = np.random.uniform(-5.0, 5.0, num_samples)
v0_values = np.random.uniform(-2.0, 2.0, num_samples)

# Time grid
t_eval = np.linspace(0, 10, 100)  # 100 time steps

# Generate dataset
data = []
for i in range(num_samples):
    m, c, k, x0, v0 = m_values[i], c_values[i], k_values[i], x0_values[i], v0_values[i]
    sol = scipy.integrate.solve_ivp(dho_system, [0, 10], [x0, v0], args=(m, c, k), t_eval=t_eval)
    for j, t in enumerate(t_eval):
        data.append([t, m, c, k, x0, v0, sol.y[0, j], sol.y[1, j]])  # t, params, x, v

# Save dataset
df = pd.DataFrame(data)
df.to_csv("dho_dataset.csv", index=False, header=False)
