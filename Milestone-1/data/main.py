import numpy as np

def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Example: 5 data points
data = np.array([5, 10, 15, 20, 25])
result = moving_average(data, 3)

print("Data:", data)
print("Moving Average (3):", result)
