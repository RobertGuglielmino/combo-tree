# # Example time series data
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import SimpleRNN, Dense

# # Let's assume this is your data, with 100 timesteps and 1 feature per step
# time_series_data = np.sin(np.linspace(0, 100, 100))  # Sine wave as toy data

# # Reshape data into (samples, timesteps, features)
# X = np.array([time_series_data[i:i+10] for i in range(len(time_series_data)-10)])  # Input sequences
# y = time_series_data[10:]  # Targets shifted by one time step
