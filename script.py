# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Read the dataset
dataset = pd.read_csv('admissions_data.csv')

# Split features and labels
labels = dataset.iloc[:, -1]
features = dataset.iloc[:, 1:-1]

# Split the dataset into training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.22, random_state=42)

# Apply column transformation to scale numeric features
ct = ColumnTransformer([("only numeric", StandardScaler(), features.columns)], remainder="passthrough")
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

# Define the model architecture
def my_model(input_shape):
  model = Sequential(name="deep_learning_regression")
  model.add(InputLayer(input_shape=input_shape))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(1))
  return model

# Create the model
model = my_model(features_train_scaled.shape[1])

# Compile the model
opt = Adam(learning_rate=0.01)
model.compile(loss='mse', metrics=['mae'], optimizer=opt)

# Fit the model to the training data
history = model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)

# Evaluate the model on the training data
res_mse, res_mae = model.evaluate(features_train_scaled, labels_train, verbose=1)

# Predict chance of admission for each line in the training set
admission_predictions = model.predict(features_train_scaled)
for i, prediction in enumerate(admission_predictions):
    print(f"Row In Dataset {i+1}: Chance of Admit: {prediction[0]}")

# Create plots to visualize model metrics
"""
fig = plt.figure()

# Plot MAE over each epoch
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('Model MAE')
ax1.set_ylabel('MAE')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')

# Plot loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper left')

# Save the plots as an image
fig.tight_layout()
fig.savefig('static/images/my_plots.png')
"""
