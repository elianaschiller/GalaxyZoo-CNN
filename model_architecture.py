import tensorflow as tf
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers

df = pd.read_csv('https://astro.uchicago.edu/~andrey/classes/a211/data/training_solutions_rev1.csv')  # accessing dataframe with morphology probabilities

# function to add .jpg to the end of Galaxy IDs to access training samples
def jpg(file):
    return f"{file}.jpg"

df["GalaxyID"] = df["GalaxyID"].apply(jpg)

df["GalaxyID"] = df["GalaxyID"].astype(str)

# reducing dataframe to only Elliptical and Disk classes, adding index column
columns_mapper = {
    "GalaxyID": "GalaxyID",
    "Class1.1": "Elliptical",
    "Class1.2": "Disk",}

columns = list(columns_mapper.values())
gal_df = df.rename(columns=columns_mapper)[columns]
gal_df.set_index("GalaxyID", inplace=True)

gal_df = gal_df.reset_index()
gal_df.insert(0, 'Index', gal_df.index)

gal_df.head(10)  # prints out first 10 rows of df

# model construction
inputs = keras.Input(shape=(64, 64, 3))  

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
x = layers.BatchNormalization()(x) #added batch normalization
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(x)  
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=2)(x)  

x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)  
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=2)(x)  

x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=2)(x)  
x = layers.Dropout(0.4)(x) 

x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)  
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)

outputs = layers.Dense(2, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()  # prints summary of model 

# choosing optimizer and loss functions, and metric to evaluate results 
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])





