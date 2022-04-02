# Mouse Fatigue Estimation by GSR and EMG Values w/ TensorFlow
#
# Windows, Linux, or Ubuntu
#
# By Kutluhan Aktar
#
# Collate forearm muscle soreness data on the SD card, build and train a neural network model, and run the model directly on Wio Terminal.
# 
#
# For more information:
# https://www.theamplituhedron.com/projects/Mouse_Fatigue_Estimation_by_GSR_and_EMG_Values_w_TensorFlow/

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tflite_to_c_array import hex_to_c_array
from test_data import test_inputs, test_labels

# Create a class to build a neural network model after visualizing and scaling (normalizing) the soreness data (GSR and EMG) collected by Wio Terminal.
class Mouse_Fatigue:
    def __init__(self, csv_path):
        self.inputs = []
        self.labels = []
        self.model_name = "mouse_fatigue_level"
        self.scale_val = 1000
        # Read the collated soreness data set (GSR and EMG):
        self.df = pd.read_csv(csv_path)
    # Create graphics for each requested column.
    def graphics(self, column_1, column_2, x_label, y_label):
        # Show the requested data column from the data set:
        plt.style.use("dark_background")
        plt.gcf().canvas.set_window_title('Mouse Fatigue Estimation by GSR and EMG Values')
        plt.hist2d(self.df[column_1], self.df[column_2], cmap="coolwarm")
        plt.colorbar()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(x_label)
        plt.show()
    # Visualize data before creating and training the neural network model.
    def data_visualization(self):
        # Scrutinize data columns to build a model with appropriately formatted data:
        self.graphics('GSR', 'EMG', 'GSR', 'EMG')
    # Scale (normalize) data to define appropriately formatted inputs.
    def scale_data_and_define_inputs(self):
        self.df["scaled_GSR"] = self.df["GSR"] / self.scale_val
        self.df["scaled_EMG"] = self.df["EMG"] / self.scale_val
        # Create the inputs array by utilizing the scaled variables:
        for i in range(len(self.df)):
            self.inputs.append(np.array([self.df["scaled_GSR"][i], self.df["scaled_EMG"][i]]))
        self.inputs = np.asarray(self.inputs)
    # Assign labels for each input according to the predefined soreness classes for each data record.
    def define_and_assign_labels(self):
        self.labels = self.df["Soreness"]
    # Split inputs and labels into training and test sets.
    def split_data(self):
        # (training)
        self.train_inputs = self.inputs
        self.train_labels = self.labels
        # (test)
        self.test_inputs = test_inputs / self.scale_val
        self.test_labels = test_labels
    # Build and train an artificial neural network (ANN) model to make predictions on mouse fatigue levels (classes) based on GSR and EMG measurements.
    def build_and_train_model(self):
        # Build the neural network:
        self.model = keras.Sequential([
            keras.Input(shape=(2,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        # Compile:
        self.model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        # Train:
        self.model.fit(self.train_inputs, self.train_labels, epochs=150)
        # Test the model accuracy:
        print("\n\nModel Evaluation:")
        test_loss, test_acc = self.model.evaluate(self.test_inputs, self.test_labels) 
        print("Evaluated Accuracy: ", test_acc)
    # Save the model for further usage:
    def save_model(self):
        self.model.save("model/{}.h5".format(self.model_name))
    # Convert the TensorFlow Keras H5 model (.h5) to a TensorFlow Lite model (.tflite).
    def convert_TF_model(self, path):
        #model = tf.keras.models.load_model(path + ".h5")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        # Save the recently converted TensorFlow Lite model.
        with open(path + '.tflite', 'wb') as f:
            f.write(tflite_model)
        print("\r\nTensorFlow Keras H5 model converted to a TensorFlow Lite model!\r\n")
        # Convert the recently created TensorFlow Lite model to hex bytes (C array) to generate a .h file string.
        with open("model/{}.h".format(self.model_name), 'w') as file:
            file.write(hex_to_c_array(tflite_model, self.model_name))
        print("\r\nTensorFlow Lite model converted to a C header (.h) file!\r\n")
    # Run Artificial Neural Network (ANN):
    def Neural_Network(self, save):
        self.scale_data_and_define_inputs()
        self.define_and_assign_labels()
        self.split_data()
        self.build_and_train_model()
        if save:
            self.save_model()
            
# Define a new class object named 'mouse_fatigue_level':
mouse_fatigue_level = Mouse_Fatigue("data/mouse_fatigue_data_set.csv")

# Visualize data columns:
mouse_fatigue_level.data_visualization()

# Artificial Neural Network (ANN):        
mouse_fatigue_level.Neural_Network(True)

# Convert the TensorFlow Keras H5 model to a TensorFlow Lite model:
mouse_fatigue_level.convert_TF_model("model/{}".format(mouse_fatigue_level.model_name))
