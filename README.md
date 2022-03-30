# mouse-fatigue-detection

To detect fluctuating levels of muscle pain in different muscle groups, we will use electromyography EMG and galvanic skin response (GSR) measurements denoting EMG of muscle pain in the forearm. 

To interpret forearm muscle soreness levels simply using limited data without applying complex algorithms. 

We are managing to predict muscle pain levels in hopes of avoiding permanent mouse-fatigue injuries.

In retrospect, our project and research in coordination with Artificial Intelligence combined will present a neural network model with TensorFlow in Python. 

We will use the Wio terminal as it allows us to collect muscle soreness data: GSR and EMG measurements and run a trained neural network model to predict forearm muscle soreness levels. 

We then connect a GSR sensor and an EMG detector to the Wio Terminal to display the collected muscle soreness data on the WIO Terminal LCD screen.

Since Wio Terminal supports reading and writing information from/to files on an SD card, We stored the collected data in a CSV file on the SD card to create the data set. 

Although Wio Terminal provides Wi-Fi connectivity, We was able to save data packets via Wio Terminal without requiring any additional procedures.
 
After completing the data set, We built an artificial neural network model (ANN) with TensorFlow to predict forearm muscle soreness levels (classes) based on GSR and EMG measurements. 

As labels, We employed the empirically assigned muscle soreness classes for each data record while collecting data:

Relaxed

Tense

Exhausted

After training and testing the neural network model, We converted it from a TensorFlow Keras H5 model to a C array (.h file) to execute the model on Wio Terminal. 

Therefore, the device can detect precise forearm muscle soreness levels (classes) by running the model independently.

After uploading and running the code for collecting forearm muscle soreness data and saving information to a given CSV file on the SD card on Wio Terminal:

The device displays real-time GSR and EMG measurements on the TFT screen as line charts.
