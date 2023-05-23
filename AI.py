import os
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

location = os.path.abspath('data.h5')

if os.path.exists(location):
    model = load_model(location)
    print("Data loaded.")

    predict_value = float(input('Enter infrared sensor value...\n'))

    while predict_value:
        result = model.predict(np.array([[predict_value]]))[0][0]

        print(f"Predicted body temperatures are {result:.1f} degrees Celsius.\n")
        
        predict_value = float(input('Enter infrared sensor value...\n'))

else:
    print("No training data found.")
    data = pd.read_csv(os.path.abspath('data.csv')) 

    x = data.iloc[:, 1].values  # Infrared Sensor temperature
    y = data.iloc[:, 0].values  # Body temperature

    epochs = 100 # Frequency
    batch_size = 10 # Splitting
    validation_split = 0.2 # Data rate for verification

    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, model_checkpoint], validation_split=validation_split)
    print("Create training data.")

    model.save(location)
    print("Training data saved.")
