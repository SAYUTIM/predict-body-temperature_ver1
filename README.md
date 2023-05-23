# predict-body-temperature_ver1
AI predicts underarm body temperature from infrared sensor body temperature

# Install
Before running the script, ensure that you have the following libraries installed:
- `numpy`
- `pandas`
- `keras`

You can install these libraries using pip:
```
pip install numpy pandas keras
```

# Usage
If the training data file (data.h5) exists, the script will load the pre-trained model and prompt you to enter an infrared sensor value.   Enter the value and press Enter to get the temperature prediction.   
Repeat this step as needed.  
  
If the training data file does not exist, the script will generate a new model by training it on the data in the data.csv file.  
The model will be saved as data.h5 for future use.

# Data.csv
The `data.csv` file contains the data used for training the temperature prediction model.  
This file is in CSV format and consists of two columns: infrared sensor values and corresponding body temperatures.
  
The data.csv file has the following format:

|   Body Temperature   | Infrared Sensor Value |
|----------------------|-----------------------|
| 36.7                 | 32.1                  |
| 36.1                 | 29.5                  |
| 35.7                 | 30.4                  |
| ...                  | ...                   |

Each row in the CSV file represents a data sample.  
The "Infrared Sensor Value" column contains the reading from the infrared sensor, and the "Body Temperature" column contains the corresponding body temperature measurement.
    
The more of this data, the more accurate it will be.  
Please prepare at least 100 pieces of data.
