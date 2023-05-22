import os
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

location = os.path.abspath('data.h5')

if os.path.exists(location):
    model = load_model(location)
    print("学習データを読み込みました！")

    predict_value = float(input('赤外線センサーの値を入力してください\n'))

    while predict_value:
        result = model.predict(np.array([[predict_value]]))[0][0]

        print(f"予測される体温は{result:.1f}度です\n")
        
        predict_value = float(input('赤外線センサーの値を入力してください\n'))

else:
    print("学習データが見つかりません！")
    data = pd.read_csv(os.path.abspath('data.csv')) 

    x = data.iloc[:, 1].values  # 赤外線センサー
    y = data.iloc[:, 0].values  # 脇下体温

    epochs = 100 # 回数
    batch_size = 10 # 分割
    validation_split = 0.2 # 検証用データ率

    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, model_checkpoint], validation_split=validation_split)
    print("学習データを作成！")

    model.save(location)
    print("学習データを保存しました！")