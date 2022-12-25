import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import SMOTE
import pickle as pkl


def finalModel():
    data = pd.read_pickle("../data/processed_data.pkl")
    data = data.replace([True, False], [1, 0])
    data.drop_duplicates(inplace=True)
    data.rename(columns={"Start_Weekday": "Day", "Start_Hour": "Hour", "Start_Year": "Year"}, inplace=True)
    feature_lst = ['Severity', 'Wind_Speed(mph)', 'Pressure(in)', 'Humidity(%)', 'Visibility(mi)', 'Temperature(F)',
                   'Wind_Chill(F)', 'Traffic_Signal', 'Crossing', 'Junction', 'Year', 'Hour', 'Day']
    df = data[feature_lst].copy()
    Years_list = [2016., 2017., 2021., 2020., 2019., 2018.]

    df_y1 = df[df['Year'] == 2020.0]
    df_y1.drop('Year', axis=1, inplace=True)

    target = 'Severity'
    y = df_y1[target]
    X = df_y1.drop(target, axis=1)
    sm = SMOTE(random_state=42)
    X_final, y_final = sm.fit_resample(X, y)
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_final)
    pkl.dump(scaler, open("../data/standardScalerModel.pkl", "wb"))
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=21)
    y_train = y_train - 1
    filepath = '../data/z_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')
    inputs = tf.keras.Input(shape=(X.shape[1],))
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    batch_size = 20
    epochs = 100

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint,
                   tf.keras.callbacks.ReduceLROnPlateau(),
                   tf.keras.callbacks.EarlyStopping(
                       monitor='val_loss',
                       patience=3,
                       restore_best_weights=True
                   )
                   ]
    )
    y_test = y_test - 1
    print("Test Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])


if __name__ == "__main__":
    finalModel()
