import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


def NN_Model():
    df = pd.read_pickle("../data/processed_data.pkl")
    categorical_columns = df.select_dtypes(include=['object', 'bool']).columns
    for i in categorical_columns:
        en = preprocessing.LabelEncoder()
        df[i] = en.fit_transform(df[i])
    y = df['Severity'].copy()
    X = df.drop('Severity', axis=1).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=100)

    inputs = tf.keras.Input(shape=(X.shape[1],))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    epochs = 100

    kerasMdl = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )

    loss = pd.DataFrame(kerasMdl.history)
    plt = pd.DataFrame({'accuracy': loss['accuracy'], 'validation_accuracy': loss['val_accuracy']})
    plt.plot()


if __name__ == "__main__":
    NN_Model()
