import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib


with open('results2.txt', encoding='utf-8-sig') as file:
    matches_X = []
    matches_Y = []
    i = 0
    for line in file:
        matches_X.append([])
        matches_Y.append([])
        elements = [element.strip() for element in line.split(',')]
        for j, element in enumerate(elements):
            if j < 3:
                if j == 2:
                    matches_X[i].append(int(element))
                else:
                    matches_X[i].append(element)
            else:
                matches_Y[i].append(int(element))
        i += 1

data = np.array(matches_X)
print(matches_Y)
y = np.array(matches_Y)

data_flipped = np.hstack((data[:, 1].reshape(-1, 1), data[:, 0].reshape(-1, 1), data[:, 2].reshape(-1, 1)))
data_combined = np.vstack((data, data_flipped))

y_combined = np.vstack((y, y))


encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
encoded_teams = encoder.fit_transform(data[:, :2])  # Koduje tylko nazwy drużyn

joblib.dump(encoder, 'encoder.pkl')

X = np.hstack((encoded_teams, data[:, 2].reshape(-1, 1)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')


def r_squared(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='linear')  # 2 wyjścia dla wyników meczów
])


model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae', r_squared])

history = model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_val_scaled, y_val))

test_loss, test_mae, test_r2 = model.evaluate(X_test_scaled, y_test)
print('Test MAE:', test_mae)


model.save('model.h5')