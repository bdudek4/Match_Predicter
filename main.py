import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib  


with open ('results.txt') as file:
    matches_X = []
    matches_Y = []
    i = 0
    for line in file:
        matches_X.append([])
        matches_Y.append([])
        elements = line.split()
        for j, element in enumerate(elements):
            if j < 2:
                matches_X[i].append(element)
            elif j==2:
                matches_X[i].append(int(element))
            else:
                matches_Y[i].append(int(element))
        i += 1
'''
# Przykładowe dane
data = np.array([
    ['Team A', 'Team B', 2003],
    ['Team A', 'Team C', 2020],
    ['Team D', 'Team E', 2019],
    ['Team F', 'Team G', 2020],
    ['Team H', 'Team I', 2020],
    ['Team J', 'Team K', 2018]
])

# Wyniki meczów
y = np.array([
    [2, 1],
    [1, 1],
    [0, 0],
    [3, 2],
    [2, 2],
    [1, 0]
])
'''


data = np.array(matches_X)
y = np.array(matches_Y)



# Kodowanie gorącojedynkowe dla nazw drużyn
encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
encoded_teams = encoder.fit_transform(data[:, :2])  # Koduje tylko nazwy drużyn

# Zapisywanie kategorii encodera
joblib.dump(encoder, 'encoder.pkl')

# Dodawanie roku do zakodowanych danych drużyn
X = np.hstack((encoded_teams, data[:, 2].reshape(-1, 1)))

# Dzielenie danych na zestawy treningowe, testowe i walidacyjne
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Skalowanie cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Zapisywanie skalera
joblib.dump(scaler, 'scaler.pkl')

# Definicja metryki R^2
def r_squared(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


# Budowanie modelu
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='linear')  # 2 wyjścia dla wyników meczów
])

# Kompilacja modelu z dodatkową metryką R^2
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae', r_squared])

# Trenowanie modelu
history = model.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_val_scaled, y_val))

# Ocena modelu
test_loss, test_mae, test_r2 = model.evaluate(X_test_scaled, y_test)
print('Test MAE:', test_mae)
print('Test R^2:', test_r2)

# Zapisywanie modelu
model.save('model.h5')