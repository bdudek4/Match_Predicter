import numpy as np
import joblib
import tensorflow as tf

def load_model_and_scaler():
    model = tf.keras.models.load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')  


    return model, scaler, encoder


def encode_input(team1, team2, year, encoder):

    teams = np.array([[team1, team2]])
    encoded_teams = encoder.transform(teams)
    year_array = np.array([[year]])
    features = np.hstack((encoded_teams, year_array))
    return features


def predict_results(model, scaler, encoder, input_data):
    try:
        encoded_data = np.vstack([encode_input(row[0], row[1], row[2], encoder) for row in input_data])
        scaled_data = scaler.transform(encoded_data)
        predictions = model.predict(scaled_data)
        return predictions
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    # Wczytywanie modelu, skalera i encodera
    model, scaler, encoder = load_model_and_scaler()
    '''
    # Przykładowe dane wejściowe
    input_data = np.array([
        ['Team A', 'Team F', 2023]

    ])

    '''

    input_data = np.array([
    ['Katar', 'Ekwador', 2022],
    ['Senegal', 'Holandia', 2022],
    ['Katar', 'Senegal', 2022],
    ['Holandia', 'Ekwador', 2022],
    ['Katar', 'Holandia', 2022],
    ['Ekwador', 'Senegal', 2022],
    ['Anglia', 'Iran', 2022],
    ['USA', 'Walia', 2022],
    ['Walia', 'Iran', 2022],
    ['Anglia', 'USA', 2022],
    ['Walia', 'Anglia', 2022],
    ['Iran', 'USA', 2022],
    ['Argentyna', 'Arabia Saudyjska', 2022],
    ['Meksyk', 'Polska', 2022],
    ['Polska', 'Arabia Saudyjska', 2022],
    ['Argentyna', 'Meksyk', 2022],
    ['Polska', 'Argentyna', 2022],
    ['Arabia Saudyjska', 'Meksyk', 2022],
    ['Dania', 'Tunezja', 2022],
    ['Francja', 'Australia', 2022],
    ['Tunezja', 'Australia', 2022],
    ['Francja', 'Dania', 2022],
    ['Australia', 'Dania', 2022],
    ['Tunezja', 'Francja', 2022],
    ['Hiszpania', 'Kostaryka', 2022],
    ['Niemcy', 'Japonia', 2022],
    ['Japonia', 'Kostaryka', 2022],
    ['Hiszpania', 'Niemcy', 2022],
    ['Japonia', 'Hiszpania', 2022],
    ['Kostaryka', 'Niemcy', 2022],
    ['Belgia', 'Kanada', 2022],
    ['Maroko', 'Chorwacja', 2022],
    ['Belgia', 'Maroko', 2022],
    ['Chorwacja', 'Kanada', 2022],
    ['Chorwacja', 'Belgia', 2022],
    ['Kanada', 'Maroko', 2022],
    ['Brazylia', 'Serbia', 2022],
    ['Szwajcaria', 'Kamerun', 2022],
    ['Kamerun', 'Serbia', 2022],
    ['Brazylia', 'Szwajcaria', 2022],
    ['Kamerun', 'Brazylia', 2022],
    ['Serbia', 'Szwajcaria', 2022],
    ['Portugalia', 'Ghana', 2022],
    ['Urugwaj', 'Korea Południowa', 2022],
    ['Korea Południowa', 'Ghana', 2022],
    ['Portugalia', 'Urugwaj', 2022],
    ['Korea Południowa', 'Portugalia', 2022],
    ['Ghana', 'Urugwaj', 2022]
]
)

    # Przewidywanie wyników
    predictions = predict_results(model, scaler, encoder, input_data)
    print("Predicted Results:")
    for i, pred in enumerate(predictions):
        if i==0:print("Grupa A")
        if i==6:print("Grupa B")
        if i==12:print("Grupa C")
        if i ==18:print("Grupa D")
        if i ==24:print("Grupa E")
        if i ==30:print("Grupa F")
        if i ==36:print("Grupa G")
        if i ==42:print("Grupa H")
        print(f"Match {input_data[i][0]} vs {input_data[i][1]} ({input_data[i][2]}): {max(0,round(pred[0]))} - {max(0,round(pred[1]))}")