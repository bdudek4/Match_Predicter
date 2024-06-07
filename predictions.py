import tkinter as tk
import numpy as np
import joblib
import tensorflow as tf
from collections import defaultdict

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

def calculate_group_results(matches, predictions):
    results = defaultdict(lambda: {"points": 0, "goal_diff": 0, "goals_scored": 0})

    for match, pred in zip(matches, predictions):
        team1, team2, year = match
        goals_team1, goals_team2 = round(pred[0]), round(pred[1])

        if goals_team1 > goals_team2:
            results[team1]["points"] += 3
        elif goals_team2 > goals_team1:
            results[team2]["points"] += 3
        else:
            results[team1]["points"] += 1
            results[team2]["points"] += 1

        results[team1]["goal_diff"] += goals_team1 - goals_team2
        results[team2]["goal_diff"] += goals_team2 - goals_team1
        results[team1]["goals_scored"] += goals_team1
        results[team2]["goals_scored"] += goals_team2

    sorted_teams = sorted(results.items(), key=lambda item: (item[1]["points"], item[1]["goal_diff"], item[1]["goals_scored"]), reverse=True)
    return sorted_teams[:2], results

class SoccerPredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Soccer Match Predictions")
        self.geometry("1600x1000")
        self.model, self.scaler, self.encoder = load_model_and_scaler()

        # Data grouped by categories
        self.groups = {
            "Group A": [
                ['Katar', 'Ekwador', 2022],
                ['Senegal', 'Holandia', 2022],
                ['Katar', 'Senegal', 2022],
                ['Holandia', 'Ekwador', 2022],
                ['Katar', 'Holandia', 2022],
                ['Ekwador', 'Senegal', 2022]
            ],
            "Group B": [
                ['Anglia', 'Iran', 2022],
                ['USA', 'Walia', 2022],
                ['Walia', 'Iran', 2022],
                ['Anglia', 'USA', 2022],
                ['Walia', 'Anglia', 2022],
                ['Iran', 'USA', 2022]
            ],
            "Group C": [
                ['Argentyna', 'Arabia Saudyjska', 2022],
                ['Meksyk', 'Polska', 2022],
                ['Polska', 'Arabia Saudyjska', 2022],
                ['Argentyna', 'Meksyk', 2022],
                ['Polska', 'Argentyna', 2022],
                ['Arabia Saudyjska', 'Meksyk', 2022]
            ],
            "Group D": [
                ['Dania', 'Tunezja', 2022],
                ['Francja', 'Australia', 2022],
                ['Tunezja', 'Australia', 2022],
                ['Francja', 'Dania', 2022],
                ['Australia', 'Dania', 2022],
                ['Tunezja', 'Francja', 2022]
            ],
            "Group E": [
                ['Hiszpania', 'Kostaryka', 2022],
                ['Niemcy', 'Japonia', 2022],
                ['Japonia', 'Kostaryka', 2022],
                ['Hiszpania', 'Niemcy', 2022],
                ['Japonia', 'Hiszpania', 2022],
                ['Kostaryka', 'Niemcy', 2022]
            ],
            "Group F": [
                ['Belgia', 'Kanada', 2022],
                ['Maroko', 'Chorwacja', 2022],
                ['Belgia', 'Maroko', 2022],
                ['Chorwacja', 'Kanada', 2022],
                ['Chorwacja', 'Belgia', 2022],
                ['Kanada', 'Maroko', 2022]
            ],
            "Group G": [
                ['Brazylia', 'Serbia', 2022],
                ['Szwajcaria', 'Kamerun', 2022],
                ['Kamerun', 'Serbia', 2022],
                ['Brazylia', 'Szwajcaria', 2022],
                ['Kamerun', 'Brazylia', 2022],
                ['Serbia', 'Szwajcaria', 2022]
            ],
            "Group H": [
                ['Portugalia', 'Ghana', 2022],
                ['Urugwaj', 'Korea Południowa', 2022],
                ['Korea Południowa', 'Ghana', 2022],
                ['Portugalia', 'Urugwaj', 2022],
                ['Korea Południowa', 'Portugalia', 2022],
                ['Ghana', 'Urugwaj', 2022]
            ]
        }

        self.top_teams = []
        self.create_widgets()

    def create_widgets(self):
        self.group_frame = tk.Frame(self)
        self.group_frame.grid(column=0, row=0, padx=10, pady=10, sticky='nw')

        group_headers = list(self.groups.keys())
        split_index = len(group_headers) // 2
        left_groups = group_headers[:split_index]
        right_groups = group_headers[split_index:]

        self.result_labels = {}

        # Group frames
        for idx, group in enumerate(left_groups):
            header_label = tk.Label(self.group_frame, text=group, font=("Arial", 10, "bold"))
            header_label.grid(column=0, row=idx * 2, columnspan=2, sticky="w")

            result_text = tk.Text(self.group_frame, height=10, width=55)
            result_text.grid(column=0, row=(idx * 2) + 1, columnspan=2, padx=10, pady=5)
            self.result_labels[group] = result_text

        for idx, group in enumerate(right_groups):
            header_label = tk.Label(self.group_frame, text=group, font=("Arial", 10, "bold"))
            header_label.grid(column=2, row=idx * 2, columnspan=2, sticky="w")

            result_text = tk.Text(self.group_frame, height=10, width=55)
            result_text.grid(column=2, row=(idx * 2) + 1, columnspan=2, padx=10, pady=5)
            self.result_labels[group] = result_text

        # Bracket frame
        self.bracket_label = tk.Label(self, text="Knockout Bracket", font=("Arial", 14, "bold"))
        self.bracket_label.grid(column=1, row=0, pady=10, sticky='nw')

        self.bracket_text = tk.Text(self, height=30, width=90)
        self.bracket_text.grid(column=1, row=0, padx=10, pady=10)

        self.show_all_predictions()

    def show_all_predictions(self):
        self.top_teams = []
        for group, input_data in self.groups.items():
            predictions = predict_results(self.model, self.scaler, self.encoder, input_data)

            top_teams, all_results = calculate_group_results(input_data, predictions)
            self.top_teams.extend([team[0] for team in top_teams])

            result_text = self.result_labels[group]
            result_text.delete(1.0, tk.END)

            for match, pred in zip(input_data, predictions):
                team1, team2, year = match
                goals_team1, goals_team2 = max(0, round(pred[0])), max(0, round(pred[1]))
                result_text.insert(tk.END, f"{team1} vs {team2}: {goals_team1} - {goals_team2}\n")

            result_text.insert(tk.END, "\nTop Two Teams:\n")
            for team, stats in top_teams:
                result_text.insert(tk.END, f"{team}: {stats['points']} points, {stats['goal_diff']} GD\n")

        self.show_knockout_predictions()

    def show_knockout_predictions(self):
        knockout_rounds = {"Round of 16": [], "Quarterfinals": [], "Semifinals": [], "Final": []}

        def knockout_stage(matches):
            winners = []
            predictions = predict_results(self.model, self.scaler, self.encoder, matches)
            for match, pred in zip(matches, predictions):
                team1, team2, year = match
                goals_team1, goals_team2 = max(0, round(pred[0])), max(0, round(pred[1]))
                winner = team1 if goals_team1 > goals_team2 else team2
                winners.append((team1, team2, winner, goals_team1, goals_team2))
            return winners

        knockout_rounds["Round of 16"] = knockout_stage([
            (self.top_teams[0], self.top_teams[3], 2022),
            (self.top_teams[1], self.top_teams[2], 2022),
            (self.top_teams[4], self.top_teams[7], 2022),
            (self.top_teams[5], self.top_teams[6], 2022),
            (self.top_teams[8], self.top_teams[11], 2022),
            (self.top_teams[9], self.top_teams[10], 2022),
            (self.top_teams[12], self.top_teams[15], 2022),
            (self.top_teams[13], self.top_teams[14], 2022)
        ])

        quarterfinals_matches = [
            (knockout_rounds["Round of 16"][0][2], knockout_rounds["Round of 16"][1][2], 2022),
            (knockout_rounds["Round of 16"][2][2], knockout_rounds["Round of 16"][3][2], 2022),
            (knockout_rounds["Round of 16"][4][2], knockout_rounds["Round of 16"][5][2], 2022),
            (knockout_rounds["Round of 16"][6][2], knockout_rounds["Round of 16"][7][2], 2022)
        ]
        knockout_rounds["Quarterfinals"] = knockout_stage(quarterfinals_matches)

        semifinals_matches = [
            (knockout_rounds["Quarterfinals"][0][2], knockout_rounds["Quarterfinals"][1][2], 2022),
            (knockout_rounds["Quarterfinals"][2][2], knockout_rounds["Quarterfinals"][3][2], 2022)
        ]
        knockout_rounds["Semifinals"] = knockout_stage(semifinals_matches)

        final_match = [
            (knockout_rounds["Semifinals"][0][2], knockout_rounds["Semifinals"][1][2], 2022)
        ]
        knockout_rounds["Final"] = knockout_stage(final_match)

        self.bracket_text.delete(1.0, tk.END)

        def display_matches(stage, matches):
            self.bracket_text.insert(tk.END, f"\n{stage}:\n\n")
            for match in matches:
                team1, team2, winner, goals_team1, goals_team2 = match
                self.bracket_text.insert(tk.END, f"{team1} vs {team2}: {goals_team1} - {goals_team2} (Winner: {winner})\n")

        display_matches("Round of 16", knockout_rounds["Round of 16"])
        display_matches("Quarterfinals", knockout_rounds["Quarterfinals"])
        display_matches("Semifinals", knockout_rounds["Semifinals"])
        display_matches("Final", knockout_rounds["Final"])

if __name__ == "__main__":
    app = SoccerPredictionApp()
    app.mainloop()