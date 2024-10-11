import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Load the data
matches = pd.read_csv("matches.csv", index_col=0)

# Data Preprocessing
matches["date"] = pd.to_datetime(matches["date"])
matches["h/a"] = matches["venue"].astype("category").cat.codes
matches["opp"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

# Train-test split
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
predictors = ["h/a", "opp", "hour", "day"]

# Model training
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])

# Prediction and evaluation
preds = rf.predict(test[predictors])
acc = accuracy_score(test["target"], preds)
precision = precision_score(test["target"], preds)

# Rolling averages function
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Applying rolling averages
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

# Making predictions with rolling averages
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)

# Merging with original data
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

# Class for mapping team names
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

# Mapping new team names
combined["new_team"] = combined["team"].map(mapping)

# Merging home and away predictions
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

print("Data Loaded Successfully")
print(matches.head())

print("Predictions Complete")
print(combined.head())

print("Merged Data Preview")
print(merged.head())
merged.to_csv("prediction_results.csv", index=False)
print("Results saved to prediction_results.csv")
