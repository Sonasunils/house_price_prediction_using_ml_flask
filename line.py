import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    "area": [800, 1000, 1200, 1500, 1800],
    "bedrooms": [1, 2, 2, 3, 3],
    "bathrooms": [1, 1, 2, 2, 3],
    "age": [15, 10, 8, 5, 2],
    "loc_urban": [1, 1, 0, 1, 0],
    "loc_semi": [0, 0, 1, 0, 1],
    "price": [3000000, 4000000, 4500000, 6000000, 6500000]
}

df = pd.DataFrame(data)

X = df.drop("price", axis=1)
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# SAVE MODEL (THIS CREATES model2.pkl)
pickle.dump(model, open("model2.pkl", "wb"))

print("model2.pkl created successfully")
