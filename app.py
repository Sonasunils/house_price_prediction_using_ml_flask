from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model2.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    age = int(request.form["age"])
    location = request.form["location"]

    # Location encoding
    if location == "Urban":
        loc1, loc2 = 1, 0
    elif location == "Semi-Urban":
        loc1, loc2 = 0, 1
    else:
        loc1, loc2 = 0, 0

    features = np.array([[area, bedrooms, bathrooms, age, loc1, loc2]])
    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction_text=f"Estimated House Price: ₹ {prediction:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)

