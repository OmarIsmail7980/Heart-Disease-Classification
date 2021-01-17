from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("model.pkl", 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    chest_pain = request.form['chest_pain']
    resting_bp = request.form['resting_bp']
    cholestrol = request.form['cholestoral']
    high_sugar = request.form['high_sugar']
    ecg = request.form['ecg']
    max_rate = request.form['max_rate']
    exercise_angina = request.form['exercise_angina']
    st_depression = request.form['st_depression']
    slope = request.form['slope']
    thalium_scan = request.form['thalium_scan']
    
    arr= np.array([[age,sex, chest_pain, resting_bp, cholestrol, high_sugar,
                    ecg ,max_rate ,exercise_angina , st_depression ,slope ,
                    thalium_scan]])
    
    prediction = model.predict(arr)
    
    if(prediction ==1):
        return render_template('index.html', prediction_text = "You might have heart disease")
    else:
        return render_template('index.html', prediction_text = "You don't have heart disease")


if __name__ == "__main__":
    app.run(debug=True)
    
