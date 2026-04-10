import pickle
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('home.html')

@app.route('/predict_datapoint', methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            region = float (request.form.get('region'))

            final_input = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes,region]])

            scaled_input = standard_scaler.transform(final_input)

            prediction = ridge_model.predict(scaled_input)[0]

            return render_template('home.html', results=prediction)

        except:
            return render_template('home.html', result="Invalid Input ")
    
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)