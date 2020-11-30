from flask import Flask, render_template,request
import requests
import pickle
from pricing import real_estate_prediction

app = Flask(__name__)

model = pickle.load(open('begaluru_prediction_pickle', 'rb'))

columns = real_estate_prediction.X.columns[4:]


@app.route('/', methods=['POST', 'GET'])
def results():
    if request.method == 'POST':
        sqft = request.form["Squareft"]
        bhk = request.form['uiBHK']
        bath = request.form['uiBathrooms']
        location = request.form['locations']

        bhk, bath, sqft = float(bhk), float(bath), int(sqft)

        result = real_estate_prediction.predict_price(location, sqft, bath, bhk)

        return render_template('results.html', result=result)
    else:
        return render_template('index.html', columns=columns)


if __name__ == "__main__":
    app.run()
