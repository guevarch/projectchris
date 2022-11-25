from flask import Flask, redirect, render_template, request
import numpy as np
import pandas as pd
from prophet import Prophet



app = Flask(__name__)

@app.route("/")
def index():
   
   return render_template("index.html")

@app.route("/home")
def home():
   
   return render_template("home.html")


@app.route('/run', methods=['GET', 'POST'])
def route():
    print('Inside route')
    # If a form is submitted
    if request.method == "POST":
        
        df = pd.read_csv("Resources/mldatapriceslog.csv")

        # instantiate the model and set parameters
        model = Prophet()

        # fit the model to historical data
        model.fit(df)

        # # Get values through input bars
        start = "2010-09-25"
        Date = request.form.get("Date")    

        insample = pd.DataFrame(pd.date_range(start, Date, periods=92))
        insample.columns = ['ds']

        # in-sample prediction
        prediction = model.predict(insample)

        # Get prediction
        prediction = prediction[prediction['ds'].dt.strftime('%Y-%m-%d') == Date]
        prediction = np.exp(prediction.yhat)
        prediction = prediction.values[0].round(2)
    else:
        prediction = ""
    

    return render_template("index.html", output = prediction, date=Date)

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000)