from flask import Flask, render_template
import numpy as np
import pandas as pd
import joblib
import request
from prophet import Prophet

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        df = pd.read_csv("Resources/mldata.csv")

        # instantiate the model and set parameters
        model = Prophet()

        # fit the model to historical data
        model.fit(df)

        # # Get values through input bars
        start = request.form.get("start")
        end = request.form.get("end")    

        insample = pd.DataFrame(pd.date_range(start, end, periods=92))
        insample.columns = ['ds']

        # in-sample prediction
        prediction = model.predict(insample)

        # Get prediction
        prediction = prediction[prediction['ds'].dt.strftime('%Y-%m-%d') == end]
        prediction = np.exp(prediction.yhat)
    else:
        prediction = ""
        
    return render_template("index.html", output = prediction)

if __name__ == '__main__':
   app.run()