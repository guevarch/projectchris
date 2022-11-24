from flask import Flask, redirect, render_template
import numpy as np
import requests
from prophet import Prophet

app = Flask(__name__)

@app.route("/index")
def index():
   
   return render_template("index.html")

@app.route('/run', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if requests.method == "POST":
        
        df = pd.read_csv("Resources/mldata.csv")

        # instantiate the model and set parameters
        model = Prophet()

        # fit the model to historical data
        model.fit(df)

        # # Get values through input bars
        start = requests.form.get("start")
        end = requests.form.get("end")    

        insample = pd.DataFrame(pd.date_range(start, end, periods=92))
        insample.columns = ['ds']

        # in-sample prediction
        prediction = model.predict(insample)

        # Get prediction
        prediction = prediction[prediction['ds'].dt.strftime('%Y-%m-%d') == end]
        prediction = np.exp(prediction.yhat)
    else:
        prediction = ""
        
    return redirect("/index", output = prediction)

if __name__ == '__main__':
   app.run()