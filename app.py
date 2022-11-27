from flask import Flask, redirect, render_template, request
import numpy as np
import pandas as pd
from prophet import Prophet



app = Flask(__name__)

@app.route("/")
def view_home():
    return render_template("index.html", title="")

@app.route("/Metrics")
def view_first_page():
    return render_template("Metrics.html", title="Metrics")


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
        predictionprice = model.predict(insample)

        # Get prediction
        predictionprice = predictionprice[predictionprice['ds'].dt.strftime('%Y-%m-%d') == Date]
        predictionprice = np.exp(predictionprice.yhat)
        predictionprice = predictionprice.values[0].round(2)
        # predictionprice = ("On " +str(Date) + ", the forecasted price of bitcoin is $" + str(predictionprice))

        dfvalue = pd.read_csv("Resources/mlvaluedata.csv")

        # instantiate the model and set parameters
        model = Prophet()

        # fit the model to historical data
        model.fit(dfvalue)

        # in-sample prediction
        predictionvalue = model.predict(insample)

        # Get prediction
        predictionvalue = predictionvalue[predictionvalue['ds'].dt.strftime('%Y-%m-%d') == Date]
        predictionvalue = np.exp(predictionvalue.yhat)
        predictionvalue = predictionvalue.values[0].round(2)
        # predictionvalue = ("The forecasted value of bitcoin is $" + str(predictionvalue))
        predictionvalue
        

        dfwallet = pd.read_csv("Resources/mlwalletsdata.csv")

        # instantiate the model and set parameters
        model = Prophet()

        # fit the model to historical data
        model.fit(dfwallet)

        # in-sample prediction
        predictionwall = model.predict(insample)

        # Get prediction
        predictionwall = predictionwall[predictionwall['ds'].dt.strftime('%Y-%m-%d') == Date]
        predictionwall = (predictionwall.yhat)
        predictionwall = predictionwall.values[0].round(0)
        # predictionwall = ("The forecasted number of bitcoin wallets are " + str(predictionwall))
        predictionwall

        prediction = ("On " +str(Date) + ", the forecasted price of bitcoin is $" + str(predictionprice)) + (", the value of bitcoin is $" + str(predictionvalue) + (", and the number of bitcoin wallets are " + str(predictionwall))) 

    else:
        prediction = ""
    

    return render_template("index.html", output = prediction, date=Date)

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000)