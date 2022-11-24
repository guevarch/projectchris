from flask import Flask, render_template
import numpy as np
import pandas as pd
from datetime import date
from prophet import Prophet

df = pd.read_csv("Resources/mldata.csv")

# instantiate the model and set parameters
model = Prophet()

# fit the model to historical data
model.fit(df)


import joblib

joblib.dump(model, "model.pkl")