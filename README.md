# Introduction

"Bitcoin is a bank in cyberspace, run by incorruptible software, offering a global, affordable, simple, & secure savings account to billions of people that don’t have the option or desire to run their own hedge fund." - Michael Saylor

On 3 January 2009, the bitcoin network was created when Nakamoto mined the starting block of the chain, known as the genesis block. Embedded in the coinbase of this block was the text "The Times 03/Jan/2009 Chancellor on brink of second bailout for banks". Bitcoin is a decentralized digital currency that can be transferred on the peer-to-peer bitcoin network. Bitcoin transactions are verified by network nodes through cryptography and recorded in a public distributed ledger called a blockchain. The cryptocurrency was invented in 2008 by an unknown person or group of people using the name Satoshi Nakamoto. The currency began use in 2009,] when its implementation was released as open-source software.

I chose this topic because I believe bitcoin is a tool for economic sovereignty, individual property rights and financial inclusion. We can see evidence of this in the adoption rates in emerging markets or developing nations.

<p align="center">
  <img src="static\developingnationsadoption.webp" width="350" title="hover text">
</p>

The first objective is to use the Prophet time series model to forecast price, active addresses and value(according to Metcafe’s Law) by 2024. The second objective is to determine if the current price is overvalued or undervalued compared to Metcafe’s Law of Network Adoptions. The supervised learning models used to model the data are DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, LogisticRegression, and RandomForestClassifier.

# Method

The first step in the process is to find the data. The data sources included market data such as price, open, close, adjusted close, and date. Other data sources that were needed for the analysis were active addresses, coin supply, and wallets. The file types are comma separated values.

List Data Sources: 

- Circulating Bitcoin: https://www.blockchain.com/explorer/charts/total-bitcoins
- Wallets: https://www.blockchain.com/explorer/charts/my-wallet-n-users
- Bitcoin Market Data: https://www.investing.com/crypto/bitcoin
- Active Addresses: https://studio.glassnode.com/metrics


Next, is to create a database in PostgreSQL and create tables to house the csv data. A join function was used to combine two csv tables and psycopg2 was used to connect PostgreSQL to pandas dataframe. The new dataframe was further cleaned, cured and prepared for analysis and Prophet time series and supervised machine learning. 



# Results

## Formulas and Terminology

<pre><code>
df['Metcafe']=df['address']**2
df['value'] = df['Metcafe']/df['mined']
df["value"] = df["value"].map("{:.2f}".format)
df['value']=df['value'].astype("float")
df['networkvalue'] = df["price"] - df["value"]
</code></pre>

### Value = Metcafe's law = (Active Addressess)^2

The value of a network is famously accredited to Bob Metcalfe, the inventor of Ethernet and founder of the computer networking company 3Com. Metcalfe’s Law states that a network’s value is proportional to the square of the number of its users. It also reveals when Bitcoin has been overvalued.  Wheatley and co point to four occasions when Bitcoin has become overvalued and then crashed; in other words, when the bubble has burst.

### Price = Market Cap/Current Supply

For a cryptocurrency like Bitcoin, market capitalization (or market cap) is the total value of all the coins that have been mined. It’s calculated by multiplying the number of coins in circulation by the current market price of a single coin. This is very similar to stock valuations, shares x stock price = market cap.

Moreover, the circulating supply of a cryptocurrency can be used for calculating its market capitalization, which is generated by multiplying the current market price with the number of coins in circulation. So if a certain cryptocurrency has a circulating supply of 1,000,000 coins, which are being traded at $5.00 each, the market cap would be equal to $5,000,000.

### Network Value = Value - Price

Network Value is simply subtracting the current value or Metcalfe's law by the current price.

## Preprocessing

### Determining Stationary

Stationary Time Series - The observations in a stationary time series are not dependent on time. Time series are stationary if they do not have trend or seasonal effects. Summary statistics calculated on the time series are consistent over time, like the mean or the variance of the observations. When a time series is stationary, it can be easier to model. Statistical modeling methods assume or require the time series to be stationary to be effective.

Observations from a non-stationary time series show seasonal effects, trends, and other structures that depend on the time index. Summary statistics like the mean and variance do change over time, providing a drift in the concepts a model may try to capture. Classical time series analysis and forecasting methods are concerned with making non-stationary time series data stationary by identifying and removing trends and removing seasonal effects.

Mean and Variance Test = non stationary, large differences in mean and variances.
Data was split into two and ran mean and var tests.

<pre><code>
Mean and Var Test Linear
mean1 = 230.57, mean2 = 16936.32
variance1=61688.13, variance2=288670101.60
</code></pre>

Histogram Linear

<p align="center">
  <img src="static\histogramlinear.png" width="350" title="hover text">
</p>

Non Gaussian curve indicates that this squashed distribution of the observations may be another indicator of a non-stationary time series.
Reviewing the plot of the time series again, we can see that there is an obvious seasonality component, and it looks like the seasonality component is growing.
This may suggest an exponential growth from season to season. A log transform can be used to flatten out exponential change back to a linear relationship.

Histogram Log

<p align="center">
  <img src="static\histogramlog.png" width="350" title="hover text">
  <img src="static\logprices.png" width="350" title="hover text">
</p>

We also create a line plot of the log transformed data and can see the exponential growth seems diminished, but we still have a trend and seasonal elements.

<pre><code>
Mean and Var Test Log
mean1 = 3.963961, mean2=9.174585
variance1=5.856168, variance2=1.358006
</code></pre>

Augmented Dickey-Fuller Test

The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.

Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.

Linear ADF - Running the example prints the test statistic value of -1.769203. The more negative this statistic, the more likely we are to reject the null hypothesis (we have a stationary dataset). As part of the output, we get a look-up table to help determine the ADF statistic. We can see that our statistic value of  -1.769203 is greater than the value of -3.432 at 1%. This suggests that we cannot reject the null hypothesis with a significance level of less than 1%. Not rejecting the null hypothesis means that the process has unit root, and in turn that the time series is non stationary or does have time-dependent structure

<pre><code>
ADF Statistic: -1.769203
p-value: 0.395855
Critical Values:
	1%: -3.432
	5%: -2.862
	10%: -2.567_test_split(X,
   y, random_state=1, stratify=y)
</code></pre>

Log ADF - We can see that the value is larger than the critical values, again, meaning that we can fail to reject the null hypothesis and in turn that the time series is non-stationary.


<pre><code>
ADF Statistic: -3.182831
p-value: 0.021000
	1%: -3.432
	5%: -2.862
	10%: -2.567
</code></pre>


Preprocessing Conclusion: DATA IS NON STATIONARY - IT is time dependant

A stationary time series is one whose statistical properties do not depend on the time at which the series is observed.15 Thus, time series with trends, or with seasonality, are not stationary — the trend and seasonality will affect the value of the time series at different times. On the other hand, a white noise series is stationary — it does not matter when you observe it, it should look much the same at any point in time.

Prophet does not perform well on non-stationary data because it is difficult to find the actual seasonality and trend of the data if the patterns are inconsistent.

Price prediction is very difficult to begin with. If we had working prediction models, then we would all be amazing stock pickers and billionaires. Nonetheless, the model does incredibly poor in predicting prices. The forecasts upper/lower bound are very off from the actual price.

## Prophet

The process for prophet is to create a df_train, fitting it into a prophet model, and m.predict forecast. The forecast function splits the y value into yhat, yhat_lower and yhat_upper. This creates upper, lower and middle projections. By using m.plot(forecast), the df_train and forecast values are plotted. However, there is another method called insample wherein the analyst can set the pd.date_range of the prediction.




### Prices, Wallets, Active Addresses and Value

<p align="center">
  <img src="static\prices.png" width="400" title="hover text">
  <img src="static\active_addressess.png" width="400" title="hover text">
  <img src="static\value.png" width="400" title="hover text"> 
</p>


## Machine Learning Models 

### Binary Outcome using Logistic Regression on Price. Used Oversampling, undersampling and smoteenn method. 

#### LogisticRegression
<pre><code>
LOGISTIC REGRESSION

            Confusion Matrix
                        Predicted 0	Predicted 1
            Actual 0	    1242	0
            Actual 1	    74	        0
            Training Score: 0.9368
            Testing Score: 0.9437
                          precision    recall  f1-score   support
            
                       0       0.94      1.00      0.97      1242
                       1       0.00      0.00      0.00        74
            
                accuracy                           0.94      1316
               macro avg       0.47      0.50      0.49      1316
            weighted avg       0.89      0.94      0.92      1316
            
            Balanced Accuracy Score = 0.5
            Accuracy Score = 0.94376
        
     
        LOGISTIC REGRESSION - Oversampling Using RandomOverSampler
    
            Confusion Matrix
                        Predicted 0	Predicted 1
            Actual 0	    0	            1242
            Actual 1	    0	            74
            
            Training Score: 0.5
            Testing Score: 0.05623

            Imbalanced Classification Report
                               pre       rec       spe        f1       geo       iba       sup
            
                      0       0.00      0.00      1.00      0.00      0.00      0.00      1242
                      1       0.06      1.00      0.00      0.11      0.00      0.00        74
            
            avg / total       0.00      0.06      0.94      0.01      0.00      0.00      1316
            
            Balanced Accuracy Score = 0.5
            Accuracy Score = 0.056

        LOGISTIC REGRESSION - Undersampling Using ClusterCentroids
    
            Confusion Matrix
   
                        Predicted 0	Predicted 1
            Actual 0	    1046	    196
            Actual 1	    34	            40
            
            Training Score: 0.770
            Testing Score: 0.825

            Imbalanced Classification Report
            pre       rec       spe        f1       geo       iba       sup

            0       0.97      0.84      0.54      0.90      0.67      0.47      1242
            1       0.17      0.54      0.84      0.26      0.67      0.44        74

            avg / total       0.92      0.83      0.56      0.86      0.67      0.47      1316
            
            Balanced Accuracy Score = 0.6913
            Accuracy Score = 0.8252



        LOGISTIC REGRESSION - Over and Under sampling using SMOTEENN
    
            Training Score: 0.6774
            Testing Score: 0.8199
            
            Confusion Matrix
                        Predicted 0	Predicted 1
            Actual 0	    1037	    205
            Actual 1	    32	            42
            Balanced Accuracy Score = 0.7012
            Accuracy Score = 0.8199
            
            Imbalanced Classification Report
                            pre       rec       spe        f1       geo       iba       sup
            
                    0       0.97      0.83      0.57      0.90      0.69      0.49      1242
                    1       0.17      0.57      0.83      0.26      0.69      0.46        74
            
            avg / total       0.93      0.82      0.58      0.86      0.69      0.49      1316
        
</code></pre>


# Conclusion

# References

References
1. 9.1 Stationarity and differencing | Forecasting: Principles and Practice (3rd ed). (n.d.). https://otexts.com/fpp3/stationarity.html
2. Author At, |. (n.d.). Top Cryptocurrency Countries by Adoption (2022 Data). Bankless Times. https://www.banklesstimes.com/cryptocurrency/top-countries-leading-in-cryptocurrency-adoption/
3. Bitcoin. (n.d.). MicroStrategy. https://www.michael.com/en/bitcoin
4. Bitcoin Inflation : Woobull Charts. (n.d.). https://charts.woobull.com/bitcoin-inflation/
5. Brownlee, J. (2016, December 30). How to Check if Time Series Data is Stationary with Python. Machine Learning Mastery. https://machinelearningmastery.com/time-series-data-stationary-python/
6. keziesuemo. (2021, October 15). Analysis Shows that about 85% of Circulating Bitcoin Has Not Been Sold in over Three Months. Remitano. https://remitano.com/news/dk/post/13973-analysis-shows-that-about-85-percent-of-circulating-bitcoin-has-not-been-sold-in-over-three-months
7. Metcalfe’s Law - calculator. (n.d.). fxSolver. https://www.fxsolver.com/browse/formulas/Metcalfe%E2%80%99s+Law
8. Wikipedia contributors. (2022, November 7). Bitcoin. Wikipedia. https://en.wikipedia.org/wiki/Bitcoin