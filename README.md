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

<pre><code>CREATE TABLE IF NOT EXISTS public.activeadd
(
    date character varying(40) COLLATE pg_catalog."default" NOT NULL,
    address character varying(40) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT activeadd_pkey PRIMARY KEY (date)
);

CREATE TABLE IF NOT EXISTS public.btc
(
    date date NOT NULL,
    price character varying(40) COLLATE pg_catalog."default" NOT NULL,
    open character varying(40) COLLATE pg_catalog."default" NOT NULL,
    high character varying(40) COLLATE pg_catalog."default" NOT NULL,
    low character varying(40) COLLATE pg_catalog."default" NOT NULL,
    volume character varying(40) COLLATE pg_catalog."default" NOT NULL,
    change character varying(40) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT btc_pkey PRIMARY KEY (date)
);

CREATE TABLE IF NOT EXISTS public.btcjoin
(
    date date,
    price character varying(40) COLLATE pg_catalog."default",
    open character varying(40) COLLATE pg_catalog."default",
    high character varying(40) COLLATE pg_catalog."default",
    low character varying(40) COLLATE pg_catalog."default",
    volume character varying(40) COLLATE pg_catalog."default",
    change character varying(40) COLLATE pg_catalog."default",
    value character varying(40) COLLATE pg_catalog."default"
);

CREATE TABLE IF NOT EXISTS public.wall
(
    date date NOT NULL,
    value character varying(40) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT wall_pkey PRIMARY KEY (date)
);
END;</code></pre>


# Results

## Formulas and Terminology

<pre><code>
df['Metcafe']=df['address']**2
df['value'] = df['Metcafe']/df['mined']
df["value"] = df["value"].map("{:.2f}".format)
df['value']=df['value'].astype("float")
df['networkvalue'] = df["price"] - df["value"]
</code></pre>

### Value = Metcalfe's law = (Active Addressess)^2

The value of a network is famously accredited to Bob Metcalfe, the inventor of Ethernet and founder of the computer networking company 3Com. Metcalfe’s Law states that a network’s value is proportional to the square of the number of its users. It also reveals when Bitcoin has been overvalued.  Wheatley and co point to four occasions when Bitcoin has become overvalued and then crashed; in other words, when the bubble has burst.

### Price = Market Cap/Current Supply

For a cryptocurrency like Bitcoin, market capitalization (or market cap) is the total value of all the coins that have been mined. It’s calculated by multiplying the number of coins in circulation by the current market price of a single coin.

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

Add two graphs showing bad predictions

*** Paste two different timelines with different predictions**

## Prophet

The process for prophet is to create a df_train, fitting it into a prophet model, and m.predict forecast. The forecast function splits the y value into yhat, yhat_lower and yhat_upper. This creates upper, lower and middle projections. By using m.plot(forecast), the df_train and forecast values are plotted. However, there is another method called insample wherein the analyst can set the pd.date_range of the prediction.

<pre><code>
df_train = df[['date', 'price']]
df_train = df_train.rename(columns = {"date":"ds", "price":"y"})

m = Prophet()
m.fit(df_train)

n_years =2
period = n_years * 365
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower','yhat_upper']].tail()

>fig1 = m.plot(forecast)

# Create a data frame that lists dates from Oct - Dec 2017
insample = pd.DataFrame(pd.date_range("2010-09-25", "2024-01-01", periods=92))

# Change the column name
insample.columns = ['ds']

# in-sample prediction
prediction = model.predict(insample)

# Plot
fig = model.plot(prediction, figsize=(10,5))
ax = fig.gca()
ax.set_title("BTC Prediction", size=20)
ax.set_xlabel("Date", size=18)
ax.set_ylabel("value", size=18)
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', rotation=45, labelsize=15)
ax.set_xlim(pd.to_datetime(['2010-09-25', '2024-12-31'])) 
plt.show();
</code></pre>
	


### Prices

<p align="center">
  <img src="static\BTC_price_prediction2024.png" width="650" title="hover text">
</p>

### Wallets

<p align="center">
  <img src="static\wallets.png" width="650" title="hover text">
</p>

### Active Addresses

<p align="center">
  <img src="static\active_addressess.png" width="650" title="hover text">
</p>

## Machine Learning Models 

### DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, LogisticRegression, and RandomForestClassifier.

#### Preprocessing Data

Creat New Column that indicates binary outcome - over/under valuation. Create targets - X all columns except date and status, and use get.dummies to convert categorical data into dummy or indicator variables. Then use train_test_split x,y data. Next is to classify into each model. Then use standardscaler to fit the x_train and x_test data 

<pre><code>

df['status'] = df['networkvalue'].apply(lambda x: '1' if x > 0 else '0')
# Create our features
X = df.drop(columns="status")
X = pd.get_dummies(X)

# Create our target
X = df.drop(columns="date")
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X,
   y, random_state=1, stratify=y)

</code></pre>

### classification_report for each and explanations with images

## Additional Thougth Experiment

Bitcoin’s supply is capped at 21 million and currently it is at 19,201,975 coins right now and it is estimated that 4-6 million coins are permanently lost. According to analysis, 85.14% of existing bitcoins have not been transferred or sold for more than three months. Mathematically, if the denominator is fixed, and the numerator is the only variable that can substantially change - ultimately value of the network can only increase. At 800k active addresses and 84M wallets, there is a lot more room to grow.

A quick thought experiment: lets assume a consistent halving supply issuance, ratio of 82 wallets/active addresses, and assume 30% increase in adoption per year.

*** project bitcoin projection table 2032 *** 
*** show woobull bitcoin inflation rate *** 

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
