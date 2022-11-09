# Project
Topic: Bitcoin - How to determine if the network is over or undervalued based on metcalfe's law of network value and adoption.
Reason: Bitcoin is a tool for financial freedom, inclusion and soverignty?.
Data Souce: Blockchain.com bitcoin metrics such as supply, price, marketcap, volume, and open/close market prices.
Question I hope to answer - can we reasonably determine if the network is under/over valued based on metcafle's law and network adoption.
Terminology - active addresses, metcafe's law, value, network value, wallets, price, volume, open, close, bitcoin mining and supply. 

# Method

## Metcafe's Law Explanation, Collection, Loading, Cleaning.

### Intro to Metcafe's Law and Bitcoin
Graph supply chart for bitcoin, adoption in the form of active addresses, calculation for market cap, value and metcafe's law. Show formula to get network value, and log chart for prices. 

### Collection, Loading and Cleaning 
Collected market prices data such as open, close, adjusted close, volume and price from investing.com by simply downloading the csv from the website. I also needed to collect active wallet addresses and bitcoin supply through the years from blockchain.com. 
#### SQL
I created a database on pgadmin to create the market prices and active addresses data. I did an inner join using dates as my primary key for the market prices data and the active address data to form btcjoin.csv
#### Pandas
I cleaned the data by converting the dates to the correct format and the objects to floats. I then loaded the data to cleaning.ipnyb to merge the data set with the bitcoin supply. I updated the btcjoin.csv data that now contains market price data, active wallet addresses and supply.

## Preprocessing, Machine Learning and Analysis

### Preprocessing data and interpretation
Random Walk, Acf Auto Correlation, PAct partial autocorrelation, 1st and 2nd order differencing, p values, and adf values. 

### Prophet
Describe prophet method, results and graph for price predictions, network value and price predictions, and active addresses predictions.

### Arima? Still not sure, I will finalize with Asim on the 15 nov.
Describe prophet method, results and graph for price predictions, network value and price predictions, and active addresses predictions.

### Supervised Learning
Created a separate column for network value. I assigned a binary outcome of 1 for over valued, and 0 for under valued and named the column status. I made the target data as status, and everything else except the date as x non target. I split the data into train and test and conducted supervised learning models such as DecisionTreeClassifier, ExtraTreesClassifier, AdaBoostClassifier, and RandomForestClassifier to get the confusion matrix, accuracy, and sensitivity scores. 

## Website Components
Header - Bitcoin Project
Copy website format https://bitcoin.org/en/ and colurs
Contnet - introduction with video to bitcoin, metcafe's law, terminology etc
Content - all preprocessing images and explanations which includes random walk, acf, pacf, and autocorrelation plot
Content - interactive element of the prophet forecasts
Content -  prophet forecasts, pictures and images
Content - sueprvised learning confusion matrices and results

![image](https://user-images.githubusercontent.com/107594143/200726289-87a55eb0-3baa-4c47-9046-b330cabb97c3.png)
![image](https://user-images.githubusercontent.com/107594143/200726349-853f5079-90b5-4810-8b3b-cb0eaa25fd3d.png)
![image](https://user-images.githubusercontent.com/107594143/200726408-88600c60-bfcc-4ba8-9b77-c2e9e1bd7c07.png)
![image](https://user-images.githubusercontent.com/107594143/200726446-069b5366-18ad-4289-8dff-59597cc63d56.png)







