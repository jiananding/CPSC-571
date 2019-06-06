#reference:https://machinelearningmastery.com/make-sample-forecasts-arima-python/

#import packages
import pandas as pd
import numpy as np
from pyramid.arima import auto_arima

#to plot within notebook
import matplotlib.pyplot as plt


#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('DAUPSA1.csv')

#print the head
df.head()

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))

data = df.sort_index(ascending=True, axis=0)

train = data[:314]
valid = data[293:]

training = train['Production']
validation = valid['Production']

model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)

forecast = model.predict(n_periods=21)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])


plt.plot(training)
plt.plot(validation)
plt.plot(forecast['Prediction'])
plt.suptitle("United States' Production prediction by ARIMA")
plt.show()
