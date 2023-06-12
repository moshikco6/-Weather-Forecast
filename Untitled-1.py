# %%
import pandas as pd
import json 
import requests 
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# %%

df_weather=pd.read_csv('Merge weather.csv')



# %%
df_weather


# %%
df_weather.index=pd.to_datetime(df_weather.index)


# %%

df_weather[['Temp_Ave']].plot()

# %%

df_weather[['Wind_Speed']].plot()

# %%
df_weather[['Rain']].plot()


# %%
df_weather['predict']=df_weather.shift(-1)['Temp_Ave']


# %%
df_weather=df_weather.ffill()
x=Ridge(alpha=.1)


# %%
predict=df_weather.columns[~df_weather.columns.isin(['location','predict','Unnamed: 0','Sunrise','Sunset','Description','Cloud_Conver'])]
predict



# %%
def backtest(df_weather,model,predict,start=1825,step=90):
    all_predictions=[]
    for i in range(start,df_weather.shape[0],step):
        train=df_weather.iloc[:i,:]
        test=df_weather.iloc[i:(i+step),:]
        model.fit(train[predict],train["predict"])
        preds=model.predict(test[predict])
        preds=pd.Series(preds,index=test.index)
        combined=pd.concat([test["predict"],preds],axis=1)
        combined.columns=["actual","prediction"]
        combined["diff"]=(combined["prediction"]-combined["actual"]).abs()
        all_predictions.append(combined)
    return pd.concat(all_predictions)



# %%

predictions=backtest(df_weather,x,predict)


# %%

predictions

# %%


predictions["diff"].mean()


# %%
import numpy as np
import tensorflow as tf

df_weather['date'] = pd.to_datetime(df_weather['date'])
filtered_df = df_weather[(df_weather['date'].dt.month == 8) & (df_weather['date'].dt.day == 1)]


filtered_df

# %%
dates = np.array(filtered_df["date"])
temperatures = np.array(filtered_df["Temp_Ave"])


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


model.compile(loss='mean_squared_error', optimizer='adam')


x = np.array([i+2013 for i in range(len(x))])
y = temperatures


model.fit(x, y, epochs=1000, verbose=0)


prediction = model.predict([2023])

print(f"The {prediction} temperature for August 1st, 2023")

# %%


# %%


