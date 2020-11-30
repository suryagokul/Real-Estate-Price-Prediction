import pandas as pd
import pickle
import json
import warnings
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
df = pd.read_csv('Bengaluru_House_Data.csv')

df[:5]

df.shape

df.info()

df['area_type'].value_counts()

df.columns

"""# Data Cleaning"""

df.drop(['area_type','availability','society','balcony'],axis=1,inplace=True)

df[:4]

df.isnull().sum()

df1 = df.dropna()

df1.isnull().sum()

df1['size'].unique()

df1['BHK'] = df1['size'].apply(lambda x:x.split(' ')[0])

df1[:4]

df1.drop('size',axis=1,inplace=True)

df2 = df1.copy()

df2[:4]

df2.total_sqft.unique()


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


df2[~df2['total_sqft'].apply(is_float)].head(10)


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


df3 = df2.copy()
df3.total_sqft = df3.total_sqft.apply(convert_sqft_to_num)
df3 = df3[df3.total_sqft.notnull()]
df3.head(5)

df2.total_sqft[30]

(2100 + 2850) /2

df3.total_sqft[30]

"""# Feature Engineering"""

df3['Price_per_Sqft'] = (df3.price*100000)/df3.total_sqft

df3[:5]

df3.location.nunique()

df3.location = df3.location.apply(lambda x: x.strip())

df3

locations = df3.location.value_counts(ascending=False)

locations

locations.shape

locations_less_than_10 = locations[locations<=10]

locations_less_than_10

locations_less_than_10.shape

df3.location = df3.location.apply(lambda x: "Other" if x  in locations_less_than_10 else x)

df3[:5]

df3.location.nunique()                  # Before 1298 unique locations are there but now we have only 241 locations.

"""# Outlier REMOVAL"""

df3.BHK = df3.BHK.astype('float')

df3.shape

df4 = df3[~(df3.total_sqft/df3.BHK<300)]

df4.shape

df4.Price_per_Sqft.describe()    # Minimum price per sqft is 267 which is said to be an outlier because in real Bengalru doesn't have this much low price.


def Outlier_handling(df):
    df_new = pd.DataFrame()
    for name, subdf in df.groupby('location'):
        mu = np.mean(subdf.Price_per_Sqft)
        std = np.std(subdf.Price_per_Sqft)
        reduced_df = subdf[(subdf.Price_per_Sqft>(mu-std)) & (subdf.Price_per_Sqft<(mu+std))]          # Keep all the data falls under 1 std.
        df_new = pd.concat([df_new, reduced_df], ignore_index=True)
    return df_new


df5 = Outlier_handling(df4)

df5

df5.shape

bhk2 = df5[(df5.location=='Yeshwanthpur') & (df5.BHK==2)]

bhk3 = df5[(df5.location=='Yeshwanthpur') & (df5.BHK==3)]

plt.figure(figsize=(10, 4))

plt.scatter(bhk2.total_sqft,bhk2.price,label='2 BHK',color='green')

plt.scatter(bhk3.total_sqft,bhk3.price,label='3 BHK',marker='+',color='red')

plt.xlabel("Total Square Feet Area")
plt.ylabel("Price (Lakh Indian Rupees)")
plt.title("location")

plt.legend()


def remove_bhk_outliers():
  exclude_indices = np.array([])
  for lname,ldf in df5.groupby('location'):
    bhk_stats = {}
    for bname,bdf in ldf.groupby('BHK'):
      bhk_stats[bname] = {
          'mean' : np.mean(bdf.Price_per_Sqft),
          'std' : np.std(bdf.Price_per_Sqft),
          'count' : bdf.shape[0]
      }
    for bname, bdf in ldf.groupby('BHK'):
        stats = bhk_stats.get(bname-1)
        if stats and stats['count']>5:
            exclude_indices = np.append(exclude_indices, bdf[bdf.Price_per_Sqft>(stats['mean'])].index.values)
    return df5.drop(exclude_indices, axis='index')


df6 = remove_bhk_outliers()

df6

bhk2 = df6[(df6.location=='Yeshwanthpur') & (df6.BHK==2)]

bhk3 = df6[(df6.location=='Yeshwanthpur') & (df6.BHK==3)]

plt.figure(figsize=(10, 4))

plt.scatter(bhk2.total_sqft, bhk2.price, label='2 BHK', color='green')

plt.scatter(bhk3.total_sqft, bhk3.price, label='3 BHK', marker='+', color='red')

plt.xlabel("Total Square Feet Area")
plt.ylabel("Price (Lakh Indian Rupees)")
plt.title("location")

plt.legend()

df6['bath'].hist(bins=30)

df7 = df6[df6.bath<df6.BHK+2]

df7[:5]

df7.location.nunique()

"""# Model Building"""

dummies = pd.get_dummies(df7.location)

dummies[:5]

df8 = pd.concat([df7, dummies.drop('Other', axis=1)], axis=1)

df8[:3]

df8.shape

df8.columns

df9 = df8.drop('location', axis=1)

df9[:5]

X = df9.drop('price', axis='columns')

y = df9.price


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


dr = DecisionTreeRegressor().fit(X_train, y_train)

folds = KFold(n_splits=4, shuffle=True)

cv_score = cross_val_score(dr, X_train, y_train, cv=folds)

cv_score

cv_score.mean()


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]         # Taking column index of location

    x = np.zeros(len(X.columns))

    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index>=0:
        x[loc_index] = 1
    return dr.predict([x])[0]


predict_price('Sarjapur', 5000, 2, 2)

predict_price('Singasandra',5000,2,2)

predict_price('1st Block Jayanagar',5000,2,2)

predict_price('1st Phase JP Nagar',5000,2,2)

loc_index = np.where(X.columns=='1st Phase JP Nagar')[0][0]


with open('begaluru_prediction_pickle', 'wb') as f:
    pickle.dump(dr, f)


columns = {
    'data_columns': [x.lower() for x in X.columns]
}
with open('columns.json', 'w') as j:
    j.write(json.dumps(columns))
