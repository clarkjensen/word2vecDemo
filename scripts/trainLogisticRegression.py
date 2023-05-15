from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import is_string_dtype, is_numeric_dtype
from io import StringIO

dataset_file = "../weatherAUS.csv"
abspath = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(abspath, dataset_file)
df = pd.read_csv(data_file)

# address missing data

# missing_count = df.isnull().sum() # the count of the missing values
# value_count = df.count() # the count of all values
# missing_percentage = round(missing_count / value_count * 100, 1) # the percentage of missing values
# missing_df = pd.DataFrame({'count': missing_count, 'percentage': missing_percentage}) # create a dataframe
# print(missing_df)

# drop columns with large amounts of missing values
df = df.drop(['Evaporation', 'Sunshine', 'Cloud3pm', 'Cloud9am'], axis=1)
# drop rows with missing labels (if it doesn't have the "RainTomorrow" label, that row is useless to train with)
df = df.dropna(subset = ["RainTomorrow"])

for column in df:
    if column != 'RainTomorrow':
        if is_numeric_dtype(df[column]):
            # numerical variables: impute missing values with mean
            if df[column].isnull().any():
                 df[column].fillna(df[column].mean(), inplace=True)
        elif is_string_dtype(df[column]):
            # categorical variables: replace missing values with "Unknown"
             if df[column].isnull().any():
                    df[column].fillna("Unknown", inplace=True)

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    # df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = int((nCol + nGraphPerRow - 1) / nGraphPerRow)  # convert nGraphRow to an integer
    plt.figure(num = None, figsize = (4 * nGraphPerRow, 2 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        # if nunique[i] > 1 and nunique[i] < 50: # For displaying purposes, pick columns that have between 1 and 50 unique values
            plt.subplot(nGraphRow, nGraphPerRow, i + 1)
            columnDf = df.iloc[:, i]
            valueCounts = columnDf.value_counts()
            # if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            if is_numeric_dtype(columnDf.iloc[0]):
                valueCounts.plot.hist()
            elif is_string_dtype(columnDf.iloc[0]):
                #show only the top 10 value count in each categorical data column
                valueCounts[:10].plot.hist()
            # if is_numeric_dtype(df[column]):
            #     df[column].plot(kind = 'hist')
            # elif is_string_dtype(df[column]):
            #     # show only the top 10 value count in each categorical data column
            #     df[column].value_counts()[:10].plot(kind = 'bar')
            plt.ylabel('counts')
            plt.xticks(rotation = 90)
            plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# visiualization of input data -- used to find outliers or skewed distribution
# plotPerColumnDistribution(df, 24, 4)

# address outliers in "Rainfall"
maximum = df['Rainfall'].quantile(0.9)
df = df[df["Rainfall"] < maximum]
# df["Rainfall"].plot(kind = 'hist')
# plt.show()

# date manipulation (cardinality of Date is too high)
df['Month'] = pd.to_datetime(df['Date']).dt.month.apply(str)
df = df.drop('Date', axis=1)
# df['Month'].value_counts().plot(kind = 'bar')
# plt.show()

# encoding categorical data using dummies
categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'Month', 'RainTomorrow']
for i in categorical_features:
     df[i] = LabelEncoder().fit_transform(df[i])
# print(df.info())

# Correlation matrix -- logistic regression requires there to be little multicollinearity among predictors
def plotCorrelationMatrix(df, graphWidth):
    # df = df.dropna('columns') # drop columns with NaN
    # df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {dataset_file}', fontsize=15)
    plt.show()

# plotCorrelationMatrix(df, 8)

# drop columns that are too correlated (and rearrange the others) -- got rid of Pressure9am, Temp9am, Temp3pm
df = df[['Month', 'Location', 'MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
         'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure3pm', 'RainToday', 'RainTomorrow']]
# print(df.info())

# separate training and testing data
X = df.iloc[:, :-1] # gets all of the features (columns except the last one)
Y = df["RainTomorrow"] # gets the label
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# logicist regression model
reg = LogisticRegression(max_iter= 500)
reg.fit(X_train, Y_train)
y_pred = reg.predict(X_test)
# evaluation

# confusion matrix
# confusion_matrix = ConfusionMatrixDisplay.from_estimator(reg, X_test, Y_test, cmap = "GnBu")
# # we want the top-left (predicted no rain when there was no rain) and top-right (predicted rain
# when there was rain) to be high, and the others to be low
# plt.show()

# accuracy -- we want this as close to 1 as possible
# print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))

# ROC and AUC -- 
# y_pred_prob = reg.predict_proba(X_test)[:,1]
# fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_prob)
# plt.plot(fpr, tpr)
# plt.show()
# auc = metrics.roc_auc_score(Y_test, y_pred_prob)
# print("AUC:", round(auc, 2))

# test manually

def predict_rain(tdf, reg):
    # drop columns with large amounts of missing values
    cols_to_drop = ['Evaporation', 'Sunshine', 'Cloud3pm', 'Cloud9am']
    for col in cols_to_drop:
        if col in tdf.columns:
            tdf.drop(col, axis=1, inplace=True)
    
    for column in tdf:
        if column != 'RainTomorrow':
            if is_numeric_dtype(tdf[column]):
                # numerical variables: impute missing values with mean
                if tdf[column].isnull().any():
                    tdf[column].fillna(tdf[column].mean(), inplace=True)
            elif is_string_dtype(tdf[column]):
                # categorical variables: replace missing values with "Unknown"
                if tdf[column].isnull().any():
                    tdf[column].fillna("Unknown", inplace=True)

    if 'Date' in tdf.columns:
        # date manipulation (cardinality of Date is too high)
        tdf['Month'] = pd.to_datetime(tdf['Date']).dt.month.apply(str)
        tdf = tdf.drop('Date', axis=1)

    # encoding categorical data using dummies
    categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'Month', 'RainTomorrow']
    for i in categorical_features:
        tdf[i] = LabelEncoder().fit_transform(tdf[i])

    # drop columns that are too correlated (and rearrange the others) -- got rid of Pressure9am, Temp9am, Temp3pm
    tdf = tdf[['Month', 'Location', 'MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
               'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure3pm', 'RainToday']]
    prediction = reg.predict(tdf)
    if prediction == 1:
        return "It will rain tomorrow."
    else:
        return "It will not rain tomorrow."


TESTDATA = StringIO("""Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday,RainTomorrow
2008-12-14,Albury,12.6,21,3.6,NA,NA,SW,44,W,SSW,24,20,65,43,1001.2,1001.8,NA,7,15.8,19.8,Yes,No""") # predicts no
#2008-12-12,Albury,15.9,21.7,2.2,NA,NA,NNE,31,NE,ENE,15,13,89,91,1010.5,1004.2,8,8,15.9,17,Yes,Yes""") # predicts yes
tdf = pd.read_csv(TESTDATA)
print(predict_rain(tdf, reg))