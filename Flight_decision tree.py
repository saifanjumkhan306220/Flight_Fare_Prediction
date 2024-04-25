#!/usr/bin/env python
# coding: utf-8

# In[2]:


print(pd.__version__)


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[2]:


train= pd.read_csv(r"C:\Users\kalpa\Desktop\flightData.csv")
print(train.head())


# In[3]:


pd.set_option('display.max_columns', None)
train.head()


# In[4]:


train.info()


# In[5]:


train["Duration"].value_counts()


# In[6]:


train.dropna(inplace = True)
train.isnull().sum()


# In[7]:


train["Journey_day"] = pd.to_datetime(train.Date_of_Journey, format="%d/%m/%Y").dt.day
train["Journey_month"] = pd.to_datetime(train["Date_of_Journey"], format = "%d/%m/%Y").dt.month
train.head()


# In[8]:


train.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[9]:


# Extracting Hours
train["Dep_hour"] = pd.to_datetime(train["Dep_Time"]).dt.hour

# Extracting Minutes
train["Dep_min"] = pd.to_datetime(train["Dep_Time"]).dt.minute

# Now we can drop Dep_Time as it is of no use
train.drop(["Dep_Time"], axis = 1, inplace = True)

train.head()


# In[10]:


# Extracting Hours
train["Arrival_hour"] = pd.to_datetime(train.Arrival_Time).dt.hour

# Extracting Minutes
train["Arrival_min"] = pd.to_datetime(train.Arrival_Time).dt.minute

# Now we can drop Arrival_Time as it is of no use
train.drop(["Arrival_Time"], axis = 1, inplace = True)

train.head()


# In[11]:


# Assigning and converting Duration column into list
duration = list(train["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[12]:


# Adding duration_hours and duration_mins list to train_data dataframe

train["Duration_hours"] = duration_hours
train["Duration_mins"] = duration_mins

train.drop(["Duration"], axis = 1, inplace = True)


# In[13]:


# As Airline is Nominal Categorical data we will perform OneHotEncoding

Airline = train[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()


# In[14]:


# As Source is Nominal Categorical data we will perform OneHotEncoding

# Perform one-hot encoding
Source = train[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()


# In[15]:


# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination = train[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()


# In[16]:


# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other

train.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[17]:


train.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[18]:


train1= pd.concat([train, Airline, Source, Destination], axis = 1)


# In[19]:


train1.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[21]:


#selecting variables that have data types float and int.

var=list(train1.select_dtypes(include=['float64','int64']).columns)
from sklearn.preprocessing import PowerTransformer
sc_X=PowerTransformer(method = 'yeo-johnson')
train1[var]=sc_X.fit_transform(train1[var])


# # working with test data

# In[22]:


test = pd.read_csv(r"C:\Users\kalpa\Desktop\flightData_test.csv")
test.head()


# In[23]:


print(test.info())


# In[24]:


test.dropna(inplace = True)
print(test.isnull().sum())


# In[25]:


# Date_of_Journey
test["Journey_day"] = pd.to_datetime(test.Date_of_Journey, format="%d/%m/%Y").dt.day
test["Journey_month"] = pd.to_datetime(test["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test["Dep_hour"] = pd.to_datetime(test["Dep_Time"]).dt.hour
test["Dep_min"] = pd.to_datetime(test["Dep_Time"]).dt.minute
test.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test["Arrival_hour"] = pd.to_datetime(test.Arrival_Time).dt.hour
test["Arrival_min"] = pd.to_datetime(test.Arrival_Time).dt.minute
test.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[26]:


# Duration
duration = list(test["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[27]:


# Adding Duration column to test set
test["Duration_hours"] = duration_hours
test["Duration_mins"] = duration_mins
test.drop(["Duration"], axis = 1, inplace = True)


# In[28]:


# applying one-hot encoding on Airline-Test dataset

# Perform one-hot encoding
Airline = pd.get_dummies(test["Airline"], drop_first=True)
Airline.head()


# In[29]:


# applying one-hot-encoding on Source-Test data

# Perform one-hot encoding
Source = pd.get_dummies(test["Source"], drop_first=True)
Source.head()


# In[30]:


# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination = pd.get_dummies(test["Destination"], drop_first=True)
Destination.head()


# In[31]:


test.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[32]:


test.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[33]:


test1 = pd.concat([test, Airline, Source, Destination], axis = 1)


# In[34]:


test1.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[35]:


test1.shape


# In[36]:


test1.head()


# In[37]:


train1.shape


# In[38]:


train1.head()


# In[39]:


train1.columns


# In[40]:


X = train1.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


# In[41]:


y = train1.iloc[:, 1]
y.head()


# In[42]:


# X1 = train1.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
#        'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
#        'Duration_mins', 'IndiGo',
#        'Jet Airways','Air India',
#        'Multiple carriers', 'Delhi',
#        'Cochin', 'New Delhi',"Mumbai",'Hyderabad','SpiceJet','Jet Airways Business',]]


# In[43]:


#splitting our dataset in 80% training and 20% testset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[44]:


X_train.shape


# In[45]:


# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (25, 25))
sns.heatmap(train1.corr(), annot = True, cmap = "RdYlGn")
plt.show()


# In[ ]:





# # Fitting the model

# In[46]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[49]:


# Performing GridSearchCV on Decision Tree Regression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

depth = list(range(3,30))
param_grid = dict(max_depth = depth)
tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv = 10)
tree.fit(X_train,y_train)


# In[50]:


# Predicting train and test results
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)


# In[54]:


# Calculating Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[56]:


from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

print("Train Results for Decision Tree Regressor Model:")
print("Root Mean squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("Mean Absolute % Error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))


# In[58]:


print("Test Results for Decision Tree Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test, y_test_pred)))
print("Mean Absolute % Error: ", round(mean_absolute_percentage_error(y_test, y_test_pred)))
print("R-Squared: ", r2_score(y_test, y_test_pred))

