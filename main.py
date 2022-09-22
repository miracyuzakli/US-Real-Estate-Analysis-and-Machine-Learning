#Import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load data
Data = pd.read_csv("realtor-data.csv")

print(Data.isnull().sum())
Data.dropna(inplace = True) # drop missing values

Data.duplicated(subset="full_address")

Data.drop_duplicates(subset=['full_address'], inplace = True)
Data.index = np.arange(Data.index.size)

Data.head(n = -20)



# Visualization and Analysis of Data
plt.style.use('default')



###########################################################################
plt.figure(figsize=(17, 10))
plt.suptitle("Number of Bedrooms and Bathrooms Distribution", fontsize = 15)

# # # # # # # # # # # # # # # # # # # # # # # # Bedroom

bed = Data.value_counts("bed")
labels = list()
for i, j in enumerate(bed):
    labels.append("{} - %{:.2f}".format(bed.index[i], 100*j/bed.sum()))
    
plt.subplot(2,1,1)
plt.bar(labels, bed)
plt.xticks(rotation = 20)
plt.xlabel("Bedroom")
plt.ylabel("Number")

# # # # # # # # # # # # # # # # # # # # # # # # Bathroom

bath = Data.value_counts("bath")
labels = list()
for i, j in enumerate(bath):
    labels.append("{} - %{:.2f}".format(bath.index[i], 100*j/bath.sum()))
    
plt.subplot(2,1,2)
plt.bar(labels, bath, color = "purple")
plt.xticks(rotation = 20)
plt.xlabel("Bathroom")
plt.ylabel("Number")



###########################################################################
plt.figure(figsize=(20, 8))
x, y, z = Data["bed"], Data["bath"], Data["price"]

sc = plt.scatter(x, y, c = z)
plt.xlabel("Bedroom")
plt.ylabel("Bathroom")
plt.colorbar(sc, label = "Price")
plt.title("Relation of Price to Bedroom and Bathroom", fontsize = 15)



###########################################################################
plt.figure(figsize = (15, 6))
stateCount = Data.value_counts("state")

for i, j in enumerate(bath):
    labels.append("{} - %{:.2f}".format(bath.index[i], 100*j/bath.sum()))

for i, j in enumerate(stateCount):
    label = "{} - %{:.2f}".format(stateCount.index[i], 100*j/stateCount.sum())
    plt.barh(label, j,height = 0.5)
    
plt.xlabel("Count")
plt.title("Real Estate Distribution by Province", fontsize = 15)





# I just create a new column with year value.
year = list()
for i in Data["sold_date"]:
    year.append(int(str(i)[:4]))
Data["year"] = year
print(Data.head(5))


###########################################################################
meanPrice = Data.groupby("year")["price"].mean()
medianPrice = Data.groupby("year")["price"].median()

meanPrice.index.sort_values(ascending = False)
medianPrice.index.sort_values(ascending = False)

plt.figure(figsize = (20, 12))
plt.suptitle("Median and Average Price by Years", fontsize = 15)

plt.subplot(2,1,1)
plt.bar(meanPrice.index, meanPrice, width = 0.2, color = "blue")
plt.bar(medianPrice.index+0.2, medianPrice, width = 0.2, color = "red")
plt.legend(["Mean", "Median"], fontsize = 15)

plt.subplot(2,1,2)
plt.scatter(meanPrice.index, meanPrice, color = "purple")
plt.scatter(medianPrice.index+1, medianPrice, color = "orange")
plt.plot(meanPrice.index, meanPrice, linestyle = ":", color = "purple")
plt.plot(medianPrice.index+1, medianPrice, linestyle = ":", color = "orange",)
plt.legend(["Mean", "Median"], fontsize = 15)



###########################################################################
plt.figure(dpi=100)
plt.title('Correlation Matrix')
sns.heatmap(Data.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')
plt.xticks(rotation=60)
plt.yticks(rotation = 60)
plt.show()


scaMat = pd.plotting.scatter_matrix(Data, alpha = 0.5, figsize=(18, 18), marker = ".")




# Machine Learning
LB = LabelEncoder()

Data["cityEnc"] = LB.fit_transform(Data["city"])
Data["stateEnc"] = LB.fit_transform(Data["state"])
print(Data.head(5))


# Train - Test Place
x = Data[["bed", "bath", "acre_lot", "stateEnc","zip_code", "house_size"]]
y = Data[["price"]]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.33, random_state = 0)


# Linear Regression
LR = LinearRegression()
LR.fit(xTrain, yTrain)
predLR = LR.predict(xTest)


# Decision Tree Regressor
DT = DecisionTreeRegressor(random_state = 0)  
DT.fit(xTrain, yTrain)
predDT = DT.predict(xTest)


# Random Forest Regressor
RF = RandomForestRegressor(n_estimators = 10, random_state=0)
RF.fit(xTrain , yTrain.values.ravel())
predRF = RF.predict(xTest)

# R-squared Scores
r2_0 = r2_score(yTest, predLR)
print("Linear Regression R-squared Score        : {}".format(r2_0))

r2_1 = r2_score(yTest, predDT)
print("Decision Tree Regressor R-squared Score  : {}".format(r2_1))

r2_2 = r2_score(yTest, predRF)
print("Random Forest Regressor R-squared Score  : {}".format(r2_2))

plt.barh(["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"], [r2_0, r2_1, r2_2],
        height = 0.3, color = ["#306E38", "#47924F", "#61B668"])
plt.xlim(0, 1)
plt.xlabel("Value")
plt.title("Scores")


























































































































































































































