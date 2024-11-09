import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df= pd.read_csv("Crop_recommendation.csv")
#split the dataset
x= df.iloc[:, :-1] #features
y= df.iloc[:, -1] #labels
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state= 42)
model = RandomForestClassifier()
model.fit(x_train, y_train)
pickle.dump(model, open("model.pkl", "wb"))