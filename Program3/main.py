from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from keras.optimizers import SGD

df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_data.csv")


#getting rid of the ID column
df = df.drop(['id'], axis=1)
test_df = test_df.drop(['id'], axis=1)
#gender

def encoding_gender(item):
    if item == 'Male':
        return 0
    else:
        return 1


df["Gender"] = df["Gender"].apply(encoding_gender)
test_df["Gender"] = test_df["Gender"].apply(encoding_gender)
#df['Gender'].replace(0, 'Female',inplace=True)
#df['Gender'].replace(1, 'Male',inplace=True)
#test_df['Gender'].replace(0, 'Female',inplace=True)
#test_df['Gender'].replace(1, 'Male',inplace=True)

#age into zscore
df['Age'] = df['Age'] / df['Age'].max()
df['Age'] = (df['Age'] - df['Age'].mean())/df['Age'].std(ddof=0)
test_df['Age'] = test_df['Age'] / test_df['Age'].max()
test_df['Age'] = (test_df['Age'] - test_df['Age'].mean())/test_df['Age'].std(ddof=0)

#Driving License Good
df['Region_Code'] = df['Region_Code'].apply(lambda x: np.int(x))
test_df['Region_Code'] = test_df['Region_Code'].apply(lambda x: np.int(x))

#Previously insured good

#Vehicle_age preprocessed
vehicle_age = {'> 2 Years': 3, '1-2 Year': 2, '< 1 Year': 1}
df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age)
test_df['Vehicle_Age'] = test_df['Vehicle_Age'].map(vehicle_age)

#vechicle damage
def encoding_vehicle_dmg(item):
    if item == "No":
        return 0
    else:
        return 1


df["Vehicle_Damage"] = df["Vehicle_Damage"].apply(encoding_vehicle_dmg)
test_df["Vehicle_Damage"] = test_df["Vehicle_Damage"].apply(encoding_vehicle_dmg)

#standarize annual premium; these two methods give a .05 variation on numbers
#df['Annual_Premium'] = df['Annual_Premium'] / df['Annual_Premium'].max()
mm = MinMaxScaler()
df[['Annual_Premium']] = mm.fit_transform(df[['Annual_Premium']])
test_df[['Annual_Premium']] = mm.fit_transform(test_df[['Annual_Premium']])

#policy sales
df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].apply(lambda x: np.int(x))
test_df['Policy_Sales_Channel'] = test_df['Policy_Sales_Channel'].apply(lambda x: np.int(x))

#vintage
df[['Vintage']] = mm.fit_transform(df[['Vintage']])
test_df[['Vintage']] = mm.fit_transform(test_df[['Vintage']])
#response

print(df.head())

x = df.drop(['Response'],axis=1)
y = df['Response']


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.15, random_state=87)


preprocessor = ColumnTransformer(transformers = [('onehot', OneHotEncoder(), df['Response'])])

print(df.head())
from keras.layers import Flatten
from keras import layers
model = Sequential()
#model.add(Flatten())
#model.add(Dense(1, input_dim=10, activation='softmax'))
model.add(layers.Dense(1,input_dim=10, activation='relu'))

epochs = 25
lrate = 0.11
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train)  #,validation_data=(test_x, test_y), epochs=epochs, batch_size=32)


preds = model.predict(test_df)
y_pred_final_model = model.predict(x_test)
accuracy_score(y_test, y_pred_final_model)
# Final evaluation of the model
print(model.summary())
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


