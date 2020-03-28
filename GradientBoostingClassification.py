# importing libraries

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# Loading titanic dataset (test and train) :

train_data = pd.read_csv("train.csv")	# using pandas to read csv files and saving them.
test_data = pd.read_csv("test.csv")

'''
Our label data, the y data is the Survived column. So we'll make that it's own dataframe and then remove it from the features:
'''

y_train = train_data["Survived"]	# saving column from the Train_data into new data.
train_data.drop(labels="Survived", axis=1, inplace=True)	# dropping "survived" column from train_data; axis = 1 means columns-wise ; inplace = true means changes are done in original dataset. 

# Now we have to create a concatenated new data set:

full_data = train_data.append(test_data)	#appending test_data in the latest train_data.

# Let's drop any columns that aren't necessary or helpful for training, although you could leave them in and see how they affect things:

drop_columns = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]
full_data.drop(labels=drop_columns, axis=1, inplace=True)	# dropping unnecessary columns.

# Any text data needs to be converted into numbers that our model can use, so let's change that now. We'll also fill any empty cells with 0:

full_data = pd.get_dummies(full_data, columns=["Sex"])	# 
full_data.fillna(value=0.0, inplace=True)	# filling empty value with 0

# Let's split the data into training and testing sets:

X_train = full_data.values[0:891]
X_test = full_data.values[891:]

# We'll now scale our data by creating an instance of the scaler and scaling it:

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

''' Now we can split the data into training and testing sets. 
Let's also set a seed (so you can replicate the results) and select the percentage of the data for testing on:
'''

state = 12  
test_size = 0.30  
  
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  
    test_size=test_size, random_state=state)

'''Now we can try setting different learning rates, so that we can compare the performance of the classifier's performance at different learning rates.
''' 

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))


# Let's see what the performance was for different learning rates:
'''
Learning rate:  0.05
Accuracy score (training): 0.801
Accuracy score (validation): 0.731

Learning rate:  0.075
Accuracy score (training): 0.814
Accuracy score (validation): 0.731

Learning rate:  0.1
Accuracy score (training): 0.812
Accuracy score (validation): 0.724

Learning rate:  0.25
Accuracy score (training): 0.835
Accuracy score (validation): 0.750

Learning rate:  0.5
Accuracy score (training): 0.864
Accuracy score (validation): 0.772

Learning rate:  0.75
Accuracy score (training): 0.875
Accuracy score (validation): 0.754

Learning rate:  1
Accuracy score (training): 0.875
Accuracy score (validation): 0.739
''' 

'''We're mainly interested in the classifier's accuracy on the validation set, but it looks like a learning rate of 0.5 gives us the best performance on the validation set and good performance on the training set.
'''

''' Now we can evaluate the classifier by checking its accuracy and creating a confusion matrix. Let's create a new classifier and specify the best learning rate we discovered.
'''

gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))

''' 
Here's the output of our tuned classifier:

Confusion Matrix:
[[142  19]
 [ 42  65]]
Classification Report
              precision    recall  f1-score   support

           0       0.77      0.88      0.82       161
           1       0.77      0.61      0.68       107

    accuracy                           0.77       268
   macro avg       0.77      0.74      0.75       268
weighted avg       0.77      0.77      0.77       268
'''
