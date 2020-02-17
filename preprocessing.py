import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy
# Load dataset
url = "data_arrhythmia.csv"
df = pandas.read_csv(url)

X = df.drop(["diagnosis"],axis=1)
Y = df.diagnosis.values



X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 20)



from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train,y_train)

y_pred=svm_clf.predict(X_test)
print ("Logistic regression", accuracy_score(y_test,y_pred) )

#creating and training a model
#serializing our model to a file called model.pkl
# Save your model
from sklearn.externals import joblib
joblib.dump(svm_clf, 'model.pkl')
print("Model dumped!")



