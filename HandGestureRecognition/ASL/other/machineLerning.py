from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train(data):
    # Split the data into features and labels
    X = []
    Y = []
    for gesture in data.keys():
        label = gesture
        for item in data[gesture]:
            features = item['FeatureCombinations']['C1']  # You can choose any feature combination here
            X.append(features)
            Y.append(label)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train and evaluate multiple classification models
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Support Vector Machine', SVC()),
        ('Random Forest', RandomForestClassifier()),
        ('Neural Network', MLPClassifier(max_iter=1000))
    ]

    results = {}

    for model_name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy

    # Print the results
    for model_name, accuracy in results.items():
        print(f"{model_name}: Accuracy = {accuracy:.2f}")
