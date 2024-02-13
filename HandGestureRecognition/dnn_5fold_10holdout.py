from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)
y_onehot = to_categorical(all_labels_encoded)

# Function to create the DNN model
def create_dnn(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare a dictionary to hold the cross-validation results
cross_val_results = {}

for combination, X_comb in all_data_combinations.items():
    # Convert to NumPy array
    X_comb_np = np.array(X_comb).astype('float32')

    # Split data into 90% for cross-validation, 10% for holdout test
    X_cv, X_holdout_test, y_cv, y_holdout_test = train_test_split(X_comb_np, y_onehot, test_size=0.10, random_state=42, stratify=y_onehot)

    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Prepare to collect the last fold's data
    last_fold_history = None
    last_fold_X_val = None
    last_fold_y_val = None
    last_fold_model = None

    # Perform K-Fold Cross-Validation
    for fold, (train_index, val_index) in enumerate(kf.split(X_cv), start=1):
        X_train, X_val = X_cv[train_index], X_cv[val_index]
        y_train, y_val = y_cv[train_index], y_cv[val_index]

        # Define model
        model = create_dnn(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

        # Fit model
        history = model.fit(X_train, y_train, epochs=75, verbose=0, validation_data=(X_val, y_val))

        # If it's the last fold, keep the history and validation data for plotting
        if fold == 10:
            last_fold_history = history
            last_fold_X_val = X_val
            last_fold_y_val = y_val
            last_fold_model = model

    # Evaluate the model on the holdout test set
    holdout_test_loss, holdout_test_accuracy = last_fold_model.evaluate(X_holdout_test, y_holdout_test, verbose=0)

    # Store the results (including holdout test accuracy and loss)
    cross_val_results[combination] = {
        'validation_accuracy': last_fold_history.history['val_accuracy'][-1],
        'validation_loss': last_fold_history.history['val_loss'][-1],
        'test_accuracy': holdout_test_accuracy,
        'test_loss': holdout_test_loss
    }

    # Plot training and validation history for the last fold
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(last_fold_history.history['accuracy'], label='Training Accuracy')
    plt.plot(last_fold_history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(last_fold_history.history['loss'], label='Training Loss')
    plt.plot(last_fold_history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Performance for {combination}')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot confusion matrix for the holdout test set
    y_pred = last_fold_model.predict(X_holdout_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_holdout_test, axis=1)
    conf_mtx = confusion_matrix(y_true, y_pred_classes)
    plt.subplot(1, 2, 2)
    sns.heatmap(conf_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - Holdout Test Set for {combination}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

# Convert the cross-validation results to a DataFrame and display it
cross_val_results_df = pd.DataFrame.from_dict(cross_val_results, orient='index')
display(cross_val_results_df)
