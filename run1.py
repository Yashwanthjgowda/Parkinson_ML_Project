import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset and prepare the model (same as the original code)
df = pd.read_csv("C://Users//Yashwanth//Downloads//parkinsons data.csv")

# Splitting the dataset into features and target variable
X = df.drop(columns=['name','status'], axis=1)
Y = df['status']

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Scaling the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Training the SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Getting the accuracy score on training data
X_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, X_pred)

# Getting the accuracy score on test data
X_pred1 = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_pred1)

# Streamlit App
st.title("Parkinson's Disease Prediction")

st.write(f"### Accuracy on Training Data: {train_accuracy * 100:.2f}%")
st.write(f"### Accuracy on Test Data: {test_accuracy * 100:.2f}%")

# Create input fields for all features in a single field
st.write("Enter all 22 feature values separated by commas:")

input_df = st.text_input('Input all features (separated by commas)')

# Create a button to submit the input and get prediction
submit = st.button("Submit")

if submit:
    try:
        # Split the input string into a list of values and convert them to floats
        input_df_lst = input_df.split(',')
        
        # Ensure the user input is valid (should have 22 features)
        if len(input_df_lst) == 22:
            input_values = [float(i) for i in input_df_lst]  # Convert to float values
            
            # Convert the inputs to a numpy array and scale them
            input_array = np.array(input_values).reshape(1, -1)
            scaled_input = scaler.transform(input_array)

            # Make the prediction
            prediction = model.predict(scaled_input)

            # Show the result
            if prediction[0] == 0:
                st.write("The person does not have Parkinson's Disease.")
            else:
                st.write("The person has Parkinson's Disease.")
        else:
            st.write("Please enter exactly 22 feature values separated by commas.")

    except ValueError:
        st.write("Invalid input. Please ensure all values are numeric and separated by commas.")
