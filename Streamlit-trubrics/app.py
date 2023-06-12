import os
import joblib
import pandas as pd
import streamlit as st

# Set GIT_PYTHON_GIT_EXECUTABLE

#The environment variable ensures that the git module can function properly within the trubrics library and avoids any potential errors related to Git execution.
os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = "D:\\Program Files\\Git\\bin\\git.exe"

from trubrics.integrations.streamlit import FeedbackCollector 

@st.cache_resource
def load_artifacts():
    # Load the trained model and test data from files
    model = joblib.load("model.pickle")
    X_test = pd.read_csv("X.csv")
    y_test = pd.read_csv("y.csv")
    return model, X_test, y_test

def get_user_input():
    # Display a subheader for the input section
    st.subheader("Enter the flower measurements in cm")

    # Divide the input fields into two columns using streamlit.columns()
    col1, col2 = st.columns(2)

    with col1:
        # Display a number input field for sepal length
        sepal_length = st.number_input("Sepal Length", min_value=0.0, format='%f')
        # Display a number input field for petal length
        petal_length = st.number_input("Petal Length", min_value=0.0, format='%f')

    with col2:
        # Display a number input field for sepal width
        sepal_width = st.number_input("Sepal Width", min_value=0.0, format='%f')
        # Display a number input field for petal width
        petal_width = st.number_input("Petal Width", min_value=0.0, format='%f')

    # Create a DataFrame with the user input
    user_data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(user_data, index=[0])
    return features
 
def main():
    # Set the title of the application
    st.title("Iris Dataset ML Application")
    # Display an image
    st.image('image.png', width=600)
    # Display a text message
    st.write("### Please enter the measurements of your iris flower in centimeters:")
    
    # Load the model and test data
    model, X_test, _ = load_artifacts()
    
    # Get user input
    user_input = get_user_input()
    
    # Display the labels for Setosa, Versicolor, and Virginica
    st.write("Setosa:")
    st.write("Versicolor:")
    st.write("Virginica:")
    # Display the user input as a DataFrame
    st.dataframe(user_input)
    
    # Display a text message
    st.write("### Predicting...")
    # Make prediction using the model
    prediction = model.predict(user_input)

    # Display the predicted Iris class and corresponding image based on the prediction
    if prediction[0] == 0:
        st.success("The predicted Iris class is: Iris-setosa")
        st.image('setosa.jpg', width=200)
    elif prediction[0] == 1:
        st.success("The predicted Iris class is: Iris-versicolor")
        st.image('versicolor.jpg', width=200)
    elif prediction[0] == 2:
        st.success("The predicted Iris class is: Iris-virginica")
        st.image('virginica.jpg', width=200)

    # Step 2: Initialize the FeedbackCollector
    collector = FeedbackCollector()

    # Step 3: Collect feedback
    # Define a custom feedback question
    custom_question = "Custom feedback slider"
    # Display a slider for the custom question
    slider = st.slider(custom_question, max_value=10, value=5)
    # Display a button to save the feedback
    submit = st.button("Save feedback")

    # Save the custom feedback if the button is clicked and a value is selected
    if submit and slider:
        collector.st_feedback(
            "custom",
            user_response={custom_question: slider},
        )

    # Collect other types of feedback
    collector.st_feedback(feedback_type="faces")
    collector.st_feedback(feedback_type="thumbs")
    collector.st_feedback(feedback_type="issue")


if __name__ == '__main__':
    main()