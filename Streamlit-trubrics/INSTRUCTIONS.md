# AI Engineer Intern - Technical Test

## Objective

Your task is to develop an application that assists homeowners in identifying the types of [iris](https://www.kaggle.com/datasets/uciml/iris) flowers growing in their garden. As the ML model you'll be using is not perfect, you need to incorporate user feedback within the application to help to identify issues. Keep in mind that the application should be user-friendly and easily understandable by individuals who are not familiar with ML.

## Install

```
pip install "trubrics[streamlit]"
```

## Requirements

- The app must be built with [Streamlit](https://streamlit.io/) (in Python).
- Users must be able to insert flower measurements into the application in order to identify the iris type. See an example list of measurements in `X.csv`
- Utilise the pre-trained model `model.pickle` for the classification task. The model has be trained on data formatted as shown in `X.csv` and `y.csv`.
- Implement the [trubrics-streamlit integration](https://trubrics.github.io/trubrics-sdk/streamlit/) to gather user feedback and store it in the local filesystem.
- We've provided a template in [app.py](./app.py) for you to use as a starting point. Feel free to modify it as needed.
- Ensure that your application can be executed by anyone. Provide comprehensive documentation to facilitate code reproduction.

## Submission

To submit, push your code to a private GitHub repository and share it with Trubrics (jeff.kayne@trubrics.com & joel.hodgson@trubrics.com) up to 2 hours before your technical interview. We will then go through the solution together in the technical interview. The estimated time to complete the test is 2 hours.
