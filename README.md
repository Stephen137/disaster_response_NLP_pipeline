# Disaster Response Pipeline Project

This project analyses a dataset containing real messages that were sent during disaster events provided by [Appen](https://www.figure-eight.com/) (formerly Figure 8). Some essential pre-processing is carried out via an ETL pipeline to get the raw data into shape for use in a machine learning pipeline that uses **NLTK**, as well as scikit-learn's **Pipeline** and **GridSearchCV** to build a multi-output classification model.

There are three components to this project, the details of which are included below.

## 1. Extract, Transform Load (ETL) Pipeline

A Python script which:

- loads the messages and categories csv files
- merges the two datasets
- cleans the data
- stores it in an SQLite database


## 2. Machine Learning (ML) Pipeline

A Python script which:

- loads the pre-processed data from the SQLite database
- splits the dataset into training and test sets
- builds a text processing and machine learning pipeline
- trains and tunes a model using GridSearchCV
- outputs results on the test set
- exports the final model as a pickle file for use in a web app


## 3. Flask Web App

The final project output is a web app which leverages the [Bootstrap library](https://getbootstrap.com/) and [Flask framework](http://flask.pocoo.org/) and:

- uses the model built in the previous step to classify user input text across 36 pre-defined categories. This tool could be used by emergency workers to direct resource allocation during disaster events 
- includes three interactive visualizations created using [Plotly](https://plotly.com/) 

### File Structure

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- source_language.py # detect message source language
|- disaster_response.db   # database to save clean data to

- models
|- train_classifier.py
|- random_forest.pkl  # saved model 

- README.md

## Reproducing the project locally

### **Step 1: Fork this repo**

Forking a GitHub repository is a common operation done to create your copy of someone else's repository on GitHub. Here's how you can do it:

1. Visit the Repository: https://github.com/Stephen137/disaster_response_NLP_pipeline

2. Fork the Repository: In the top-right corner of the page, you'll find a "Fork" button. Click on it. This action will create a copy of the repository under your GitHub account.

3. Wait for the Fork: GitHub will take a moment to fork the repository. Once it's done, you'll be redirected to your copy of the repository.

4. Clone Your Fork: Now that you have your fork, you can clone it to your local machine using Git. On the repository page, click the green "Code" button, and then copy the URL.

5. Open Terminal (or Command Prompt): Navigate to the directory where you want to store the cloned repository.

6. Clone the Repository: Use the git clone command followed by the URL you copied. It will look something like this:

`git clone https://github.com/your-username/repository-name.git`


### **Step 2: Run the ETL Pipeline**

Navigate to the project's root directory and run the following command : 

`python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`

### **Step 3: Run the ML Pipeline**

Navigate to the project's root directory and run the following command : 

`python3 models/train_classifier.py data/dbase/disaster_response.db models/classifier.pkl`

### **Step 4: View the Dashboard app**

Navigate to the app directory and run the following command: 

`python3 run.py`

The dashboard app should render locally at http://0.0.0.0:3001/

## Acknowledgements



