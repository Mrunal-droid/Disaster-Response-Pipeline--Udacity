# Dog-breed-classifier--Udacity

 1. Installation
 2. Project overview
 3. File descriptions
 4. Instructions
 5. Results
 
 # Installation
 Libraries used for this project are: Python, Numpy, re, pickle, nltk, flask, json,plotly, sklearn, sqlalchemy, sys and warnings.
 
 # Project Overview
 
 The project is designed to create a web app and using this app to classify the disaster apps so that the emergency warriors can help people. The model used pipelines to iniate the progress.
 
 # File Description
 
 - process_data.py:- It contains executable code which has etl pipeline that inputs messages data, message categories and creates a SQL database.
 - train_classifier.py: It contains executable code which has ML pipeline which extracts tables from the created database.
 - run.py: This file has uses the plotly files to improvise the look of app.
 
 # Instructions
 1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db.
    - To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl.
 2. Run the following command in the app's directory to run your web app. python run.py
 
  
# Results
 The web app runs successfully.
 You can access my app [here](https://view6914b2f4-3001.udacity-student-workspaces.com/go?query=earthquake)
 
