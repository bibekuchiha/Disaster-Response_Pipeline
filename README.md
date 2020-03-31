# Disaster Response Pipelines

--------------------------------------
<h4> 1. [Description]</h4>
<h4>2. [Project Motivation]</h4>
<h4>3. [File Descriptions]</h4>
<h4>4. [Installation]</h4>
<h3> 1. Description <a name="description"></a></h3>
<br>
 Disaster Response Pipeline Project from Udacity's Data Science Nanodegree. 
</br>
<br>
We will make an AI pipeline to arrange calamity occasions with the goal that we can send the messages to a proper catastrophe alleviation office. We will prepare on informational index containing genuine messages that were sent during catastrophe occasions from Figure Eight.
</br>
<h3> 2. Project Motivation <a name="motivation"></a></h3>  

Create a machine learning pipeline to categorize these events so that users can send the messages to an appropriate disaster relief agency.
<h3> 3. File Descriptions <a name="files"></a> </h3>  

> * **data/disaster_messages.csv data/disaster_categories.csv :** original data
> * **data/process_data.py:** to run ETL pipeline that cleans data and stores in database
> * **data/DisasterResponse.db:** database that stores cleaned data 
> * **models/train_classifier.py:** to run ML pipeline that trains classifier and saves
> * **models/classifier.pkl:** a pickle file which saves model
> * **data/:** a Flask framework for presenting data
<h3> 4. Instructions</h3>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Code structure

- app
  - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py
  - InsertDatabaseName.db   # database to save clean data to

- models
  - train_classifier.py
  - classifier.pkl  # saved model 

- README.md
