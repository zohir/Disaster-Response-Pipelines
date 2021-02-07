# Disaster Response Pipeline Project

Github repository
https://github.com/zohir/Disaster-Response-Pipelines


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### LIBRARIES:
  - python=3.6.12
  - numpy=1.12.1
  - pandas=0.23.3
  - plotly=4.14.3
  - scikit-learn=0.19.1
  - sqlalchemy=1.1.13
  - pickleshare=0.7.5
  - flask=1.1.2

