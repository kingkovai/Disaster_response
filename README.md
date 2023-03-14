# Disaster_response

## Project motivation
   At the time of diaster team handling it need to quickly classify the nature of text message that they receive so  they can quickly responds to it . In this project I intend to create an application that would take an text message as input in Web UI and outputs related labels for the text message.For that I have build [multioutput classifier model](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) and trained it based on the available dataset.

 
#### Installations
 - Pandas ,Flask ,sklearn ,plotly
#### Dataset
 Based on the provided input files , data is cleaned/transformed and used for training the model
 
#### Files in the project 
app\
| - template\
| |- master.html # main page of web app\
| |- go.html # classification result page of web app\
|- run.py # Flask file that runs app\
data\
|- disaster_categories.csv # Input dataset of categories\
|- disaster_messages.csv # Input dataset of message\
|- process_data.py # Python file for ETL processing \
|- DisasterResponse.db # database to save clean data to\
models\
|- train_classifier.py # Python file to train and save the classifier model \
|- classifier.pkl # saved model\
README.md 
 
#### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database \
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves \
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

 
#### Github
  - [Link of Github](https://github.com/kingkovai/Disaster_response.git)
 


