# import libraries
import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# download necessary NLTK data
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet'])

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report ,precision_score,accuracy_score,recall_score

import pickle


def load_data(database_filepath):
    """
    Loads the SqliteDB table based on given filepath
    Input:
          database_filepath - Path of the DB file
    Output:
          X - Dataframe of input features
          Y - Target variables
          List of target variable names
    """
#     engine = create_engine('sqlite:///InsertDatabaseName.db')
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('cleanedmessage',engine)
    
    print( "Target label that dont have variance" + str( list(df.columns[df.sum()==0]) )  )
    X = df.message
    #Drop child_alone feature as they only have one class
    Y = df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military',  'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']]
    
    #feature related which is an indicator field is having value as 2 .As that is not possible replacing 2 with 1
    Y.loc[Y.related==2,'related']=1
    return X,Y ,list(Y.columns)


def tokenize(text):
    '''
    Input:
          Raw text message
    Output:
          Tokenized text
    '''
    clean_tkn = []
    tken = word_tokenize(text)
    lmtzr = WordNetLemmatizer()
    
    for tok in tken:
        clean_tk = lmtzr.lemmatize(tok).lower().strip()
        clean_tkn.append(clean_tk)

    return clean_tkn


def build_model():
    """
    Builts and return cross validation pipeline with data transformation and classification steps
    """
    vect = CountVectorizer(tokenizer=tokenize)
    tfid_trn = TfidfTransformer()
    clf = MultiOutputClassifier(LogisticRegression())
    
    pipeline = Pipeline([
        ('vect_stp', CountVectorizer(tokenizer=tokenize)),
        ('tfid_trn_stp', TfidfTransformer()),
        ('clf_stp', MultiOutputClassifier(LogisticRegression()) )] )
    
    parameters = {'clf_stp__n_jobs': [1, 2, 4] }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    
def f_score(func,y_test,y_hat):
    """
    Input:
          func - metric function 
          y_test - Actual target variable
          y_hat  - predicted target values
    Output:
          returns metric for all the target variable as dataframe
    """
    score_df=pd.DataFrame( np.zeros(y_test.shape[1]) )
    for x in np.arange(y_test.shape[1]):
        score_df.loc[x,0]=func(y_test.iloc[:,x],y_hat.iloc[:,x])
    return score_df

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Displays the Accuracy , recall and precision metrics for each of the target variable
    Input:
          model - trained model for evaluation
          X_test - Dataframe of input features
          Y_test - Actual target values
          category_names - List of target variable names
    """
    Y_test_pred=pd.DataFrame(model.predict(X_test) ,columns=category_names )
    
    recall_met=f_score(recall_score,Y_test,Y_test_pred)
    prec_met=f_score(precision_score,Y_test,Y_test_pred)
    acc_met=f_score(accuracy_score,Y_test,Y_test_pred)

    result = pd.concat([acc_met,recall_met, prec_met], axis=1, join='inner')

    result.columns=['accuracy','recall','precision']
    result.set_axis( category_names , inplace=True )
    print('Metrics of each of the target variable')
    print('-------------------------------------------------------------------------------------------')
    print(result)
    print('-------------------------------------------------------------------------------------------')


def save_model(model, model_filepath):
    """Save model in path using pickle"""
    pickle.dump( model, open(model_filepath, 'wb') )
    


def main():
    """
    Main execution block
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        #select the best estimator from the cross validation
        bst_model=model.best_estimator_
        
        print('Evaluating model...')
        evaluate_model(bst_model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(bst_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()