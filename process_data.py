import sys

import nltk
import re
import pandas as pd
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Input:
          messages_filepath - Path of message dataset along with filename
          categories_filepath - Path of category dataset along with filename
    Output:
           merged dataset as Panda dataframe
    """
    messages =pd.read_csv(messages_filepath)
    categories =pd.read_csv(categories_filepath)
    merge_df = messages.merge(categories,how='inner', on='id')
    return merge_df
    
    


def clean_data(df):
    """
    Transform the passed on dataframe by spliting message into multiple indicator fields with value 
    """
    categories = df.categories.str.split(pat =';',expand=True)
    category_colnames = list(categories.iloc[0,:].str.replace('-1','').str.replace('-0',''))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df=df.drop(['categories'], axis=1)
    # Join the message dataset and new derived Indicator fields
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df=df[~df.duplicated()]
    
    return df


def save_data(df, database_filename):
    """
    Create Sqlite DB and save the cleaned data as table
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('cleanedmessage', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()