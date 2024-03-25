import sys
import pandas as pd
import re
from source_language import detect_language
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    INPUT:
    messages_filepath - path to a csv file 
    categories_filepath = path to a csv file
    
    
    OUTPUT:
    A pandas DataFrame of the combind csv files           
    '''          
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = pd.merge(messages, categories, on='id')
    return df          


def clean_data(df):
    
    '''
    INPUT:
    df - merged pandas DataFrame
        
    
    OUTPUT:
    A clean pandas DataFrame with duplicates removed and categorical columns
    converted to binary representation (one-hot encoded)       
    '''     
            
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names from the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Convert category values to 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
        
    # Replace categories column in df with new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Define url pattern
    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Apply a function to replace url with a placeholder
    df['message'] = df['message'].apply(lambda x:re.sub(url_pattern, 'urlplaceholder', x))
    
    # Define punctuation pattern
    punctuation = '[^a-zA-Z0-9]'

    # Apply a function to remove anything not a lower or uppercase letter or integer
    df['message'] = df['message'].apply(lambda x:re.sub(punctuation, " ", x))

    # Apply a function to detect source language of original message
    df['source_language'] = df['original'].apply(lambda x: detect_language(x))
    
    # Remove duplicates
    df = df.drop_duplicates()
    return df    
           
      

def save_data(df, database_filename):
        
    '''
    INPUT:
    df - clean pandas DataFrame
       
    
    OUTPUT:
    .db - sqlite database file 
    '''      
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')
    
   
    


def main():
    # sys.arg refers to the additional arguments passed in to the command line with the file name
    # Note that sys.argv[0] refers to the file name and as such the 1st argument is accessed using sys.argv[1]
    
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
              'dbase/disaster_response.db')


if __name__ == '__main__':
    main()