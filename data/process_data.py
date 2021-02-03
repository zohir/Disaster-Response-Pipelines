import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load, clean and merge the disasters messages and categories.
    
    Args:
        messages_filepath (str): name of the messages file including the path
        categories_filepath (str): name of the categories file including the path

    Returns:
        A dataframe with the merged data
    '''
    
    print('Loading data...\n    MESSAGES: {}'.format(messages_filepath))
    messages_raw = pd.read_csv(messages_filepath)
    print('        rows loaded from file = {}'.format(messages_raw.shape[0]))
    messages_df = clean_messages(messages_raw)
    print('        clean rows = {}'.format(messages_df.shape[0]))

    print('\nLoading data...\n    CATEGORIES: {}'.format(categories_filepath))
    categories_raw = pd.read_csv(categories_filepath)
    print('        rows loaded from file = {}'.format(categories_raw.shape[0]))
    categories_df = prepare_categories(categories_raw)
    print('        prepared rows = {}'.format(categories_df.shape[0]))

    print('\nMerging data...')
    df = messages_df.merge(categories_df,left_index=True, right_index=True)
    print('        merged rows = {}'.format(df.shape[0]))
    
    return df


def prepare_categories(categories_raw):
    '''
    Function to prepare the categories data. Spliting the unique field into multiple one-hot-encoded fields
    
    Args:
        categories_raw (dataframe): Dataframe containing the loaded categories
        
    Returns:
        A dataframe with the categories expanded
    '''

    def split_categories(row):
        ret = {}
        for kv in row.categories.split(';'):
            k, v = kv.split('-')
            ret[k]= v
        return pd.Series(ret)
    categories_df = categories_raw.apply(split_categories, axis=1)

    # convert column from string to numeric
    for column in categories_df:
        categories_df[column] = categories_df[column].astype('int32') 

    return categories_df


def clean_messages(messages_raw):
    '''
    Function to clean the messages loaded. 
     - Discard duplicated messages
     - Discard rows without message content
     - Discard rows with less 20 than caracters
     - Removing left and right spaces in the message content
    
    Args:
        messages_raw (dataframe): Dataframe containing the loaded messages
        
    Returns:
        A dataframe with the clean message
    '''

    condition_empty_messages = messages_raw['message']=='#NAME?'
    messages_raw['message_stripped'] = messages_raw['message'].str.strip()
    condition_tiny_messages = messages_raw['message_stripped'].str.len() < 20

    columns_to_get = ['message_stripped', 'genre']
    messages_clean_df = messages_raw[(~condition_empty_messages)&(~condition_tiny_messages)][columns_to_get]
    messages_clean_df.rename(columns={'message_stripped':'text'}, inplace=True)

    cond_duplicated_rows = messages_clean_df['text'].duplicated()
    messages_unique_df = messages_clean_df[~cond_duplicated_rows]
    
    return messages_unique_df


def save_data(df, database_filename):
    '''
    Function to save the dataframe into a local database file.
    
    Args:
        df (dataframe): clean messages and categories
        database_filename (str): name of the output database file including the path
    
    Returns:
        None
    '''

    print('\nSaving data...\n    DATABASE: {}'.format(database_filename))
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disasters_messages', engine, index=False)


def main():
    '''
    Function to load, clean, merge and save the disasters messages and categories to a database.
    
    System arguments:
        messages_filepath (str): name of the messages file including the path
        categories_filepath (str): name of the categories file including the path
        database_filepath (str): name of the output database file including the path

    Returns:
        None
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        df = load_data(messages_filepath, categories_filepath)

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