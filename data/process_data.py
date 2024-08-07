import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data from two CSV files and merge them into a single DataFrame.

    Parameters:
        messages_filepath (str): The path to the CSV file containing the messages data.
        categories_filepath (str): The path to the CSV file containing the categories data.

    Returns:
        pandas.DataFrame: The merged DataFrame containing the messages and categories data.
    """
            
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id')
    print(df.head())
    return df

def clean_data(df):
    """
    Clean the given DataFrame by merging it with the messages and categories datasets,
    creating a new DataFrame of individual category columns, renaming the columns,
    converting the values to numeric, dropping the original categories column,
    concatenating the original DataFrame with the new categories DataFrame,
    dropping duplicates, and replacing the value 2 in the 'related' column with 1.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame to be cleaned.
    
    Returns:
    - pandas.DataFrame: The cleaned DataFrame.
    """
    
    # Check if the DataFrame is not None
    if df is None:
        raise ValueError("Input DataFrame is None")
    #print(df.head())
    # Check if 'categories' column exists in the DataFrame
    #if 'categories' not in df.columns:
        #raise ValueError("'categories' column does not exist in the DataFrame")
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    #print(categories.head())
    
    # Check if the categories DataFrame is not empty
    if categories.empty:
        raise ValueError("'categories' DataFrame is empty")
    
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    
    # Check if the row is not empty
    if row.empty:
        raise ValueError("Row is empty")
    
    category_colnames = []
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
        
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors='coerce')
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_final = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df_final.drop_duplicates(inplace=True)
    
    # Check if 'related' column exists in the DataFrame
    #if 'related' in df_final.columns:
        # Replace the value 2 in the 'related' column with 1
        #Sdf_final.loc[df_final['related'] == 2, 'related'] = 1
    
    return df_final
    

def save_data(dataframe, database_filename):
    """
    Save dataframe to SQLite database.

    Args:
        dataframe (pandas.DataFrame): Dataframe to be saved.
        database_filename (str): Name of the SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    #cleaned_data = clean_data(dataframe)
    dataframe.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    """
    Main function that processes data from two CSV files and saves the cleaned data to a SQLite database.
    Args:
        None
    Returns:
        None
    Usage:
        python process_data.py messages_filepath categories_filepath database_filepath
        - messages_filepath: The path to the CSV file containing the messages data.
        - categories_filepath: The path to the CSV file containing the categories data.
        - database_filepath: The path to the SQLite database file where the cleaned data will be saved.
    Example:
        python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df_clean = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df_clean, database_filepath)
        
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