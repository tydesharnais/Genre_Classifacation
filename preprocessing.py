import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from prepare import set_index, prepare_df, load_Data

################################################## Feature Engineering ###################################################

def create_features(df):

    # Feature Engineering
    # converting track length from ms to seconds as new column
    df['duration_seconds'] = df.duration_ms / 1_000
    # converting track length from seconds to minutes as new column
    df['duration_minutes'] = df.duration_seconds / 60
    # creating boolean if track has a featured artist
    df['has_feat'] = df.song.str.contains('feat' or 'ft').astype('int')

    # Lowercasing String
    # using string function to convert all characters to lowercase
    df['artist'] = df.artist.str.lower()
    
    df['song'] = df.song.str.lower()

    return df

#################################################### Split the Data #####################################################

def split_df(df):

    '''
    Splits dataframe into train, validate, and test - 70%, 20%, 10% respectively.
    Prints out the percentage shape and row/column shape of the split dataframes.
    Returns train, validate, test.
    '''

    # Import to use split function, can only split two at a time
    from sklearn.model_selection import train_test_split

    # First, split into train + validate together and test by itself
    # Test will be %10 of the data, train + validate is %70 for now
    # Set random_state so we can reproduce the same 'random' data
    train_validate, test = train_test_split(df, test_size = .10, random_state = 666)

    # Second, split train + validate into their seperate dataframes
    # Train will be %70 of the data, Validate will be %20 of the data
    # Set random_state so we can reproduce the same 'random' data
    train, validate = train_test_split(train_validate, test_size = .22, random_state = 666)

    # These two print functions allow us to ensure the date is properly split
    # Will print the shape of each variable when running the function
    print("train shape: ", train.shape, ", validate shape: ", validate.shape, ", test shape: ", test.shape)

    # Will print the shape of each variable as a percentage of the total data set
    # Variable to hold the sum of all rows (total observations in the data)
    total = df.count()[0]
    
    #calculating percentages of the split df to the original df
    train_percent = round(((train.shape[0])/total),2) * 100
    validate_percent = round(((validate.shape[0])/total),2) * 100
    test_percent = round(((test.shape[0])/total),2) * 100
    
    print("\ntrain percent: ", train_percent, ", validate percent: ", validate_percent, 
            ", test percent: ", test_percent)

    return train, validate, test

def spotify_split(df, target : str):
    '''
    This function takes in a dataframe and the string name of the target variable
    and splits it into test (15%), validate (15%), and train (70%). 
    It also splits test, validate, and train into X and y dataframes.
    Returns X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test.
    '''
    # first, since the target is a continuous variable and not a categorical one,
    # in order to use stratification, we need to turn it into a categorical variable with binning.
    bin_labels_5 = ['Low', 'Moderately Low', 'Moderate', 'Moderately High', 'High']
    column_Name = f'{target[0:3]}_strat_bin'
    df[column_Name] = pd.qcut(df[target], q=5, precision=0, labels=bin_labels_5)

    # split df into test (15%) and train_validate (85%)
    train_validate, test = train_test_split(df, test_size=.15, stratify=df[column_Name], random_state=666)

    # drop column used for stratification
    train_validate = train_validate.drop(columns=[column_Name])
    test = test.drop(columns=[column_Name])

    # split train_validate off into train (82.35% of 85% = 70%) and validate (17.65% of 85% = %15)
    train, validate = train_test_split(train_validate, test_size=.1765, random_state=666)

    # split train into X & y
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X & y
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X & y
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    print('Shape of train:', X_train.shape, '| Shape of validate:', X_validate.shape, '| Shape of test:', X_test.shape)
    print('Percent train:', round(((train.shape[0])/df.count()[0]),2) * 100, '       | Percent validate:', round(((validate.shape[0])/df.count()[0]),2) * 100, '      | Percent test:', round(((test.shape[0])/df.count()[0]),2) * 100)

    return X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test

def scale_data(train, validate, test, predict, scaler):

    '''
    Scales a df based on scaler chosen: 'MinMax', 'Standard', or 'Robust'. 
    Needs three dfs: train, validate, and test. Fits the scaler object to train 
    only, transforms on all 3. Returns the three dfs scaled.
    'predict' is the target variable name.
    '''
    
    import sklearn.preprocessing
    
    # removing predictive feature
    X_train = train.drop(predict, axis=1)
    X_validate = validate.drop(predict, axis=1)
    X_test = test.drop(predict, axis=1)
    
    if scaler == 'MinMax':

        # create scaler object for MinMax Scaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        
    elif scaler == 'Standard':
        
        # create scaler object for Standard Scaler
        scaler = sklearn.preprocessing.StandardScaler()
        
    elif scaler == 'Robust':
        
        # create scaler object for Robust Scaler
        scaler = sklearn.preprocessing.StandardScaler()
        

    scaler.fit(X_train)

    # transforming all three dfs with the scaler object
    # this turns it into an array
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # converting scaled array back to df
    # first by converting to a df, it will not have the original index and column names
    X_train_scaled = pd.DataFrame(X_train_scaled)
    X_validate_scaled = pd.DataFrame(X_validate_scaled)
    X_test_scaled = pd.DataFrame(X_test_scaled)
        
    # setting index to original dfs
    X_train_scaled.index = X_train.index
    X_validate_scaled.index = X_validate.index
    X_test_scaled.index = X_test.index
        
    # renaming columns to original dfs
    X_train_scaled.columns = X_train.columns
    X_validate_scaled.columns = X_validate.columns
    X_test_scaled.columns = X_test.columns

    return X_train_scaled, X_validate_scaled, X_test_scaled


def modeling_prep():
    '''
    This function prepares the data for modeling
    '''
    # all local csv data compiled into a dataframe
    #df = load_Data()
    # adds new features, handles nulls, fixes data types, 
    # set the index to track_id, and fixes the tempo feature
    df = create_features(df)
    df = prepare_df(df)

    # drop any columns that won't contribute to modeling

    df = df.drop(columns=['song', 'artist', 
                        'year', 'duration_minutes', 
                        'duration_seconds', 'timeframe', 'explicit'])
    return df

def select_kbest(X, y, n):
    '''
    This function creates a list of n features which the
    select k best algorithm determines will perform best
    on a model
    '''
    from sklearn.feature_selection import SelectKBest, f_regression
    # create the skb object and fit to the data
    f_selector = SelectKBest(f_regression, k=n).fit(X, y)
    # create an array
    X_reduced = f_selector.transform(X)
    # get the feature support boolean mask list 
    f_support = f_selector.get_support()
    # reduce the dataframe to just those features and 
    # create list of features
    f_feature = X.iloc[:,f_support].columns.tolist()
    return f_feature

def rfe(X, y, n):

    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    # create the lr model object
    lm = LinearRegression()
    # create the rfe object
    rfe = RFE(lm, n)
    # fit and transform to the data
    X_rfe = rfe.fit_transform(X, y)
    # get the feature support boolean mask list
    mask = rfe.support_
    # reduce the dataframe to just those features
    X_reduced_scaled_rfe = X.iloc[:,mask]
    # create list of features
    f_feature = X_reduced_scaled_rfe.columns.tolist()
    return f_feature

