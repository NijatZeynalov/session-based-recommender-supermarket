import os
import pandas as pd
import numpy as np
from numpy.random import default_rng

from utils.file_handling import pickle_load, pickle_save, absolute_filename, create_path

rng = default_rng(123)

ECOMM_PATH = "data/supermarket/"
ECOMM_FILENAME = "supermarket_sales.csv"

def load_ecomm(filename=None):
    """
    Checks to see if the processed Online Retail ecommerce session sequence file exists
        If True: loads and returns the session sequences
        If False: creates and returns the session sequences constructed from the original data file
    """
    original_filename = absolute_filename(ECOMM_PATH, ECOMM_FILENAME)
    if filename is None:
        processed_filename = original_filename.replace(".csv", "_sessions.pkl")
        if os.path.exists(processed_filename):
            return pickle_load(processed_filename)
    else:
        if os.path.exists(absolute_filename(filename)):
            return pickle_load(absolute_filename(filename))

    df = load_original_ecomm(original_filename)
    session_sequences = preprocess_ecomm(df)
    return session_sequences


def load_original_ecomm(pathname=ECOMM_PATH):
    df = pd.read_csv(
        absolute_filename(ECOMM_PATH, ECOMM_FILENAME),
        encoding="ISO-8859-1",
        parse_dates=["InvoiceDate"],
    )
    return df


def preprocess_ecomm(df, min_session_count=3):
    df.dropna(inplace=True)
    item_counts = df.groupby(["CustomerID"]).count()["StockCode"]
    df = df[
        df["CustomerID"].isin(item_counts[item_counts >= min_session_count].index)
    ].reset_index(drop=True)

    # TODO: track preprocessed version by appending the filename with min_session_count
    filename = absolute_filename(
        ECOMM_PATH, ECOMM_FILENAME.replace(".csv", "_sessions.pkl")
    )
    sessions = construct_session_sequences(
        df, "CustomerID", "StockCode", save_filename=filename
    )
    return sessions

def construct_session_sequences(df, sessionID, itemID, save_filename):
    """
    Given a dataset in pandas df format, construct a list of lists where each sublist
    represents the interactions relevant to a specific session, for each sessionID.
    These sublists are composed of a series of itemIDs (str) and are the core training
    data used in the Word2Vec algorithm.

    This is performed by first grouping over the SessionID column, then casting to list
    each group's series of values in the ItemID column.
    """
    grp_by_session = df.groupby([sessionID])

    session_sequences = []
    for name, group in grp_by_session:
        session_sequences.append(list(group[itemID].values))

    filename = absolute_filename(save_filename)
    create_path(filename)
    pickle_save(session_sequences, filename=save_filename)
    return session_sequences


def train_test_split(session_sequences, test_size: int = 10000, rng=rng):
    """
    Next Event Prediction (NEP) does not necessarily follow the traditional train/test split.

    Instead training is perform on the first n-1 items in a session sequence of n items.
    The test set is constructed of (n-1, n) "query" pairs where the n-1 item is used to generate
    recommendation predictions and it is checked whether the nth item is included in those recommendations.

   """
    train = [sess[:-1] for sess in session_sequences]

    if test_size > len(train):
        print(
            f"Test set cannot be larger than train set. Train set contains {len(train)} sessions."
        )
        return

    ### Construct test and validation sets
    # sub-sample 10k sessions, and use (n-1 th, n th) pairs of items from session_squences to form the
    # disjoint validaton and test sets
    test_validation = [sess[-2:] for sess in session_sequences]
    # TODO: set numpy random seed! NM: added it at the top
    index = rng.choice(range(len(test_validation)), test_size * 2, replace=False)
    test = np.array(test_validation)[index[:test_size]].tolist()
    validation = np.array(test_validation)[index[test_size:]].tolist()

    return train, test, validation
if __name__ == "__main__":
    # load data
    sessions = load_ecomm()

    print(len(sessions))

    train, test, valid = train_test_split(sessions, test_size=1000)
    print("validation set:", len(valid))
    print("train set:", len(train))
    print("test set", len(test))
# """

