import pandas as pd
from ml_pipeline.utils import count_fn, conv_values, flatten_list, feat_imp

def score_fn(data, count_path, features, list_cols, seed_ids, neighbors, label):
    """
    Calculate a score for each user in the extended set based on feature importance.

    :param data: DataFrame containing user features
    :param count_path: Path to the feature count file
    :param features: Features in the dataset
    :param list_cols: Columns that can have multiple values per user
    :param seed_ids: Customer IDs part of the seed set
    :param neighbors: Customer IDs extracted from the LSH graph
    :param label: Label column indicating whether the user clicked the ad or not
    :return: DataFrame containing user features along with a score for each user
    """

    # Read the feature count file
    count_df = pd.read_csv(count_path)
    # Select user records from the seed set
    seed_df = data[data["id"].isin(seed_ids)]
    # Calculate feature count in the seed set
    seed_count = count_fn(seed_df, features, list_cols)
    seed_count.rename({"prob": "s_prob"}, axis=1, inplace=True)
    # Merge seed set feature count with global feature count
    seed_count = seed_count.merge(count_df, on=["feature", "value"], how="left")
    # Calculate feature importance
    seed_count["imp"] = seed_count.apply(lambda x: feat_imp(x["s_prob"], x["prob"]), axis=1)
    # Create a new column with feature values as string prefixed by column name
    seed_count["feat"] = seed_count["feature"] + "_" + seed_count["value"].astype(str)
    seed_count = seed_count[["feat", "imp"]]
    df = data.drop(label, axis=1)
    # Filter records in the neighbors
    df = df[df["id"].isin(neighbors)]
    # Convert the features to strings prefixed by column value
    for f in features:
        list_c = f in list_cols
        df[f] = df[f].apply(lambda x: conv_values(x, f, list_c))
    # Collect all features for a user as a list
    df["feat"] = df.apply(lambda x: list(x[1:].values), axis=1)
    df["feat"] = df["feat"].apply(flatten_list)
    df = df[["id", "feat"]]
    # Explode the new feature column
    df = df.explode("feat").reset_index(drop=True)
    # Merge with the seed count dataframe
    df = df.merge(seed_count, on="feat", how="left")
    df = df[["id", "imp"]]
    # Calculate the score for every user
    df = df.groupby("id")["imp"].sum().reset_index()
    df.columns = ["id", "score"]
    # Merge with the original user dataset
    data = data.merge(df, on="id", how="left")
    return data

def get_extn(data, seed_ids, label, extn_path, x=2):
    """
    Retrieve a set of users from the neighbor set based on their score and save them to a file.

    :param data: DataFrame containing user data along with the scores
    :param seed_ids: List of customer IDs that are in the seed set
    :param label: Label column indicating whether the user clicked the ad
    :param extn_path: Path to store the extended user set
    :param x: Scale of extension needed
    :return: Click rate of the extended user set
    """
    # Drop users who don't have a score (those are not neighbors)
    data = data.dropna(subset=["score"])
    # Sort the users by score in descending order
    data = data.sort_values(by="score", ascending=False)
    # Select the top users
    extn = data.iloc[:x * len(seed_ids), :][["id"]]
    # Write the extended user IDs to a file
    extn.to_csv(extn_path, index=False)
    # Calculate the click rate of the extended user set
    extn_click_rate = data.iloc[:x * len(seed_ids), :][label].mean()
    extn_click_rate = round(extn_click_rate * 100, 2)
    return extn_click_rate
