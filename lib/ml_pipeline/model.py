from datasketch import MinHash, MinHashLSHForest

class LSHGraph:
    """
    Locality-Sensitive Hashing (LSH) Graph Model
    """

    def __init__(self, df, model, features, id_col="id", n_perm=10):
        """
        Initialize the LSHGraph.

        :param df: DataFrame containing user features
        :param model: MinHashLSHForest model for LSH operations
        :param features: List of features in the dataset
        :param id_col: Column name for user IDs
        :param n_perm: Number of permutations for the LSH model
        """
        self.df = df
        self.model = model
        self.features = features
        self.id_col = id_col
        self.n_perm = n_perm

    def update_graph(self):
        """
        Update the LSH graph by adding MinHash values for each user in the DataFrame.
        """
        for i, row in self.df[self.features].iterrows():
            if i % 5000 == 0:
                print(f"Processing {i} of {self.df.shape[0]}")
            m = MinHash(num_perm=self.n_perm)
            m = self.get_hash(m, row)
            self.model.add(self.df[self.id_col][i], m)
        self.model.index()

    def extract_neighbors(self, seed, k=10):
        """
        Retrieve neighbors of seed set users from the LSH graph.

        :param seed: List of customer IDs from the seed set
        :param k: Number of neighbors to retrieve for each seed set user
        :return: List of neighbors of seed set users
        """
        neighbors = []
        seed_df = self.df[self.df[self.id_col].isin(seed)]
        for i, row in seed_df[self.features].iterrows():
            m = MinHash(num_perm=self.n_perm)
            m = self.get_hash(m, row)
            neighbors.extend(self.model.query(m, k))
        neighbors = list(set(neighbors) - set(seed))
        return neighbors

    def get_hash(self, m, row):
        """
        Encode a user's features using MinHash.

        :param m: MinHash object to update
        :param row: User's feature list
        :return: Updated MinHash object
        """
        for d in row:
            if type(d) == list:
                for e in d:
                    m.update(str(e).encode('utf-8'))
            else:
                m.update(str(d).encode('utf-8'))
        return m
