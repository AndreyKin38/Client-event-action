from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class BinaryClassificator(BaseEstimator, TransformerMixin):

    def __init__(self, input_data: dict | list):
        super(BinaryClassificator, self).__init__()
        self.input_data = input_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for k, v in self.input_data.items():
            X[k] = X[k].apply(lambda x: 1 if x in v else 0)

        return X


class CategoryClassificator(BinaryClassificator):

    @staticmethod
    def splitter(x, category_list):
        if x:
            x = x.split()[0]
            if x in category_list:
                return x
            else:
                return 'other'
        return 'other'

    def transform(self, X):
        for k, v in self.input_data.items():
            X[k] = X[k].apply(lambda x: self.splitter(x, v))
        return X


class ColumnsToDrop(BinaryClassificator):

    def transform(self, X):
        return X.drop(self.input_data, axis=1)


class Encoder(BinaryClassificator):

    def __init__(self, input_data: dict | list):
        super().__init__(input_data)
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(X[self.input_data.keys()])

        return self

    def transform(self, X):
        feature_names = self.encoder.get_feature_names_out()
        X[feature_names] = self.encoder.transform(X[self.input_data.keys()])

        return X.drop(self.input_data.keys(), axis=1)










