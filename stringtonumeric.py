from h2oaicore.transformer_utils import CustomTransformer
from sklearn.feature_extraction.text import CountVectorizer
import datatable as dt
import numpy as np


class StringtoNumeric(CustomTransformer):
    
    _classification = True
    _binary = False
    _multiclass = True
    _numeric_output = True
    _is_reproducible = True
    _included_model_classes = None  # List[str]
    _excluded_model_classes = None  # List[str]

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols="all", relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)
      

    def transform(self, X: dt.Frame):
        def getKmers(sequence, size=6):
            return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
        textcols = get_default_properties()
        for key in textcols.keys():
            X['words'] = X.apply(lambda x: getKmers(x[key]), axis=1)
        X_texts = list(X['words'])
        for item in range(len(X_texts)):
            X_texts[item] = ' '.join(X_texts[item])
        y_h = X.iloc[:, 0].values   
        cv = CountVectorizer(ngram_range=(4,4))
        X = cv.fit_transform(X_texts)
