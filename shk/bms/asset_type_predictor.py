from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

class AssetTypePredictor:

    def __init__(self, points):
        self.points = points

    def vectorize(self, references_raw, vocabulary=[]):
        token_pattern = "(?u)\\b[\\w#-]+\\b"
        references = references_raw.apply(standardize)
        if len(vocabulary) == 0:
            vectorizer = TfidfVectorizer(token_pattern=token_pattern, lowercase=False)
            vocabulary = vectorizer.fit(references).get_feature_names_out()
            vocabulary = [v for v in vocabulary if not any([re.search(t, v) for t in SIMPLE_ASSET_TYPES])]
            vocabulary = superlist(vocabulary, SIMPLE_ASSET_TYPES)
        vectorizer = TfidfVectorizer(token_pattern=token_pattern, vocabulary=vocabulary, lowercase=False)
        refs = vectorizer.fit_transform(references)
        refs = pd.DataFrame.sparse.from_spmatrix(refs)
        refs.columns = vectorizer.get_feature_names_out()
        return refs

    def train_asset_type_clf(self, points, clf=None):
        X = self.vectorize(points['reference'])
        y = points['asset_type']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        test_df = y_test.rename('actual_asset_type').to_frame()
        if not clf:
            clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf.fit(X_train, y_train)
        test_df['predicted_asset_type'] = clf.predict(X_test)
        test_df = test_df.merge(points, left_index=True, right_index=True).merge(y_test, left_index=True,
                                                                                 right_index=True)
        print(
            f"Predicted asset type with {round(accuracy_score(test_df['actual_asset_type'], test_df['predicted_asset_type']), 3)} accuracy")
        test_df['predicted_asset_type_raw'] = test_df['predicted_asset_type']
        test_df['matched'] = test_df.apply(lambda x: bool(re.search(f"\.{x['predicted_asset_type']}", x['reference'])),
                                           axis=1)
        test_df.loc[~test_df['matched'], 'predicted_asset_type'] = None
        missed = test_df[test_df['predicted_asset_type'] != test_df['actual_asset_type']]
        print(
            f"Failed to predict asset type for {len(missed)} points:\n {missed[['reference', 'predicted_asset_type_raw']]}")
        return clf

    def show_feature_importances(self, clf, show_count=10):
        importances = clf.feature_importances_
        reverse = show_count < 0
        show_count = min(abs(show_count), len(importances))
        print(f"\n{'Bottom' if reverse else 'Top'} {show_count} features (out of {len(importances)}):")
        indices = importances.argsort()[:show_count] if reverse else (-importances).argsort()[:show_count]
        for i in indices:
            print(f'{clf.feature_names_in_[i]} ({round(importances[i], 4)})')

    def predict_asset_type(self, clf, points):
        X = self.vectorize(points['reference'], vocabulary=clf.feature_names_in_)
        points['predicted_asset_type'] = clf.predict(X)
        points['matched'] = points.apply(lambda x: bool(re.search(x['predicted_asset_type'], x['reference'])), axis=1)
        # points['predicted_asset_type_raw'] = points['predicted_asset_type']
        points.loc[points['matched'], 'asset_type'] = points.loc[points['matched'], 'predicted_asset_type']
        print(f"\nPredicted asset type for {len(points[~points['asset_type'].isna()])} additional points")
        return points


