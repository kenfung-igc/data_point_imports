import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from datetime import datetime
import re
pd.options.mode.chained_assignment = None

ASSET_TYPE_FILE = 'asset_groups.csv'
ASSET_GROUP = pd.read_csv(ASSET_TYPE_FILE, index_col=False).fillna('')
ASSET_GROUP.set_index('asset_type', inplace=True)
ASSET_TYPES = ASSET_GROUP.index
SIMPLE_ASSET_TYPES = ['CHWSH', 'CHDSYS', 'WCC', 'CDWBYPASS', 'TWP', 'CLWT']
vocab_size = 10000


def split(vec):
    return tf.strings.split(tf.strings.split(vec, sep='.'), sep='/')


def standardize(name):
    return re.sub('(?<!CO)(\d+)', '#', re.sub('DDC-T[-\w]+\.', '', re.sub('FC\d\.', '', name.upper())))


def point_type(point):
    return re.sub('(?<!CO)(\d+)', '#', point.BMS_Name.upper())


def point_describe(point, field):
    global point_description
    try:
        return point_description.loc[point.asset_group, point.point_type][field][0]
    except KeyError:
        return None


def asset_seq(point):
    try:
        return re.search(f'{point.asset_type}-?(?:T\d)?-?(?:L\d\d|RF)?-?(?:Z\d)?-?(?:\w\w\w?)?-?(\d+)', point.reference).group(1)
    except (AttributeError, IndexError, TypeError):
        return None


def asset_group(point):
    if point.asset_type:
        return ASSET_GROUP.loc[point.asset_type].values[0] or point.asset_type
    else:
        return None


def asset_code(point):
    if point.asset_type:
        tower = f'-{point.tower}' if point.tower else ''
        floor = f'-{point.floor}' if point.floor else ''
        asset = f'-{asset_group(point)}' if point.asset_group else ''
        zone = f'-{point.zone}' if point.zone else ''
        seq = f'-{point.asset_seq}' if point.asset_seq else ''
        return f'98HMS{tower}{floor}{asset}{zone}{seq}'
    return None


def set_asset_codes(points):
    points['asset_seq'] = points.apply(asset_seq, axis=1)
    points['asset_group'] = points.apply(asset_group, axis=1)
    points['asset_code'] = points.apply(asset_code, axis=1)
    points['point_type'] = points.apply(point_type, axis=1)


def asset_type(point):
    if hasattr(point, 'asset_type') and point.asset_type:
        return point.asset_type
    ref = point['BMS_Item Reference']
    name = point['BMS_Name']
    for t in ASSET_TYPES:
        if re.search(f'^{t}', name) or re.search(f'\.{t}', ref.split('.')[-1]):
            return t
    if re.search('FC1.', ref):
        type_ref = ref.split('FC1.')[1].split('-')[0]
        if type_ref != 'DDC':
            return re.sub(r'\d$', '', type_ref)
    return None


def floor(ref):
    try:
        match = re.search('(L\d\dM?)', ref) or re.search('-(RF)-', ref) or re.search('-(UR)-', ref)
        return match.group(1)
    except AttributeError:
        return None


def tower(ref):
    try:
        return re.search('T(\d)', ref).group(1)
    except AttributeError:
        return None


def zone(ref):
    try:
        return re.search('(Z\d)', ref).group(1)
    except AttributeError:
        return None


def suffix(point):
    current_suffix = point.reference.split('.')[-1]
    alternate = point.BMS_Name.upper()
    if current_suffix != alternate:
        return alternate.upper()
    else:
        return ''


def load_points(csv_filename, train=False, feel_lucky=False):
    points = pd.read_csv(csv_filename, index_col=False)
    points['tower'] = points['BMS_Item Reference'].apply(tower)
    points['floor'] = points['BMS_Item Reference'].apply(floor)
    points['zone'] = points['BMS_Item Reference'].apply(zone)
    points['reference'] = (points['BMS_Item Reference'].apply(lambda x: x.split('/')[-1])).str.upper()
    points['suffix'] = points.apply(suffix, axis=1)
    points['reference'] = points['reference'].str.cat(points['suffix'], sep='.').str.rstrip('.')
    points['point_type'] = points.apply(point_type, axis=1)

    if feel_lucky:
        points['asset_type'] = points.apply(asset_type, axis=1)
    if train:
        points = points[~points['asset_type'].isna()]
        print(f'\nFitting model on {len(points)} points')
    else:
        print(f'\nPredicting {len(points)} points')

    return points


def superlist(*args):
    superset = set()
    for k in args:
        superset = superset.union(set(k))
    return list(superset)


def vectorize(references_raw, vocabulary=[]):
    token_pattern = "(?u)\\b[\\w#-]+\\b"
    references = references_raw.apply(standardize)
    if len(vocabulary) == 0:
        vectorizer = TfidfVectorizer(token_pattern=token_pattern, lowercase=False)
        vocabulary = vectorizer.fit(references).get_feature_names_out()
        vocabulary = [v for v in vocabulary if not any([re.search(t,v) for t in SIMPLE_ASSET_TYPES])]
        vocabulary = superlist(vocabulary, SIMPLE_ASSET_TYPES)
    vectorizer = TfidfVectorizer(token_pattern=token_pattern, vocabulary=vocabulary, lowercase=False)
    refs = vectorizer.fit_transform(references)
    refs = pd.DataFrame.sparse.from_spmatrix(refs)
    refs.columns = vectorizer.get_feature_names_out()
    return refs


def train_asset_type_clf(points, clf=None):
    X = vectorize(points['reference'])
    y = points['asset_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    test_df = y_test.rename('actual_asset_type').to_frame()
    if not clf:
        clf = DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=42)
    clf.fit(X_train, y_train)
    test_df['predicted_asset_type'] = clf.predict(X_test)
    test_df = test_df.merge(points, left_index=True, right_index=True).merge(y_test, left_index=True, right_index=True)
    print(f"Predicted asset type with {round(precision_score(test_df['actual_asset_type'], test_df['predicted_asset_type'], average='micro'), 3)} precision")
    test_df['predicted_asset_type_raw'] = test_df['predicted_asset_type']
    test_df['matched'] = test_df.apply(lambda x: bool(re.search(x['predicted_asset_type'], x['reference'])), axis=1)
    test_df.loc[~test_df['matched'], 'predicted_asset_type'] = None
    missed = test_df[test_df['predicted_asset_type'] != test_df['actual_asset_type']]
    print(f"Failed to predict asset type for {len(missed)} points:\n {missed[['reference', 'predicted_asset_type_raw']]}")
    return clf


def show_feature_importances(clf, show_count=10):
    importances = clf.feature_importances_
    reverse = show_count < 0
    show_count = min(abs(show_count), len(importances))
    print(f"\n{'Bottom' if reverse else 'Top'} {show_count} features (out of {len(importances)}):")
    indices = importances.argsort()[:show_count] if reverse else (-importances).argsort()[:show_count]
    for i in indices:
        print(f'{clf.feature_names_in_[i]} ({round(importances[i], 4)})')


def predict_asset_type(clf, points):
    X = vectorize(points['reference'], vocabulary=clf.feature_names_in_)
    points['predicted_asset_type'] = clf.predict(X)
    points['matched'] = points.apply(lambda x: bool(re.search(x['predicted_asset_type'], x['reference'])), axis=1)
    points['predicted_asset_type_raw'] = points['predicted_asset_type']
    points.loc[~points['matched'], 'predicted_asset_type'] = None
    points['asset_type'] = points['predicted_asset_type']
    print(f"\nFailed to predict asset type for {len(points[~points['matched']].index)} points")
    return clf


def export_to_csv(points):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    points.sort_values(by=['asset_group', 'asset_code', 'point_type'])
    points[['asset_group', 'asset_code', 'BMS_Item Reference', 'BMS_Name', 'point_type']].to_csv(f'point_asset_code_{timestamp}.csv')
    print(f'\nExported asset code mappings to point_asset_code_{timestamp}.csv')
    point_types = points[['asset_group', 'point_type']].drop_duplicates().sort_values(by=['asset_group', 'point_type'])
    point_types = point_types[~point_types.asset_group.isna()]
    point_types['point_description'] = point_types.apply(point_describe, field='point_description', axis=1)
    point_types['unit'] = point_types.apply(point_describe, field='BMS_Units', axis=1)
    point_types.to_csv(f'point_asset_group_{timestamp}.csv')
    print(f'\nExported asset group mappings to point_asset_group_{timestamp}.csv')


if __name__ == '__main__':
    known_points = load_points('points_manually_mapped.csv', train=True, feel_lucky=True)
    point_description = known_points[['asset_group', 'point_type', 'point_description', 'BMS_Units']].drop_duplicates()
    point_description.set_index(['asset_group', 'point_type'], inplace=True)
    clf = train_asset_type_clf(known_points)
    known_points = load_points('points.csv', train=True, feel_lucky=True)
    clf = train_asset_type_clf(known_points, clf)
    points = load_points('points.csv', train=False)
    predict_asset_type(clf, points)
    show_feature_importances(clf, show_count=30)
    set_asset_codes(points)
    export_to_csv(points.sort_values(by=['asset_type', 'tower', 'floor', 'zone', 'asset_seq']))
    exit(0)
