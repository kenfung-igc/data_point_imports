import pandas as pd
import numpy as np
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
POINT_TYPES = ['BI Mapper', 'BO Mapper', 'AV Mapper', 'MSV Mapper', 'AO Mapper', 'AI Mapper', 'BV Mapper', 'JCI Family BACnet Device']
ALARM_TYPES = ['Analog Alarm', 'Multistate Alarm']


def split(vec):
    return tf.strings.split(tf.strings.split(vec, sep='.'), sep='/')


def standardize(name):
    return re.sub('(?<!CO)(\d+)', '#', re.sub('DDC-T[-\w]+\.', '', re.sub('FC\d\.', '', name.upper())))


def point_type(point):
    return re.sub('(?<!CO)(\d+)', '#', point.bms_name.upper())


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


def map_attributes(points):
    points['asset_seq'] = points.apply(asset_seq, axis=1)
    points['asset_group'] = points.apply(asset_group, axis=1)
    points['asset_code'] = points.apply(asset_code, axis=1)
    points['point_type'] = points.apply(point_type, axis=1)
    points['class'] = points.apply(lambda x: 'Alarm' if re.search('alarm', x.bms_object_type.lower()) else 'Point', axis=1)
    points = points[~points['asset_group'].isna()]
    mapping = generate_mapping(points)
    points['point_description'] = points.apply(lambda x: mapping.loc[x.asset_group, x.point_type]['bms_description'][0], axis=1)
    points['unit'] = points.apply(lambda x: mapping.loc[x.asset_group, x.point_type]['bms_units'][0].lstrip('unitEnumSet.'), axis=1)
    return points


def asset_type(point):
    if hasattr(point, 'asset_type') and point.asset_type:
        return point.asset_type
    ref = point['bms_item_reference']
    name = point['bms_name']
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
    alternate = point.bms_name
    if alternate and current_suffix != alternate:
        return alternate.upper()
    else:
        return ''


def generate_mapping(points):
    points = points[~points['asset_group'].isna()]
    mapping = points[['asset_group', 'point_type', 'bms_description', 'bms_units']].drop_duplicates()
    mapping.set_index(['asset_group', 'point_type'], inplace=True)
    return mapping


def load_points(csv_filename, train=False, feel_lucky=False, include_alarms=False):
    object_types = POINT_TYPES
    if include_alarms:
        object_types += ALARM_TYPES
    points = pd.read_csv(csv_filename, index_col=False)
    points.fillna('', inplace=True)
    points.bms_units.replace(0, '', inplace=True)
    if 'bms_object_type' in points.columns:
        points = points[points.apply(lambda x: x.bms_object_type in object_types, axis=1)]
    points['tower'] = points['bms_item_reference'].apply(tower)
    points['floor'] = points['bms_item_reference'].apply(floor)
    points['zone'] = points['bms_item_reference'].apply(zone)
    points['reference'] = (points['bms_item_reference'].apply(lambda x: x.split('/')[-1])).str.upper()
    points['suffix'] = points.apply(suffix, axis=1)
    points['reference'] = points['reference'].str.cat(points['suffix'], sep='.').str.rstrip('.')
    points['point_type'] = points.apply(point_type, axis=1)

    if feel_lucky:
        points['asset_type'] = points.apply(asset_type, axis=1)
    if train:
        points = points[~points['asset_type'].isna()]
        print(f'\nFitting model on {len(points)} points from {csv_filename}')
    else:
        print(f'\nPredicting {len(points)} points from {csv_filename}')

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
    points.sort_values(by=['class', 'asset_group', 'asset_code', 'point_type'], inplace=True)
    points[['asset_group', 'asset_code', 'bms_item_reference', 'bms_name', 'point_type', 'point_description', 'unit', 'class']].to_csv(f'point_asset_code_{timestamp}.csv')
    print(f'\nExported asset code mappings to point_asset_code_{timestamp}.csv')
    point_types = points[['asset_group', 'point_type', 'point_description', 'unit', 'class']].drop_duplicates()
    point_types = point_types[~point_types.asset_group.isna()]
    point_types.to_csv(f'point_asset_group_{timestamp}.csv')
    print(f'\nExported asset group mappings to point_asset_group_{timestamp}.csv')


if __name__ == '__main__':
    known_points = load_points('points_manually_mapped.csv', train=True, feel_lucky=True)
    clf = train_asset_type_clf(known_points)
    known_points = load_points('points.csv', train=True, feel_lucky=True)
    clf = train_asset_type_clf(known_points, clf)
    points = load_points('points.csv', train=False, include_alarms=True)
    predict_asset_type(clf, points)
    show_feature_importances(clf, show_count=30)
    points = map_attributes(points)
    export_to_csv(points)
    exit(0)
