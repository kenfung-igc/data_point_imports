import pandas as pd
from datetime import datetime
import re
pd.options.mode.chained_assignment = None

ASSET_TYPE_FILE = 'asset_groups.csv'
ASSET_GROUP = pd.read_csv(ASSET_TYPE_FILE, index_col=False).fillna('')
ASSET_GROUP.set_index('asset_type', inplace=True)
ASSET_TYPES = ASSET_GROUP.index
SIMPLE_ASSET_TYPES = ['CHWSH', 'CHDSYS', 'WCC', 'CLWT', 'GEN', 'MUWP']
POINT_TYPES = ['BI Mapper', 'BO Mapper', 'AV Mapper', 'MSV Mapper', 'AO Mapper', 'AI Mapper', 'BV Mapper', 'JCI Family BACnet Device']
ALARM_TYPES = ['Analog Alarm', 'Multistate Alarm']



def standardize(name):
    return re.sub('(?<!CO)(\d+)', '#', re.sub('DDC-T[-\w]+\.', '', re.sub('FC\d\.', '', name.upper())))


def point_type(point):
    return re.sub('(?<!CO)(\d+)', '#', point.bms_name.upper())


def asset_seq(point):
    try:
        return re.findall(f'{point.asset_type}-?(?:T\d)?-?(?:L\d\d|RF)?-?(?:Z\d)?-?(\d+)', point.reference)[-1]
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
    points.loc[:, 'asset_seq'] = points.apply(asset_seq, axis=1)
    points.loc[:, 'asset_group'] = points.apply(asset_group, axis=1)
    points.loc[:, 'asset_code'] = points.apply(asset_code, axis=1)
    points.loc[:, 'point_type'] = points.apply(point_type, axis=1)
    points.loc[:, 'class'] = points.apply(lambda x: 'Alarm' if re.search('alarm', x.bms_object_type.lower()) else 'Point', axis=1)
    points_to_map = points[~points['asset_group'].isna()]
    mapping = generate_mapping(points_to_map)
    points.loc[:, 'point_description'] = points_to_map.apply(lambda x: mapping.loc[x.asset_group, x.point_type]['bms_description'][0], axis=1)
    points.loc[:, 'unit'] = points_to_map.apply(lambda x: mapping.loc[x.asset_group, x.point_type]['bms_units'][0].lstrip('unitEnumSet.'), axis=1)
    return points


def asset_type_by_ref(ref=None):
    type_by_ref = None
    if re.search('FC1.', ref):
        tokens = ref.split('FC1.')[1].split('.')
        while tokens and not type_by_ref:
            token = tokens.pop()
            type_by_ref = next((a for a in ASSET_TYPES if re.search(f'(^{a}|{a}-?\d+|T\d{a})', token)), None)
    return type_by_ref


def set_asset_type(point):
    if hasattr(point, 'asset_type') and point.asset_type:
        return point.asset_type
    ref = point.bms_item_reference
    name = point.bms_name
    type_by_name = next((a for a in ASSET_TYPES if re.search(f'^{a}', name)), None)
    type_by_ref = asset_type_by_ref(ref)
    if type_by_name and type_by_ref:
        if type_by_name == type_by_ref:
            return type_by_name
        else:
            print(f'Name {point.bms_name} indicates {type_by_name}. Ref {ref} indicates {type_by_ref}. Which to choose?')
    elif type_by_name:
        return type_by_name
    elif type_by_ref:
        return type_by_ref
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


def load_points(csv_filename, strict=False, mapping=False, include_alarms=False):
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
    points = points[points.apply(lambda x: not re.search('SPARE', x.point_type), axis=1)]

    if mapping:
        points['asset_type'] = points.apply(set_asset_type, axis=1)
    if strict:
        points = points[~points['asset_type'].isna()]
        points = points[points['asset_type'] != '']
    print(f'\nLoaded {len(points)} points from {csv_filename}')
    return points


def superlist(*args):
    superset = set()
    for k in args:
        superset = superset.union(set(k))
    return list(superset)


def reconcile(mapping, expected_mapping):
    OUTPUT = ['reference', 'bms_name', 'expected_asset_code', 'asset_code']
    mismatch = mapping.merge(expected_mapping, how='inner', on='bms_item_reference')
    mismatch.rename(columns={'asset_code_x': 'asset_code', 'asset_code_y': 'expected_asset_code', 'bms_name_x': 'bms_name', 'reference_x': 'reference'}, inplace=True)
    mismatch = mismatch[mismatch['asset_code'] != mismatch['expected_asset_code']][OUTPUT]
    if len(mismatch):
        print(f"Mappings unexpected:\n{mismatch[OUTPUT]}")


def export_to_csv(points):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    points.sort_values(by=['class', 'asset_group', 'asset_code', 'point_type'], inplace=True)
    points[['asset_group', 'asset_code', 'bms_item_reference', 'bms_name', 'point_type', 'point_description', 'bms_object_type', 'unit', 'class']].to_csv(f'point_asset_code_{timestamp}.csv')
    print(f'\nExported asset code mappings to point_asset_code_{timestamp}.csv')
    point_types = points[['asset_group', 'point_type', 'point_description', 'unit', 'class']].drop_duplicates()
    point_types = point_types[~point_types.asset_group.isna()]
    point_types.to_csv(f'point_asset_group_{timestamp}.csv')
    print(f'\nExported asset group mappings to point_asset_group_{timestamp}.csv')


if __name__ == '__main__':
    points = load_points('points.csv', mapping=True, include_alarms=True)
    map_attributes(points)
    known_points = load_points('points_manually_mapped.csv', strict=True)
    reconcile(points, known_points)
    export_to_csv(points)

    # known_points = load_points('points_manually_mapped.csv', train=True, feel_lucky=False)
    # clf = train_asset_type_clf(known_points)
    # clf = train_asset_type_clf(points[~points['asset_type'].isna()], clf)
    # points_predicted = predict_asset_type(clf, points[points['asset_type'].isna()])
    # show_feature_importances(clf, show_count=30)
    # points_predicted = map_attributes(points_predicted)
    # export_to_csv(points_predicted)
    exit(0)
