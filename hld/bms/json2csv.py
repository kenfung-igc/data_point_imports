import json
import csv


def find_item(data, item):
    if isinstance(data, dict):
        target_key = next(iter(item))
        for key, value in data.items():
            if key == target_key and value == item[target_key]:
                yield data
            else:
                yield from find_item(value, item)
    elif isinstance(data, list):
        for value in data:
            yield from find_item(value, item)


def device(node):
    return next(find_item(devices, {'guid': node['parent_guid']}))


def space(device):
    return next(find_item(spaces, {'guid': device['parent_guid']}))


def csv_filename(header):
    if 'point_type' in header:
        return 'data_points.csv'
    elif 'device_type' in header:
        return 'devices.csv'
    elif 'space_type' in header:
        return 'spaces.csv'
    else:
        raise Exception(f'Failed to parse node: {nodes[0]}')


def get_header(nodes):
    try:
        return nodes[0].keys()
    except AttributeError:
        return nodes[1].keys()


def write_csv(nodes):
    header = get_header(nodes)
    csv_obj = open(csv_filename(header), 'w')
    csv_writer = csv.writer(csv_obj)
    csv_writer.writerow(header)
    for node in nodes:
        csv_writer.writerow(node.values())
    csv_obj.close()


if __name__ == '__main__':
    f = open('data_points.json')
    nodes = json.load(f)['data']['nodes']
    spaces = [n for n in nodes if n['node_type'] == 1]
    devices = [n for n in nodes if n['node_type'] == 2]
    devices = [d | {'space_name': space(d)['name']} for d in devices]
    points = [n for n in nodes if n['node_type'] == 3]
    points = [p | {'device_name': device(p)['name'], 'space_name': space(device(p))['name']} for p in points]
    write_csv(spaces)
    write_csv(devices)
    write_csv(points)



