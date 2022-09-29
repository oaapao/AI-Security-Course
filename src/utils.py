import json
import os


def update_json(json_path, key, value):
    """
    Add key-value to existing .json file
    :param json_path: .json file path
    :param key:
    :param value:
    :return: None
    """
    assert os.path.exists(json_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
    data[key] = value
    with open(json_path, 'w') as fp:
        json.dump(data, fp)
