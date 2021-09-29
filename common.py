
import yaml


def load_settings(filename='settings.yaml'):
    cfg = {}
    with open(filename, 'rt') as f:
        cfg = yaml.safe_load(f)

    return cfg


