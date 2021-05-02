
def config(name: str):
    import sys
    sys.path.append('config')

    offsets = name.split('.')
    config_dict = __import__(offsets[0]).config

    for offset in offsets[1:]:
        config_dict = config_dict[offset]

    return config_dict


__all__ = ['config']
