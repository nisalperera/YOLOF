from yacs.config import _assert_with_logging, _valid_type, _VALID_TYPES
from detectron2.config import CfgNode


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()

def to_dict(cfg: CfgNode, key_list=[]) -> dict:
    """
    Convert a CfgNode to a dict.

    Args:
        cfg: a detectron2 CfgNode instance.

    Returns:
        a dict representation of the config.
    """
    if not isinstance(cfg, CfgNode):
        _assert_with_logging(
            _valid_type(cfg),
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg), _VALID_TYPES
            ),
        )
        return cfg
    else:
        cfg_dict = dict(cfg)
        for k, v in cfg_dict.items():
            cfg_dict[k] = to_dict(v, key_list + [k])
        return cfg_dict
