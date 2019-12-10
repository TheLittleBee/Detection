from ..network import backbone, head, neck


def build_backbone(cfg):
    assert cfg['type'] in backbone.__dict__, 'backbone not support'
    md = backbone.__dict__[cfg['type']]()
    # md.freeze(cfg['freeze'])
    return md


def build_head(cfg):
    assert cfg['type'] in head.__dict__
    type = cfg.pop('type')
    return head.__dict__[type](**cfg)


def build_neck(cfg):
    assert cfg['type'] in neck.__dict__
    type = cfg.pop('type')
    return neck.__dict__[type](**cfg)
