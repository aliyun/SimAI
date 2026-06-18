import os

def post_register_types(root_module):
    enabled_features = os.environ['NS3_ENABLED_FEATURES'].split(',')

    if 'TapBridge' not in enabled_features:
        for clsname in ['TapBridge', 'TapBridgeHelper', 'TapBridgeFdReader']:
            root_module.classes.remove(root_module['ns3::%s' % clsname])
        root_module.enums.remove(root_module['ns3::TapBridge::Mode'])
