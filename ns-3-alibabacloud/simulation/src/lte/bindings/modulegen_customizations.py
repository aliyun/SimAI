import os

def post_register_types(root_module):
    enabled_features = os.environ['NS3_ENABLED_FEATURES'].split(',')

    if 'EmuFdNetDevice' not in enabled_features:
        if 'ns3::EmuEpcHelper'in root_module:
            root_module.classes.remove(root_module['ns3::EmuEpcHelper'])

