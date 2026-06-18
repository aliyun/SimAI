import os

def post_register_types(root_module):
    enabled_features = os.environ['NS3_ENABLED_FEATURES'].split(',')
    # if GtkConfigStore support is disabled, disable the class wrapper
    if 'GtkConfigStore' not in enabled_features:
        try:
            root_module.classes.remove(root_module['ns3::GtkConfigStore'])
        except KeyError:
            pass
