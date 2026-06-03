import os

def post_register_types(root_module):
    enabled_features = os.environ['NS3_ENABLED_FEATURES'].split(',')

    # If no sqlite support, disable bindings for those (optional) features
    if 'SQLiteStats' not in enabled_features:
        for clsname in ['SimpleRefCount< ns3::SQLiteOutput, ns3::empty, ns3::DefaultDeleter<ns3::SQLiteOutput> >', 'SQLiteOutput', 'SqliteDataOutput', 'DefaultDeleter< ns3::SQLiteOutput >']: 
            try:
                root_module.classes.remove(root_module['ns3::%s' % clsname])
            except:
                pass
