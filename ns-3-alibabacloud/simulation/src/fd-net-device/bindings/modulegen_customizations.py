import os

def post_register_types(root_module):
    enabled_features = os.environ['NS3_ENABLED_FEATURES'].split(',')

    if 'EmuFdNetDevice' not in enabled_features:
        if 'ns3::EmuFdNetDeviceHelper'in root_module:
            root_module.classes.remove(root_module['ns3::EmuFdNetDeviceHelper'])

    if 'TapFdNetDevice' not in enabled_features:
        if 'ns3::TapFdNetDeviceHelper'in root_module:
            root_module.classes.remove(root_module['ns3::TapFdNetDeviceHelper'])

    if 'FdNetDevice' not in enabled_features:
        for clsname in ['FdNetDevice', 'FdNetDeviceHelper', 'FdNetDeviceFdReader']: 
            if 'ns3::%s' % clsname in root_module:
                root_module.classes.remove(root_module['ns3::%s' % clsname])
        if 'ns3::FdNetDeviceHelper::EncapsulationMode' in root_module:
            root_module.enums.remove(root_module['ns3::FdNetDeviceHelper::EncapsulationMode'])

