from pybindgen import Module, FileCodeSink, param, retval, cppclass, typehandlers


import pybindgen.settings
import warnings

class ErrorHandler(pybindgen.settings.ErrorHandler):
    def handle_error(self, wrapper, exception, traceback_):
        warnings.warn("exception %r in wrapper %s" % (exception, wrapper))
        return True
pybindgen.settings.error_handler = ErrorHandler()


import sys

def module_init():
    root_module = Module('ns.mesh', cpp_namespace='::ns3')
    return root_module

def register_types(module):
    root_module = module.get_root()
    
    ## wifi-mac.h (module 'wifi'): ns3::TypeOfStation [enumeration]
    module.add_enum('TypeOfStation', ['STA', 'AP', 'ADHOC_STA', 'MESH', 'OCB'], import_from_module='ns.wifi')
    ## wifi-mac.h (module 'wifi'): ns3::WifiMacDropReason [enumeration]
    module.add_enum('WifiMacDropReason', ['WIFI_MAC_DROP_FAILED_ENQUEUE', 'WIFI_MAC_DROP_EXPIRED_LIFETIME', 'WIFI_MAC_DROP_REACHED_RETRY_LIMIT', 'WIFI_MAC_DROP_QOS_OLD_PACKET'], import_from_module='ns.wifi')
    ## wifi-phy-common.h (module 'wifi'): ns3::WifiPreamble [enumeration]
    module.add_enum('WifiPreamble', ['WIFI_PREAMBLE_LONG', 'WIFI_PREAMBLE_SHORT', 'WIFI_PREAMBLE_HT_MF', 'WIFI_PREAMBLE_VHT_SU', 'WIFI_PREAMBLE_VHT_MU', 'WIFI_PREAMBLE_HE_SU', 'WIFI_PREAMBLE_HE_ER_SU', 'WIFI_PREAMBLE_HE_MU', 'WIFI_PREAMBLE_HE_TB'], import_from_module='ns.wifi')
    ## wifi-phy-common.h (module 'wifi'): ns3::WifiModulationClass [enumeration]
    module.add_enum('WifiModulationClass', ['WIFI_MOD_CLASS_UNKNOWN', 'WIFI_MOD_CLASS_DSSS', 'WIFI_MOD_CLASS_HR_DSSS', 'WIFI_MOD_CLASS_ERP_OFDM', 'WIFI_MOD_CLASS_OFDM', 'WIFI_MOD_CLASS_HT', 'WIFI_MOD_CLASS_VHT', 'WIFI_MOD_CLASS_HE'], import_from_module='ns.wifi')
    ## wifi-phy-common.h (module 'wifi'): ns3::WifiPpduField [enumeration]
    module.add_enum('WifiPpduField', ['WIFI_PPDU_FIELD_PREAMBLE', 'WIFI_PPDU_FIELD_NON_HT_HEADER', 'WIFI_PPDU_FIELD_HT_SIG', 'WIFI_PPDU_FIELD_TRAINING', 'WIFI_PPDU_FIELD_SIG_A', 'WIFI_PPDU_FIELD_SIG_B', 'WIFI_PPDU_FIELD_DATA'], import_from_module='ns.wifi')
    ## wifi-phy-common.h (module 'wifi'): ns3::WifiPpduType [enumeration]
    module.add_enum('WifiPpduType', ['WIFI_PPDU_TYPE_SU', 'WIFI_PPDU_TYPE_DL_MU', 'WIFI_PPDU_TYPE_UL_MU'], import_from_module='ns.wifi')
    ## wifi-phy-common.h (module 'wifi'): ns3::WifiPhyRxfailureReason [enumeration]
    module.add_enum('WifiPhyRxfailureReason', ['UNKNOWN', 'UNSUPPORTED_SETTINGS', 'CHANNEL_SWITCHING', 'RXING', 'TXING', 'SLEEPING', 'POWERED_OFF', 'TRUNCATED_TX', 'BUSY_DECODING_PREAMBLE', 'PREAMBLE_DETECT_FAILURE', 'RECEPTION_ABORTED_BY_TX', 'L_SIG_FAILURE', 'HT_SIG_FAILURE', 'SIG_A_FAILURE', 'SIG_B_FAILURE', 'PREAMBLE_DETECTION_PACKET_SWITCH', 'FRAME_CAPTURE_PACKET_SWITCH', 'OBSS_PD_CCA_RESET', 'HE_TB_PPDU_TOO_LATE', 'FILTERED'], import_from_module='ns.wifi')
    ## ht-operation.h (module 'wifi'): ns3::HtProtectionType [enumeration]
    module.add_enum('HtProtectionType', ['NO_PROTECTION', 'NON_MEMBER_PROTECTION', 'TWENTY_MHZ_PROTECTION', 'MIXED_MODE_PROTECTION'], import_from_module='ns.wifi')
    ## qos-utils.h (module 'wifi'): ns3::AcIndex [enumeration]
    module.add_enum('AcIndex', ['AC_BE', 'AC_BK', 'AC_VI', 'AC_VO', 'AC_BE_NQOS', 'AC_BEACON', 'AC_UNDEF'], import_from_module='ns.wifi')
    ## wifi-standards.h (module 'wifi'): ns3::WifiStandard [enumeration]
    module.add_enum('WifiStandard', ['WIFI_STANDARD_UNSPECIFIED', 'WIFI_STANDARD_80211a', 'WIFI_STANDARD_80211b', 'WIFI_STANDARD_80211g', 'WIFI_STANDARD_80211p', 'WIFI_STANDARD_80211n', 'WIFI_STANDARD_80211ac', 'WIFI_STANDARD_80211ax'], import_from_module='ns.wifi')
    ## wifi-standards.h (module 'wifi'): ns3::FrequencyChannelType [enumeration]
    module.add_enum('FrequencyChannelType', ['WIFI_PHY_DSSS_CHANNEL', 'WIFI_PHY_OFDM_CHANNEL', 'WIFI_PHY_80211p_CHANNEL'], import_from_module='ns.wifi')
    ## wifi-phy-band.h (module 'wifi'): ns3::WifiPhyBand [enumeration]
    module.add_enum('WifiPhyBand', ['WIFI_PHY_BAND_2_4GHZ', 'WIFI_PHY_BAND_5GHZ', 'WIFI_PHY_BAND_6GHZ', 'WIFI_PHY_BAND_UNSPECIFIED'], import_from_module='ns.wifi')
    ## log.h (module 'core'): ns3::LogLevel [enumeration]
    module.add_enum('LogLevel', ['LOG_NONE', 'LOG_ERROR', 'LOG_LEVEL_ERROR', 'LOG_WARN', 'LOG_LEVEL_WARN', 'LOG_DEBUG', 'LOG_LEVEL_DEBUG', 'LOG_INFO', 'LOG_LEVEL_INFO', 'LOG_FUNCTION', 'LOG_LEVEL_FUNCTION', 'LOG_LOGIC', 'LOG_LEVEL_LOGIC', 'LOG_ALL', 'LOG_LEVEL_ALL', 'LOG_PREFIX_FUNC', 'LOG_PREFIX_TIME', 'LOG_PREFIX_NODE', 'LOG_PREFIX_LEVEL', 'LOG_PREFIX_ALL'], import_from_module='ns.core')
    ## address.h (module 'network'): ns3::Address [class]
    module.add_class('Address', import_from_module='ns.network')
    ## address.h (module 'network'): ns3::Address::MaxSize_e [enumeration]
    module.add_enum('MaxSize_e', ['MAX_SIZE'], outer_class=root_module['ns3::Address'], import_from_module='ns.network')
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList [class]
    module.add_class('AttributeConstructionList', import_from_module='ns.core')
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::Item [struct]
    module.add_class('Item', import_from_module='ns.core', outer_class=root_module['ns3::AttributeConstructionList'])
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator', 'ns3::AttributeConstructionList::CIterator')
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator*', 'ns3::AttributeConstructionList::CIterator*')
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator&', 'ns3::AttributeConstructionList::CIterator&')
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckReqType [struct]
    module.add_class('BlockAckReqType', import_from_module='ns.wifi')
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckReqType::Variant [enumeration]
    module.add_enum('Variant', ['BASIC', 'COMPRESSED', 'EXTENDED_COMPRESSED', 'MULTI_TID'], outer_class=root_module['ns3::BlockAckReqType'], import_from_module='ns.wifi')
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckType [struct]
    module.add_class('BlockAckType', import_from_module='ns.wifi')
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckType::Variant [enumeration]
    module.add_enum('Variant', ['BASIC', 'COMPRESSED', 'EXTENDED_COMPRESSED', 'MULTI_TID', 'MULTI_STA'], outer_class=root_module['ns3::BlockAckType'], import_from_module='ns.wifi')
    ## buffer.h (module 'network'): ns3::Buffer [class]
    module.add_class('Buffer', import_from_module='ns.network')
    ## buffer.h (module 'network'): ns3::Buffer::Iterator [class]
    module.add_class('Iterator', import_from_module='ns.network', outer_class=root_module['ns3::Buffer'])
    ## packet.h (module 'network'): ns3::ByteTagIterator [class]
    module.add_class('ByteTagIterator', import_from_module='ns.network')
    ## packet.h (module 'network'): ns3::ByteTagIterator::Item [class]
    module.add_class('Item', import_from_module='ns.network', outer_class=root_module['ns3::ByteTagIterator'])
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList [class]
    module.add_class('ByteTagList', import_from_module='ns.network')
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator [class]
    module.add_class('Iterator', import_from_module='ns.network', outer_class=root_module['ns3::ByteTagList'])
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Item [struct]
    module.add_class('Item', import_from_module='ns.network', outer_class=root_module['ns3::ByteTagList::Iterator'])
    ## callback.h (module 'core'): ns3::CallbackBase [class]
    module.add_class('CallbackBase', import_from_module='ns.core')
    ## capability-information.h (module 'wifi'): ns3::CapabilityInformation [class]
    module.add_class('CapabilityInformation', import_from_module='ns.wifi')
    ## data-rate.h (module 'network'): ns3::DataRate [class]
    module.add_class('DataRate', import_from_module='ns.network')
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::AttributeAccessor> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::AttributeAccessor'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::AttributeChecker> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::AttributeChecker'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::AttributeValue> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::AttributeValue'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::CallbackImplBase> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::CallbackImplBase'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::EventImpl> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::EventImpl'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::Hash::Implementation> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::Hash::Implementation'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin> [struct]
    module.add_class('DefaultDeleter', template_parameters=['ns3::MeshWifiInterfaceMacPlugin'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::NixVector> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::NixVector'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::Packet> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::Packet'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::TraceSourceAccessor> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::TraceSourceAccessor'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::WifiInformationElement> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::WifiInformationElement'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit> [struct]
    module.add_class('DefaultDeleter', template_parameters=['ns3::dot11s::DestinationAddressUnit'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit> [struct]
    module.add_class('DefaultDeleter', template_parameters=['ns3::dot11s::IeBeaconTimingUnit'])
    ## event-id.h (module 'core'): ns3::EventId [class]
    module.add_class('EventId', import_from_module='ns.core')
    ## event-id.h (module 'core'): ns3::EventId::UID [enumeration]
    module.add_enum('UID', ['INVALID', 'NOW', 'DESTROY', 'RESERVED', 'VALID'], outer_class=root_module['ns3::EventId'], import_from_module='ns.core')
    ## hash.h (module 'core'): ns3::Hasher [class]
    module.add_class('Hasher', import_from_module='ns.core')
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address [class]
    module.add_class('Ipv4Address', import_from_module='ns.network')
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address [class]
    root_module['ns3::Ipv4Address'].implicitly_converts_to(root_module['ns3::Address'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressHash [class]
    module.add_class('Ipv4AddressHash', import_from_module='ns.network')
    ## ipv4-address.h (module 'network'): ns3::Ipv4Mask [class]
    module.add_class('Ipv4Mask', import_from_module='ns.network')
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address [class]
    module.add_class('Ipv6Address', import_from_module='ns.network')
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address [class]
    root_module['ns3::Ipv6Address'].implicitly_converts_to(root_module['ns3::Address'])
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressHash [class]
    module.add_class('Ipv6AddressHash', import_from_module='ns.network')
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix [class]
    module.add_class('Ipv6Prefix', import_from_module='ns.network')
    ## log.h (module 'core'): ns3::LogComponent [class]
    module.add_class('LogComponent', import_from_module='ns.core')
    typehandlers.add_type_alias('std::map< std::string, ns3::LogComponent * >', 'ns3::LogComponent::ComponentList')
    typehandlers.add_type_alias('std::map< std::string, ns3::LogComponent * >*', 'ns3::LogComponent::ComponentList*')
    typehandlers.add_type_alias('std::map< std::string, ns3::LogComponent * >&', 'ns3::LogComponent::ComponentList&')
    ## mac48-address.h (module 'network'): ns3::Mac48Address [class]
    module.add_class('Mac48Address', import_from_module='ns.network')
    typehandlers.add_type_alias('void ( * ) ( ns3::Mac48Address )', 'ns3::Mac48Address::TracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Mac48Address )*', 'ns3::Mac48Address::TracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Mac48Address )&', 'ns3::Mac48Address::TracedCallback&')
    ## mac48-address.h (module 'network'): ns3::Mac48Address [class]
    root_module['ns3::Mac48Address'].implicitly_converts_to(root_module['ns3::Address'])
    ## mac8-address.h (module 'network'): ns3::Mac8Address [class]
    module.add_class('Mac8Address', import_from_module='ns.network')
    ## mac8-address.h (module 'network'): ns3::Mac8Address [class]
    root_module['ns3::Mac8Address'].implicitly_converts_to(root_module['ns3::Address'])
    ## mesh-helper.h (module 'mesh'): ns3::MeshHelper [class]
    module.add_class('MeshHelper')
    ## mesh-helper.h (module 'mesh'): ns3::MeshHelper::ChannelPolicy [enumeration]
    module.add_enum('ChannelPolicy', ['SPREAD_CHANNELS', 'ZERO_CHANNEL'], outer_class=root_module['ns3::MeshHelper'])
    ## mesh-wifi-beacon.h (module 'mesh'): ns3::MeshWifiBeacon [class]
    module.add_class('MeshWifiBeacon')
    ## object-base.h (module 'core'): ns3::ObjectBase [class]
    module.add_class('ObjectBase', allow_subclassing=True, import_from_module='ns.core')
    ## object.h (module 'core'): ns3::ObjectDeleter [struct]
    module.add_class('ObjectDeleter', import_from_module='ns.core')
    ## object-factory.h (module 'core'): ns3::ObjectFactory [class]
    module.add_class('ObjectFactory', import_from_module='ns.core')
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata [class]
    module.add_class('PacketMetadata', import_from_module='ns.network')
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item [struct]
    module.add_class('Item', import_from_module='ns.network', outer_class=root_module['ns3::PacketMetadata'])
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::ItemType [enumeration]
    module.add_enum('ItemType', ['PAYLOAD', 'HEADER', 'TRAILER'], outer_class=root_module['ns3::PacketMetadata::Item'], import_from_module='ns.network')
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::ItemIterator [class]
    module.add_class('ItemIterator', import_from_module='ns.network', outer_class=root_module['ns3::PacketMetadata'])
    ## packet.h (module 'network'): ns3::PacketTagIterator [class]
    module.add_class('PacketTagIterator', import_from_module='ns.network')
    ## packet.h (module 'network'): ns3::PacketTagIterator::Item [class]
    module.add_class('Item', import_from_module='ns.network', outer_class=root_module['ns3::PacketTagIterator'])
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList [class]
    module.add_class('PacketTagList', import_from_module='ns.network')
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::TagData [struct]
    module.add_class('TagData', import_from_module='ns.network', outer_class=root_module['ns3::PacketTagList'])
    ## log.h (module 'core'): ns3::ParameterLogger [class]
    module.add_class('ParameterLogger', import_from_module='ns.core')
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::ObjectBase'], template_parameters=['ns3::Object', 'ns3::ObjectBase', 'ns3::ObjectDeleter'])
    ## status-code.h (module 'wifi'): ns3::StatusCode [class]
    module.add_class('StatusCode', import_from_module='ns.wifi')
    ## tag.h (module 'network'): ns3::Tag [class]
    module.add_class('Tag', import_from_module='ns.network', parent=root_module['ns3::ObjectBase'])
    ## tag-buffer.h (module 'network'): ns3::TagBuffer [class]
    module.add_class('TagBuffer', import_from_module='ns.network')
    ## nstime.h (module 'core'): ns3::Time [class]
    module.add_class('Time', import_from_module='ns.core')
    ## nstime.h (module 'core'): ns3::Time::Unit [enumeration]
    module.add_enum('Unit', ['Y', 'D', 'H', 'MIN', 'S', 'MS', 'US', 'NS', 'PS', 'FS', 'LAST', 'AUTO'], outer_class=root_module['ns3::Time'], import_from_module='ns.core')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time )', 'ns3::Time::TracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time )*', 'ns3::Time::TracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time )&', 'ns3::Time::TracedCallback&')
    ## nstime.h (module 'core'): ns3::TimeWithUnit [class]
    module.add_class('TimeWithUnit', import_from_module='ns.core')
    ## type-id.h (module 'core'): ns3::TypeId [class]
    module.add_class('TypeId', import_from_module='ns.core')
    ## type-id.h (module 'core'): ns3::TypeId::AttributeFlag [enumeration]
    module.add_enum('AttributeFlag', ['ATTR_GET', 'ATTR_SET', 'ATTR_CONSTRUCT', 'ATTR_SGC'], outer_class=root_module['ns3::TypeId'], import_from_module='ns.core')
    ## type-id.h (module 'core'): ns3::TypeId::SupportLevel [enumeration]
    module.add_enum('SupportLevel', ['SUPPORTED', 'DEPRECATED', 'OBSOLETE'], outer_class=root_module['ns3::TypeId'], import_from_module='ns.core')
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation [struct]
    module.add_class('AttributeInformation', import_from_module='ns.core', outer_class=root_module['ns3::TypeId'])
    ## type-id.h (module 'core'): ns3::TypeId::TraceSourceInformation [struct]
    module.add_class('TraceSourceInformation', import_from_module='ns.core', outer_class=root_module['ns3::TypeId'])
    typehandlers.add_type_alias('uint32_t', 'ns3::TypeId::hash_t')
    typehandlers.add_type_alias('uint32_t*', 'ns3::TypeId::hash_t*')
    typehandlers.add_type_alias('uint32_t&', 'ns3::TypeId::hash_t&')
    ## qos-utils.h (module 'wifi'): ns3::WifiAc [class]
    module.add_class('WifiAc', import_from_module='ns.wifi')
    ## qos-utils.h (module 'wifi'): ns3::WifiAddressHash [struct]
    module.add_class('WifiAddressHash', import_from_module='ns.wifi')
    ## qos-utils.h (module 'wifi'): ns3::WifiAddressTidHash [struct]
    module.add_class('WifiAddressTidHash', import_from_module='ns.wifi')
    ## wifi-mode.h (module 'wifi'): ns3::WifiMode [class]
    module.add_class('WifiMode', import_from_module='ns.wifi')
    ## wifi-mode.h (module 'wifi'): ns3::WifiModeFactory [class]
    module.add_class('WifiModeFactory', import_from_module='ns.wifi')
    typehandlers.add_type_alias('ns3::Callback< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::WifiModeFactory::CodeRateCallback')
    typehandlers.add_type_alias('ns3::Callback< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::WifiModeFactory::CodeRateCallback*')
    typehandlers.add_type_alias('ns3::Callback< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::WifiModeFactory::CodeRateCallback&')
    typehandlers.add_type_alias('ns3::Callback< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::WifiModeFactory::ConstellationSizeCallback')
    typehandlers.add_type_alias('ns3::Callback< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::WifiModeFactory::ConstellationSizeCallback*')
    typehandlers.add_type_alias('ns3::Callback< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::WifiModeFactory::ConstellationSizeCallback&')
    typehandlers.add_type_alias('ns3::Callback< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::WifiModeFactory::PhyRateCallback')
    typehandlers.add_type_alias('ns3::Callback< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::WifiModeFactory::PhyRateCallback*')
    typehandlers.add_type_alias('ns3::Callback< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::WifiModeFactory::PhyRateCallback&')
    typehandlers.add_type_alias('ns3::Callback< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::WifiModeFactory::DataRateCallback')
    typehandlers.add_type_alias('ns3::Callback< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::WifiModeFactory::DataRateCallback*')
    typehandlers.add_type_alias('ns3::Callback< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::WifiModeFactory::DataRateCallback&')
    typehandlers.add_type_alias('ns3::Callback< unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::WifiModeFactory::NonHtReferenceRateCallback')
    typehandlers.add_type_alias('ns3::Callback< unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::WifiModeFactory::NonHtReferenceRateCallback*')
    typehandlers.add_type_alias('ns3::Callback< unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::WifiModeFactory::NonHtReferenceRateCallback&')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::WifiTxVector const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::WifiModeFactory::AllowedCallback')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::WifiTxVector const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::WifiModeFactory::AllowedCallback*')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::WifiTxVector const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::WifiModeFactory::AllowedCallback&')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStation [struct]
    module.add_class('WifiRemoteStation', import_from_module='ns.wifi')
    ## wifi-remote-station-info.h (module 'wifi'): ns3::WifiRemoteStationInfo [class]
    module.add_class('WifiRemoteStationInfo', import_from_module='ns.wifi')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState [struct]
    module.add_class('WifiRemoteStationState', import_from_module='ns.wifi')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState [enumeration]
    module.add_enum('', ['BRAND_NEW', 'DISASSOC', 'WAIT_ASSOC_TX_OK', 'GOT_ASSOC_TX_OK'], outer_class=root_module['ns3::WifiRemoteStationState'], import_from_module='ns.wifi')
    ## empty.h (module 'core'): ns3::empty [class]
    module.add_class('empty', import_from_module='ns.core')
    ## int64x64-128.h (module 'core'): ns3::int64x64_t [class]
    module.add_class('int64x64_t', import_from_module='ns.core')
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::impl_type [enumeration]
    module.add_enum('impl_type', ['int128_impl', 'cairo_impl', 'ld_impl'], outer_class=root_module['ns3::int64x64_t'], import_from_module='ns.core')
    ## chunk.h (module 'network'): ns3::Chunk [class]
    module.add_class('Chunk', import_from_module='ns.network', parent=root_module['ns3::ObjectBase'])
    ## header.h (module 'network'): ns3::Header [class]
    module.add_class('Header', import_from_module='ns.network', parent=root_module['ns3::Chunk'])
    ## mgt-headers.h (module 'wifi'): ns3::MgtAddBaRequestHeader [class]
    module.add_class('MgtAddBaRequestHeader', import_from_module='ns.wifi', parent=root_module['ns3::Header'])
    ## mgt-headers.h (module 'wifi'): ns3::MgtAddBaResponseHeader [class]
    module.add_class('MgtAddBaResponseHeader', import_from_module='ns.wifi', parent=root_module['ns3::Header'])
    ## mgt-headers.h (module 'wifi'): ns3::MgtAssocRequestHeader [class]
    module.add_class('MgtAssocRequestHeader', import_from_module='ns.wifi', parent=root_module['ns3::Header'])
    ## mgt-headers.h (module 'wifi'): ns3::MgtAssocResponseHeader [class]
    module.add_class('MgtAssocResponseHeader', import_from_module='ns.wifi', parent=root_module['ns3::Header'])
    ## mgt-headers.h (module 'wifi'): ns3::MgtDelBaHeader [class]
    module.add_class('MgtDelBaHeader', import_from_module='ns.wifi', parent=root_module['ns3::Header'])
    ## mgt-headers.h (module 'wifi'): ns3::MgtProbeRequestHeader [class]
    module.add_class('MgtProbeRequestHeader', import_from_module='ns.wifi', parent=root_module['ns3::Header'])
    ## mgt-headers.h (module 'wifi'): ns3::MgtProbeResponseHeader [class]
    module.add_class('MgtProbeResponseHeader', import_from_module='ns.wifi', parent=root_module['ns3::Header'])
    ## mgt-headers.h (module 'wifi'): ns3::MgtReassocRequestHeader [class]
    module.add_class('MgtReassocRequestHeader', import_from_module='ns.wifi', parent=root_module['ns3::Header'])
    ## object.h (module 'core'): ns3::Object [class]
    module.add_class('Object', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter >'])
    ## object.h (module 'core'): ns3::Object::AggregateIterator [class]
    module.add_class('AggregateIterator', import_from_module='ns.core', outer_class=root_module['ns3::Object'])
    ## random-variable-stream.h (module 'core'): ns3::RandomVariableStream [class]
    module.add_class('RandomVariableStream', import_from_module='ns.core', parent=root_module['ns3::Object'])
    ## random-variable-stream.h (module 'core'): ns3::SequentialRandomVariable [class]
    module.add_class('SequentialRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::AttributeAccessor, ns3::empty, ns3::DefaultDeleter<ns3::AttributeAccessor>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::AttributeAccessor', 'ns3::empty', 'ns3::DefaultDeleter<ns3::AttributeAccessor>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::AttributeChecker, ns3::empty, ns3::DefaultDeleter<ns3::AttributeChecker>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::AttributeChecker', 'ns3::empty', 'ns3::DefaultDeleter<ns3::AttributeChecker>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::AttributeValue, ns3::empty, ns3::DefaultDeleter<ns3::AttributeValue>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::AttributeValue', 'ns3::empty', 'ns3::DefaultDeleter<ns3::AttributeValue>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter<ns3::CallbackImplBase>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::CallbackImplBase', 'ns3::empty', 'ns3::DefaultDeleter<ns3::CallbackImplBase>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::EventImpl', 'ns3::empty', 'ns3::DefaultDeleter<ns3::EventImpl>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Hash::Implementation, ns3::empty, ns3::DefaultDeleter<ns3::Hash::Implementation>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::Hash::Implementation', 'ns3::empty', 'ns3::DefaultDeleter<ns3::Hash::Implementation>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::MeshWifiInterfaceMacPlugin, ns3::empty, ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::MeshWifiInterfaceMacPlugin', 'ns3::empty', 'ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::NixVector', 'ns3::empty', 'ns3::DefaultDeleter<ns3::NixVector>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::Packet', 'ns3::empty', 'ns3::DefaultDeleter<ns3::Packet>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::TraceSourceAccessor', 'ns3::empty', 'ns3::DefaultDeleter<ns3::TraceSourceAccessor>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::WifiInformationElement, ns3::empty, ns3::DefaultDeleter<ns3::WifiInformationElement>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::WifiInformationElement', 'ns3::empty', 'ns3::DefaultDeleter<ns3::WifiInformationElement>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::dot11s::DestinationAddressUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::dot11s::DestinationAddressUnit', 'ns3::empty', 'ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::dot11s::IeBeaconTimingUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::dot11s::IeBeaconTimingUnit', 'ns3::empty', 'ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit>'])
    ## trace-source-accessor.h (module 'core'): ns3::TraceSourceAccessor [class]
    module.add_class('TraceSourceAccessor', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor> >'])
    ## trailer.h (module 'network'): ns3::Trailer [class]
    module.add_class('Trailer', import_from_module='ns.network', parent=root_module['ns3::Chunk'])
    ## random-variable-stream.h (module 'core'): ns3::TriangularRandomVariable [class]
    module.add_class('TriangularRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## random-variable-stream.h (module 'core'): ns3::UniformRandomVariable [class]
    module.add_class('UniformRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## random-variable-stream.h (module 'core'): ns3::WeibullRandomVariable [class]
    module.add_class('WeibullRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader [class]
    module.add_class('WifiActionHeader', import_from_module='ns.wifi', parent=root_module['ns3::Header'])
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::CategoryValue [enumeration]
    module.add_enum('CategoryValue', ['BLOCK_ACK', 'MESH', 'MULTIHOP', 'SELF_PROTECTED', 'VENDOR_SPECIFIC_ACTION'], outer_class=root_module['ns3::WifiActionHeader'], import_from_module='ns.wifi')
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::SelfProtectedActionValue [enumeration]
    module.add_enum('SelfProtectedActionValue', ['PEER_LINK_OPEN', 'PEER_LINK_CONFIRM', 'PEER_LINK_CLOSE', 'GROUP_KEY_INFORM', 'GROUP_KEY_ACK'], outer_class=root_module['ns3::WifiActionHeader'], import_from_module='ns.wifi')
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::MultihopActionValue [enumeration]
    module.add_enum('MultihopActionValue', ['PROXY_UPDATE', 'PROXY_UPDATE_CONFIRMATION'], outer_class=root_module['ns3::WifiActionHeader'], import_from_module='ns.wifi')
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::MeshActionValue [enumeration]
    module.add_enum('MeshActionValue', ['LINK_METRIC_REPORT', 'PATH_SELECTION', 'PORTAL_ANNOUNCEMENT', 'CONGESTION_CONTROL_NOTIFICATION', 'MDA_SETUP_REQUEST', 'MDA_SETUP_REPLY', 'MDAOP_ADVERTISMENT_REQUEST', 'MDAOP_ADVERTISMENTS', 'MDAOP_SET_TEARDOWN', 'TBTT_ADJUSTMENT_REQUEST', 'TBTT_ADJUSTMENT_RESPONSE'], outer_class=root_module['ns3::WifiActionHeader'], import_from_module='ns.wifi')
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::BlockAckActionValue [enumeration]
    module.add_enum('BlockAckActionValue', ['BLOCK_ACK_ADDBA_REQUEST', 'BLOCK_ACK_ADDBA_RESPONSE', 'BLOCK_ACK_DELBA'], outer_class=root_module['ns3::WifiActionHeader'], import_from_module='ns.wifi')
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::ActionValue [union]
    module.add_class('ActionValue', import_from_module='ns.wifi', outer_class=root_module['ns3::WifiActionHeader'])
    typehandlers.add_type_alias('ns3::WifiActionHeader::ActionValue', 'ns3::WifiActionHeader::ActionValue')
    typehandlers.add_type_alias('ns3::WifiActionHeader::ActionValue*', 'ns3::WifiActionHeader::ActionValue*')
    typehandlers.add_type_alias('ns3::WifiActionHeader::ActionValue&', 'ns3::WifiActionHeader::ActionValue&')
    module.add_typedef(root_module['ns3::WifiActionHeader::ActionValue'], 'ActionValue')
    ## wifi-information-element.h (module 'wifi'): ns3::WifiInformationElement [class]
    module.add_class('WifiInformationElement', import_from_module='ns.wifi', parent=root_module['ns3::SimpleRefCount< ns3::WifiInformationElement, ns3::empty, ns3::DefaultDeleter<ns3::WifiInformationElement> >'])
    ## wifi-information-element-vector.h (module 'wifi'): ns3::WifiInformationElementVector [class]
    module.add_class('WifiInformationElementVector', import_from_module='ns.wifi', parent=root_module['ns3::Header'])
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::WifiInformationElement > > iterator', 'ns3::WifiInformationElementVector::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::WifiInformationElement > > iterator*', 'ns3::WifiInformationElementVector::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::WifiInformationElement > > iterator&', 'ns3::WifiInformationElementVector::Iterator&')
    ## wifi-mac.h (module 'wifi'): ns3::WifiMac [class]
    module.add_class('WifiMac', import_from_module='ns.wifi', parent=root_module['ns3::Object'])
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::Packet const >, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::WifiMac::ForwardUpCallback')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::Packet const >, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::WifiMac::ForwardUpCallback*')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::Packet const >, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::WifiMac::ForwardUpCallback&')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationManager [class]
    module.add_class('WifiRemoteStationManager', import_from_module='ns.wifi', parent=root_module['ns3::Object'])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationManager::ProtectionMode [enumeration]
    module.add_enum('ProtectionMode', ['RTS_CTS', 'CTS_TO_SELF'], outer_class=root_module['ns3::WifiRemoteStationManager'], import_from_module='ns.wifi')
    typehandlers.add_type_alias('std::unordered_map< ns3::Mac48Address, ns3::WifiRemoteStation *, ns3::WifiAddressHash, std::equal_to< ns3::Mac48Address >, std::allocator< std::pair< ns3::Mac48Address const, ns3::WifiRemoteStation * > > >', 'ns3::WifiRemoteStationManager::Stations')
    typehandlers.add_type_alias('std::unordered_map< ns3::Mac48Address, ns3::WifiRemoteStation *, ns3::WifiAddressHash, std::equal_to< ns3::Mac48Address >, std::allocator< std::pair< ns3::Mac48Address const, ns3::WifiRemoteStation * > > >*', 'ns3::WifiRemoteStationManager::Stations*')
    typehandlers.add_type_alias('std::unordered_map< ns3::Mac48Address, ns3::WifiRemoteStation *, ns3::WifiAddressHash, std::equal_to< ns3::Mac48Address >, std::allocator< std::pair< ns3::Mac48Address const, ns3::WifiRemoteStation * > > >&', 'ns3::WifiRemoteStationManager::Stations&')
    typehandlers.add_type_alias('std::unordered_map< ns3::Mac48Address, ns3::WifiRemoteStationState *, ns3::WifiAddressHash, std::equal_to< ns3::Mac48Address >, std::allocator< std::pair< ns3::Mac48Address const, ns3::WifiRemoteStationState * > > >', 'ns3::WifiRemoteStationManager::StationStates')
    typehandlers.add_type_alias('std::unordered_map< ns3::Mac48Address, ns3::WifiRemoteStationState *, ns3::WifiAddressHash, std::equal_to< ns3::Mac48Address >, std::allocator< std::pair< ns3::Mac48Address const, ns3::WifiRemoteStationState * > > >*', 'ns3::WifiRemoteStationManager::StationStates*')
    typehandlers.add_type_alias('std::unordered_map< ns3::Mac48Address, ns3::WifiRemoteStationState *, ns3::WifiAddressHash, std::equal_to< ns3::Mac48Address >, std::allocator< std::pair< ns3::Mac48Address const, ns3::WifiRemoteStationState * > > >&', 'ns3::WifiRemoteStationManager::StationStates&')
    typehandlers.add_type_alias('void ( * ) ( double, double, ns3::Mac48Address )', 'ns3::WifiRemoteStationManager::PowerChangeTracedCallback')
    typehandlers.add_type_alias('void ( * ) ( double, double, ns3::Mac48Address )*', 'ns3::WifiRemoteStationManager::PowerChangeTracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( double, double, ns3::Mac48Address )&', 'ns3::WifiRemoteStationManager::PowerChangeTracedCallback&')
    typehandlers.add_type_alias('void ( * ) ( ns3::DataRate, ns3::DataRate, ns3::Mac48Address )', 'ns3::WifiRemoteStationManager::RateChangeTracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::DataRate, ns3::DataRate, ns3::Mac48Address )*', 'ns3::WifiRemoteStationManager::RateChangeTracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::DataRate, ns3::DataRate, ns3::Mac48Address )&', 'ns3::WifiRemoteStationManager::RateChangeTracedCallback&')
    ## random-variable-stream.h (module 'core'): ns3::ZetaRandomVariable [class]
    module.add_class('ZetaRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## random-variable-stream.h (module 'core'): ns3::ZipfRandomVariable [class]
    module.add_class('ZipfRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## attribute.h (module 'core'): ns3::AttributeAccessor [class]
    module.add_class('AttributeAccessor', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::AttributeAccessor, ns3::empty, ns3::DefaultDeleter<ns3::AttributeAccessor> >'])
    ## attribute.h (module 'core'): ns3::AttributeChecker [class]
    module.add_class('AttributeChecker', allow_subclassing=False, automatic_type_narrowing=True, import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::AttributeChecker, ns3::empty, ns3::DefaultDeleter<ns3::AttributeChecker> >'])
    ## attribute.h (module 'core'): ns3::AttributeValue [class]
    module.add_class('AttributeValue', allow_subclassing=False, automatic_type_narrowing=True, import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::AttributeValue, ns3::empty, ns3::DefaultDeleter<ns3::AttributeValue> >'])
    ## boolean.h (module 'core'): ns3::BooleanChecker [class]
    module.add_class('BooleanChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## boolean.h (module 'core'): ns3::BooleanValue [class]
    module.add_class('BooleanValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## callback.h (module 'core'): ns3::CallbackChecker [class]
    module.add_class('CallbackChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## callback.h (module 'core'): ns3::CallbackImplBase [class]
    module.add_class('CallbackImplBase', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter<ns3::CallbackImplBase> >'])
    ## callback.h (module 'core'): ns3::CallbackValue [class]
    module.add_class('CallbackValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## channel.h (module 'network'): ns3::Channel [class]
    module.add_class('Channel', import_from_module='ns.network', parent=root_module['ns3::Object'])
    ## random-variable-stream.h (module 'core'): ns3::ConstantRandomVariable [class]
    module.add_class('ConstantRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## data-rate.h (module 'network'): ns3::DataRateChecker [class]
    module.add_class('DataRateChecker', import_from_module='ns.network', parent=root_module['ns3::AttributeChecker'])
    ## data-rate.h (module 'network'): ns3::DataRateValue [class]
    module.add_class('DataRateValue', import_from_module='ns.network', parent=root_module['ns3::AttributeValue'])
    ## random-variable-stream.h (module 'core'): ns3::DeterministicRandomVariable [class]
    module.add_class('DeterministicRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## double.h (module 'core'): ns3::DoubleValue [class]
    module.add_class('DoubleValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## dsss-parameter-set.h (module 'wifi'): ns3::DsssParameterSet [class]
    module.add_class('DsssParameterSet', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## edca-parameter-set.h (module 'wifi'): ns3::EdcaParameterSet [class]
    module.add_class('EdcaParameterSet', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## random-variable-stream.h (module 'core'): ns3::EmpiricalRandomVariable [class]
    module.add_class('EmpiricalRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## attribute.h (module 'core'): ns3::EmptyAttributeAccessor [class]
    module.add_class('EmptyAttributeAccessor', import_from_module='ns.core', parent=root_module['ns3::AttributeAccessor'])
    ## attribute.h (module 'core'): ns3::EmptyAttributeChecker [class]
    module.add_class('EmptyAttributeChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## attribute.h (module 'core'): ns3::EmptyAttributeValue [class]
    module.add_class('EmptyAttributeValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## enum.h (module 'core'): ns3::EnumChecker [class]
    module.add_class('EnumChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## enum.h (module 'core'): ns3::EnumValue [class]
    module.add_class('EnumValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## random-variable-stream.h (module 'core'): ns3::ErlangRandomVariable [class]
    module.add_class('ErlangRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## erp-information.h (module 'wifi'): ns3::ErpInformation [class]
    module.add_class('ErpInformation', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## event-impl.h (module 'core'): ns3::EventImpl [class]
    module.add_class('EventImpl', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl> >'])
    ## random-variable-stream.h (module 'core'): ns3::ExponentialRandomVariable [class]
    module.add_class('ExponentialRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## extended-capabilities.h (module 'wifi'): ns3::ExtendedCapabilities [class]
    module.add_class('ExtendedCapabilities', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## supported-rates.h (module 'wifi'): ns3::ExtendedSupportedRatesIE [class]
    module.add_class('ExtendedSupportedRatesIE', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## random-variable-stream.h (module 'core'): ns3::GammaRandomVariable [class]
    module.add_class('GammaRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## he-capabilities.h (module 'wifi'): ns3::HeCapabilities [class]
    module.add_class('HeCapabilities', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## he-operation.h (module 'wifi'): ns3::HeOperation [class]
    module.add_class('HeOperation', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## ht-capabilities.h (module 'wifi'): ns3::HtCapabilities [class]
    module.add_class('HtCapabilities', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## ht-operation.h (module 'wifi'): ns3::HtOperation [class]
    module.add_class('HtOperation', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## integer.h (module 'core'): ns3::IntegerValue [class]
    module.add_class('IntegerValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressChecker [class]
    module.add_class('Ipv4AddressChecker', import_from_module='ns.network', parent=root_module['ns3::AttributeChecker'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressValue [class]
    module.add_class('Ipv4AddressValue', import_from_module='ns.network', parent=root_module['ns3::AttributeValue'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4MaskChecker [class]
    module.add_class('Ipv4MaskChecker', import_from_module='ns.network', parent=root_module['ns3::AttributeChecker'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4MaskValue [class]
    module.add_class('Ipv4MaskValue', import_from_module='ns.network', parent=root_module['ns3::AttributeValue'])
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressChecker [class]
    module.add_class('Ipv6AddressChecker', import_from_module='ns.network', parent=root_module['ns3::AttributeChecker'])
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressValue [class]
    module.add_class('Ipv6AddressValue', import_from_module='ns.network', parent=root_module['ns3::AttributeValue'])
    ## ipv6-address.h (module 'network'): ns3::Ipv6PrefixChecker [class]
    module.add_class('Ipv6PrefixChecker', import_from_module='ns.network', parent=root_module['ns3::AttributeChecker'])
    ## ipv6-address.h (module 'network'): ns3::Ipv6PrefixValue [class]
    module.add_class('Ipv6PrefixValue', import_from_module='ns.network', parent=root_module['ns3::AttributeValue'])
    ## random-variable-stream.h (module 'core'): ns3::LogNormalRandomVariable [class]
    module.add_class('LogNormalRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## mac48-address.h (module 'network'): ns3::Mac48AddressChecker [class]
    module.add_class('Mac48AddressChecker', import_from_module='ns.network', parent=root_module['ns3::AttributeChecker'])
    ## mac48-address.h (module 'network'): ns3::Mac48AddressValue [class]
    module.add_class('Mac48AddressValue', import_from_module='ns.network', parent=root_module['ns3::AttributeValue'])
    ## mesh-information-element-vector.h (module 'mesh'): ns3::MeshInformationElementVector [class]
    module.add_class('MeshInformationElementVector', parent=root_module['ns3::WifiInformationElementVector'])
    ## mesh-l2-routing-protocol.h (module 'mesh'): ns3::MeshL2RoutingProtocol [class]
    module.add_class('MeshL2RoutingProtocol', parent=root_module['ns3::Object'])
    typehandlers.add_type_alias('ns3::Callback< void, bool, ns3::Ptr< ns3::Packet >, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty >', 'ns3::MeshL2RoutingProtocol::RouteReplyCallback')
    typehandlers.add_type_alias('ns3::Callback< void, bool, ns3::Ptr< ns3::Packet >, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::MeshL2RoutingProtocol::RouteReplyCallback*')
    typehandlers.add_type_alias('ns3::Callback< void, bool, ns3::Ptr< ns3::Packet >, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::MeshL2RoutingProtocol::RouteReplyCallback&')
    ## mesh-stack-installer.h (module 'mesh'): ns3::MeshStack [class]
    module.add_class('MeshStack', parent=root_module['ns3::Object'])
    ## mesh-wifi-interface-mac.h (module 'mesh'): ns3::MeshWifiInterfaceMac [class]
    module.add_class('MeshWifiInterfaceMac', parent=root_module['ns3::WifiMac'])
    ## mesh-wifi-interface-mac-plugin.h (module 'mesh'): ns3::MeshWifiInterfaceMacPlugin [class]
    module.add_class('MeshWifiInterfaceMacPlugin', parent=root_module['ns3::SimpleRefCount< ns3::MeshWifiInterfaceMacPlugin, ns3::empty, ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin> >'])
    ## mgt-headers.h (module 'wifi'): ns3::MgtBeaconHeader [class]
    module.add_class('MgtBeaconHeader', import_from_module='ns.wifi', parent=root_module['ns3::MgtProbeResponseHeader'])
    ## mu-edca-parameter-set.h (module 'wifi'): ns3::MuEdcaParameterSet [class]
    module.add_class('MuEdcaParameterSet', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## net-device.h (module 'network'): ns3::NetDevice [class]
    module.add_class('NetDevice', import_from_module='ns.network', parent=root_module['ns3::Object'])
    ## net-device.h (module 'network'): ns3::NetDevice::PacketType [enumeration]
    module.add_enum('PacketType', ['PACKET_HOST', 'NS3_PACKET_HOST', 'PACKET_BROADCAST', 'NS3_PACKET_BROADCAST', 'PACKET_MULTICAST', 'NS3_PACKET_MULTICAST', 'PACKET_OTHERHOST', 'NS3_PACKET_OTHERHOST'], outer_class=root_module['ns3::NetDevice'], import_from_module='ns.network')
    typehandlers.add_type_alias('void ( * ) (  )', 'ns3::NetDevice::LinkChangeTracedCallback')
    typehandlers.add_type_alias('void ( * ) (  )*', 'ns3::NetDevice::LinkChangeTracedCallback*')
    typehandlers.add_type_alias('void ( * ) (  )&', 'ns3::NetDevice::LinkChangeTracedCallback&')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::NetDevice::ReceiveCallback')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::NetDevice::ReceiveCallback*')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::NetDevice::ReceiveCallback&')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >', 'ns3::NetDevice::PromiscReceiveCallback')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::NetDevice::PromiscReceiveCallback*')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::NetDevice::PromiscReceiveCallback&')
    ## nix-vector.h (module 'network'): ns3::NixVector [class]
    module.add_class('NixVector', import_from_module='ns.network', parent=root_module['ns3::SimpleRefCount< ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector> >'])
    ## node.h (module 'network'): ns3::Node [class]
    module.add_class('Node', import_from_module='ns.network', parent=root_module['ns3::Object'])
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >', 'ns3::Node::ProtocolHandler')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::Node::ProtocolHandler*')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::Node::ProtocolHandler&')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::NetDevice >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::Node::DeviceAdditionListener')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::NetDevice >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::Node::DeviceAdditionListener*')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::NetDevice >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::Node::DeviceAdditionListener&')
    ## random-variable-stream.h (module 'core'): ns3::NormalRandomVariable [class]
    module.add_class('NormalRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## object-factory.h (module 'core'): ns3::ObjectFactoryChecker [class]
    module.add_class('ObjectFactoryChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## object-factory.h (module 'core'): ns3::ObjectFactoryValue [class]
    module.add_class('ObjectFactoryValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## packet.h (module 'network'): ns3::Packet [class]
    module.add_class('Packet', import_from_module='ns.network', parent=root_module['ns3::SimpleRefCount< ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet> >'])
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const > )', 'ns3::Packet::TracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const > )*', 'ns3::Packet::TracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const > )&', 'ns3::Packet::TracedCallback&')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, ns3::Address const & )', 'ns3::Packet::AddressTracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, ns3::Address const & )*', 'ns3::Packet::AddressTracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, ns3::Address const & )&', 'ns3::Packet::AddressTracedCallback&')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const > const, ns3::Address const &, ns3::Address const & )', 'ns3::Packet::TwoAddressTracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const > const, ns3::Address const &, ns3::Address const & )*', 'ns3::Packet::TwoAddressTracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const > const, ns3::Address const &, ns3::Address const & )&', 'ns3::Packet::TwoAddressTracedCallback&')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, ns3::Mac48Address )', 'ns3::Packet::Mac48AddressTracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, ns3::Mac48Address )*', 'ns3::Packet::Mac48AddressTracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, ns3::Mac48Address )&', 'ns3::Packet::Mac48AddressTracedCallback&')
    typehandlers.add_type_alias('void ( * ) ( uint32_t, uint32_t )', 'ns3::Packet::SizeTracedCallback')
    typehandlers.add_type_alias('void ( * ) ( uint32_t, uint32_t )*', 'ns3::Packet::SizeTracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( uint32_t, uint32_t )&', 'ns3::Packet::SizeTracedCallback&')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, double )', 'ns3::Packet::SinrTracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, double )*', 'ns3::Packet::SinrTracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, double )&', 'ns3::Packet::SinrTracedCallback&')
    ## random-variable-stream.h (module 'core'): ns3::ParetoRandomVariable [class]
    module.add_class('ParetoRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## ssid.h (module 'wifi'): ns3::Ssid [class]
    module.add_class('Ssid', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## ssid.h (module 'wifi'): ns3::SsidChecker [class]
    module.add_class('SsidChecker', import_from_module='ns.wifi', parent=root_module['ns3::AttributeChecker'])
    ## ssid.h (module 'wifi'): ns3::SsidValue [class]
    module.add_class('SsidValue', import_from_module='ns.wifi', parent=root_module['ns3::AttributeValue'])
    ## supported-rates.h (module 'wifi'): ns3::SupportedRates [class]
    module.add_class('SupportedRates', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## nstime.h (module 'core'): ns3::TimeValue [class]
    module.add_class('TimeValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## type-id.h (module 'core'): ns3::TypeIdChecker [class]
    module.add_class('TypeIdChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## type-id.h (module 'core'): ns3::TypeIdValue [class]
    module.add_class('TypeIdValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## uinteger.h (module 'core'): ns3::UintegerValue [class]
    module.add_class('UintegerValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## vht-capabilities.h (module 'wifi'): ns3::VhtCapabilities [class]
    module.add_class('VhtCapabilities', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## vht-operation.h (module 'wifi'): ns3::VhtOperation [class]
    module.add_class('VhtOperation', import_from_module='ns.wifi', parent=root_module['ns3::WifiInformationElement'])
    ## wifi-mode.h (module 'wifi'): ns3::WifiModeChecker [class]
    module.add_class('WifiModeChecker', import_from_module='ns.wifi', parent=root_module['ns3::AttributeChecker'])
    ## wifi-mode.h (module 'wifi'): ns3::WifiModeValue [class]
    module.add_class('WifiModeValue', import_from_module='ns.wifi', parent=root_module['ns3::AttributeValue'])
    ## address.h (module 'network'): ns3::AddressChecker [class]
    module.add_class('AddressChecker', import_from_module='ns.network', parent=root_module['ns3::AttributeChecker'])
    ## address.h (module 'network'): ns3::AddressValue [class]
    module.add_class('AddressValue', import_from_module='ns.network', parent=root_module['ns3::AttributeValue'])
    ## bridge-channel.h (module 'bridge'): ns3::BridgeChannel [class]
    module.add_class('BridgeChannel', import_from_module='ns.bridge', parent=root_module['ns3::Channel'])
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['bool', 'const ns3::WifiTxVector &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['bool', 'ns3::Ptr<ns3::NetDevice>', 'ns3::Ptr<const ns3::Packet>', 'unsigned short', 'const ns3::Address &', 'const ns3::Address &', 'ns3::NetDevice::PacketType', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['bool', 'ns3::Ptr<ns3::NetDevice>', 'ns3::Ptr<const ns3::Packet>', 'unsigned short', 'const ns3::Address &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['ns3::ObjectBase *', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<unsigned long long, const ns3::WifiTxVector &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['unsigned long long', 'const ns3::WifiTxVector &', 'unsigned short', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['unsigned long long', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['unsigned short', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, bool, ns3::Ptr<ns3::Packet>, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'bool', 'ns3::Ptr<ns3::Packet>', 'ns3::Mac48Address', 'ns3::Mac48Address', 'unsigned short', 'unsigned int', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, const ns3::WifiMacHeader &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'const ns3::WifiMacHeader &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Mac48Address', 'ns3::Mac48Address', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Mac48Address', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<const ns3::Packet>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<const ns3::WifiMacQueueItem>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<ns3::NetDevice>', 'ns3::Ptr<const ns3::Packet>', 'unsigned short', 'const ns3::Address &', 'const ns3::Address &', 'ns3::NetDevice::PacketType', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<ns3::NetDevice>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Time, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Time', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::WifiMacDropReason, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::WifiMacDropReason', 'ns3::Ptr<const ns3::WifiMacQueueItem>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::dot11s::RouteChange, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::dot11s::RouteChange', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiMacQueueItem>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'unsigned char', 'ns3::Ptr<const ns3::WifiMacQueueItem>', 'const ns3::WifiTxVector &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiPsdu>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'unsigned char', 'ns3::Ptr<const ns3::WifiPsdu>', 'const ns3::WifiTxVector &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, std::unordered_map<unsigned short, ns3::Ptr<ns3::WifiPsdu>, std::hash<unsigned short>, std::equal_to<unsigned short>, std::allocator<std::pair<const unsigned short, ns3::Ptr<ns3::WifiPsdu>>>> *, const std::set<ns3::Mac48Address, std::less<ns3::Mac48Address>, std::allocator<ns3::Mac48Address>> *, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'unsigned char', 'std::unordered_map<unsigned short, ns3::Ptr<ns3::WifiPsdu>, std::hash<unsigned short>, std::equal_to<unsigned short>, std::allocator<std::pair<const unsigned short, ns3::Ptr<ns3::WifiPsdu>>>> *', 'const std::set<ns3::Mac48Address, std::less<ns3::Mac48Address>, std::allocator<ns3::Mac48Address>> *', 'unsigned long long', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## dot11s-installer.h (module 'mesh'): ns3::Dot11sStack [class]
    module.add_class('Dot11sStack', parent=root_module['ns3::MeshStack'])
    ## flame-installer.h (module 'mesh'): ns3::FlameStack [class]
    module.add_class('FlameStack', parent=root_module['ns3::MeshStack'])
    ## mesh-point-device.h (module 'mesh'): ns3::MeshPointDevice [class]
    module.add_class('MeshPointDevice', parent=root_module['ns3::NetDevice'])
    module.add_container('std::list< ns3::BlockAckReqType >', 'ns3::BlockAckReqType', container_type='list')
    module.add_container('std::vector< unsigned char >', 'unsigned char', container_type='vector')
    module.add_container('std::map< std::string, ns3::LogComponent * >', ('std::basic_string< char, std::char_traits< char >, std::allocator< char > >', 'ns3::LogComponent *'), container_type='map')
    module.add_container('ns3::WifiModeList', 'ns3::WifiMode', container_type='vector')
    module.add_container('std::unordered_map< ns3::Mac48Address, ns3::WifiRemoteStation *, ns3::WifiAddressHash, std::equal_to< ns3::Mac48Address >, std::allocator< std::pair< ns3::Mac48Address const, ns3::WifiRemoteStation * > > >', ('ns3::Mac48Address', 'ns3::WifiRemoteStation *'), container_type='map')
    module.add_container('std::unordered_map< ns3::Mac48Address, ns3::WifiRemoteStationState *, ns3::WifiAddressHash, std::equal_to< ns3::Mac48Address >, std::allocator< std::pair< ns3::Mac48Address const, ns3::WifiRemoteStationState * > > >', ('ns3::Mac48Address', 'ns3::WifiRemoteStationState *'), container_type='map')
    module.add_container('std::unordered_map< unsigned short, ns3::Ptr< ns3::WifiPsdu > >', ('short unsigned int', 'ns3::Ptr< ns3::WifiPsdu >'), container_type='map')
    module.add_container('std::set< ns3::Mac48Address >', 'ns3::Mac48Address', container_type='set')
    module.add_container('std::vector< ns3::Ptr< ns3::NetDevice > >', 'ns3::Ptr< ns3::NetDevice >', container_type='vector')
    typehandlers.add_type_alias('std::unordered_map< unsigned short, ns3::Ptr< ns3::WifiPsdu > >', 'ns3::WifiPsduMap')
    typehandlers.add_type_alias('std::unordered_map< unsigned short, ns3::Ptr< ns3::WifiPsdu > >*', 'ns3::WifiPsduMap*')
    typehandlers.add_type_alias('std::unordered_map< unsigned short, ns3::Ptr< ns3::WifiPsdu > >&', 'ns3::WifiPsduMap&')
    typehandlers.add_type_alias('std::vector< ns3::WifiMode >', 'ns3::WifiModeList')
    typehandlers.add_type_alias('std::vector< ns3::WifiMode >*', 'ns3::WifiModeList*')
    typehandlers.add_type_alias('std::vector< ns3::WifiMode >&', 'ns3::WifiModeList&')
    typehandlers.add_type_alias('std::vector< ns3::WifiMode > const_iterator', 'ns3::WifiModeListIterator')
    typehandlers.add_type_alias('std::vector< ns3::WifiMode > const_iterator*', 'ns3::WifiModeListIterator*')
    typehandlers.add_type_alias('std::vector< ns3::WifiMode > const_iterator&', 'ns3::WifiModeListIterator&')
    typehandlers.add_type_alias('uint16_t', 'ns3::WifiCodeRate')
    typehandlers.add_type_alias('uint16_t*', 'ns3::WifiCodeRate*')
    typehandlers.add_type_alias('uint16_t&', 'ns3::WifiCodeRate&')
    typehandlers.add_type_alias('uint8_t', 'ns3::WifiInformationElementId')
    typehandlers.add_type_alias('uint8_t*', 'ns3::WifiInformationElementId*')
    typehandlers.add_type_alias('uint8_t&', 'ns3::WifiInformationElementId&')
    typehandlers.add_type_alias('std::pair< ns3::Mac48Address, unsigned char >', 'ns3::WifiAddressTidPair')
    typehandlers.add_type_alias('std::pair< ns3::Mac48Address, unsigned char >*', 'ns3::WifiAddressTidPair*')
    typehandlers.add_type_alias('std::pair< ns3::Mac48Address, unsigned char >&', 'ns3::WifiAddressTidPair&')
    typehandlers.add_type_alias('void ( * ) ( std::ostream & )', 'ns3::TimePrinter')
    typehandlers.add_type_alias('void ( * ) ( std::ostream & )*', 'ns3::TimePrinter*')
    typehandlers.add_type_alias('void ( * ) ( std::ostream & )&', 'ns3::TimePrinter&')
    typehandlers.add_type_alias('void ( * ) ( std::ostream & )', 'ns3::NodePrinter')
    typehandlers.add_type_alias('void ( * ) ( std::ostream & )*', 'ns3::NodePrinter*')
    typehandlers.add_type_alias('void ( * ) ( std::ostream & )&', 'ns3::NodePrinter&')
    
    ## Register a nested module for the namespace FatalImpl
    
    nested_module = module.add_cpp_namespace('FatalImpl')
    register_types_ns3_FatalImpl(nested_module)
    
    
    ## Register a nested module for the namespace Hash
    
    nested_module = module.add_cpp_namespace('Hash')
    register_types_ns3_Hash(nested_module)
    
    
    ## Register a nested module for the namespace TracedValueCallback
    
    nested_module = module.add_cpp_namespace('TracedValueCallback')
    register_types_ns3_TracedValueCallback(nested_module)
    
    
    ## Register a nested module for the namespace dot11s
    
    nested_module = module.add_cpp_namespace('dot11s')
    register_types_ns3_dot11s(nested_module)
    
    
    ## Register a nested module for the namespace flame
    
    nested_module = module.add_cpp_namespace('flame')
    register_types_ns3_flame(nested_module)
    
    
    ## Register a nested module for the namespace internal
    
    nested_module = module.add_cpp_namespace('internal')
    register_types_ns3_internal(nested_module)
    

def register_types_ns3_FatalImpl(module):
    root_module = module.get_root()
    

def register_types_ns3_Hash(module):
    root_module = module.get_root()
    
    ## hash-function.h (module 'core'): ns3::Hash::Implementation [class]
    module.add_class('Implementation', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::Hash::Implementation, ns3::empty, ns3::DefaultDeleter<ns3::Hash::Implementation> >'])
    typehandlers.add_type_alias('uint32_t ( * ) ( char const *, std::size_t const )', 'ns3::Hash::Hash32Function_ptr')
    typehandlers.add_type_alias('uint32_t ( * ) ( char const *, std::size_t const )*', 'ns3::Hash::Hash32Function_ptr*')
    typehandlers.add_type_alias('uint32_t ( * ) ( char const *, std::size_t const )&', 'ns3::Hash::Hash32Function_ptr&')
    typehandlers.add_type_alias('uint64_t ( * ) ( char const *, std::size_t const )', 'ns3::Hash::Hash64Function_ptr')
    typehandlers.add_type_alias('uint64_t ( * ) ( char const *, std::size_t const )*', 'ns3::Hash::Hash64Function_ptr*')
    typehandlers.add_type_alias('uint64_t ( * ) ( char const *, std::size_t const )&', 'ns3::Hash::Hash64Function_ptr&')
    
    ## Register a nested module for the namespace Function
    
    nested_module = module.add_cpp_namespace('Function')
    register_types_ns3_Hash_Function(nested_module)
    

def register_types_ns3_Hash_Function(module):
    root_module = module.get_root()
    
    ## hash-fnv.h (module 'core'): ns3::Hash::Function::Fnv1a [class]
    module.add_class('Fnv1a', import_from_module='ns.core', parent=root_module['ns3::Hash::Implementation'])
    ## hash-function.h (module 'core'): ns3::Hash::Function::Hash32 [class]
    module.add_class('Hash32', import_from_module='ns.core', parent=root_module['ns3::Hash::Implementation'])
    ## hash-function.h (module 'core'): ns3::Hash::Function::Hash64 [class]
    module.add_class('Hash64', import_from_module='ns.core', parent=root_module['ns3::Hash::Implementation'])
    ## hash-murmur3.h (module 'core'): ns3::Hash::Function::Murmur3 [class]
    module.add_class('Murmur3', import_from_module='ns.core', parent=root_module['ns3::Hash::Implementation'])

def register_types_ns3_TracedValueCallback(module):
    root_module = module.get_root()
    
    typehandlers.add_type_alias('void ( * ) ( ns3::Time, ns3::Time )', 'ns3::TracedValueCallback::Time')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time, ns3::Time )*', 'ns3::TracedValueCallback::Time*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time, ns3::Time )&', 'ns3::TracedValueCallback::Time&')
    typehandlers.add_type_alias('void ( * ) ( ns3::DataRate, ns3::DataRate )', 'ns3::TracedValueCallback::DataRate')
    typehandlers.add_type_alias('void ( * ) ( ns3::DataRate, ns3::DataRate )*', 'ns3::TracedValueCallback::DataRate*')
    typehandlers.add_type_alias('void ( * ) ( ns3::DataRate, ns3::DataRate )&', 'ns3::TracedValueCallback::DataRate&')
    typehandlers.add_type_alias('void ( * ) ( bool, bool )', 'ns3::TracedValueCallback::Bool')
    typehandlers.add_type_alias('void ( * ) ( bool, bool )*', 'ns3::TracedValueCallback::Bool*')
    typehandlers.add_type_alias('void ( * ) ( bool, bool )&', 'ns3::TracedValueCallback::Bool&')
    typehandlers.add_type_alias('void ( * ) ( int8_t, int8_t )', 'ns3::TracedValueCallback::Int8')
    typehandlers.add_type_alias('void ( * ) ( int8_t, int8_t )*', 'ns3::TracedValueCallback::Int8*')
    typehandlers.add_type_alias('void ( * ) ( int8_t, int8_t )&', 'ns3::TracedValueCallback::Int8&')
    typehandlers.add_type_alias('void ( * ) ( uint8_t, uint8_t )', 'ns3::TracedValueCallback::Uint8')
    typehandlers.add_type_alias('void ( * ) ( uint8_t, uint8_t )*', 'ns3::TracedValueCallback::Uint8*')
    typehandlers.add_type_alias('void ( * ) ( uint8_t, uint8_t )&', 'ns3::TracedValueCallback::Uint8&')
    typehandlers.add_type_alias('void ( * ) ( int16_t, int16_t )', 'ns3::TracedValueCallback::Int16')
    typehandlers.add_type_alias('void ( * ) ( int16_t, int16_t )*', 'ns3::TracedValueCallback::Int16*')
    typehandlers.add_type_alias('void ( * ) ( int16_t, int16_t )&', 'ns3::TracedValueCallback::Int16&')
    typehandlers.add_type_alias('void ( * ) ( uint16_t, uint16_t )', 'ns3::TracedValueCallback::Uint16')
    typehandlers.add_type_alias('void ( * ) ( uint16_t, uint16_t )*', 'ns3::TracedValueCallback::Uint16*')
    typehandlers.add_type_alias('void ( * ) ( uint16_t, uint16_t )&', 'ns3::TracedValueCallback::Uint16&')
    typehandlers.add_type_alias('void ( * ) ( int32_t, int32_t )', 'ns3::TracedValueCallback::Int32')
    typehandlers.add_type_alias('void ( * ) ( int32_t, int32_t )*', 'ns3::TracedValueCallback::Int32*')
    typehandlers.add_type_alias('void ( * ) ( int32_t, int32_t )&', 'ns3::TracedValueCallback::Int32&')
    typehandlers.add_type_alias('void ( * ) ( uint32_t, uint32_t )', 'ns3::TracedValueCallback::Uint32')
    typehandlers.add_type_alias('void ( * ) ( uint32_t, uint32_t )*', 'ns3::TracedValueCallback::Uint32*')
    typehandlers.add_type_alias('void ( * ) ( uint32_t, uint32_t )&', 'ns3::TracedValueCallback::Uint32&')
    typehandlers.add_type_alias('void ( * ) ( int64_t, int64_t )', 'ns3::TracedValueCallback::Int64')
    typehandlers.add_type_alias('void ( * ) ( int64_t, int64_t )*', 'ns3::TracedValueCallback::Int64*')
    typehandlers.add_type_alias('void ( * ) ( int64_t, int64_t )&', 'ns3::TracedValueCallback::Int64&')
    typehandlers.add_type_alias('void ( * ) ( uint64_t, uint64_t )', 'ns3::TracedValueCallback::Uint64')
    typehandlers.add_type_alias('void ( * ) ( uint64_t, uint64_t )*', 'ns3::TracedValueCallback::Uint64*')
    typehandlers.add_type_alias('void ( * ) ( uint64_t, uint64_t )&', 'ns3::TracedValueCallback::Uint64&')
    typehandlers.add_type_alias('void ( * ) ( double, double )', 'ns3::TracedValueCallback::Double')
    typehandlers.add_type_alias('void ( * ) ( double, double )*', 'ns3::TracedValueCallback::Double*')
    typehandlers.add_type_alias('void ( * ) ( double, double )&', 'ns3::TracedValueCallback::Double&')
    typehandlers.add_type_alias('void ( * ) (  )', 'ns3::TracedValueCallback::Void')
    typehandlers.add_type_alias('void ( * ) (  )*', 'ns3::TracedValueCallback::Void*')
    typehandlers.add_type_alias('void ( * ) (  )&', 'ns3::TracedValueCallback::Void&')

def register_types_ns3_dot11s(module):
    root_module = module.get_root()
    
    ## ie-dot11s-peer-management.h (module 'mesh'): ns3::dot11s::PmpReasonCode [enumeration]
    module.add_enum('PmpReasonCode', ['REASON11S_PEERING_CANCELLED', 'REASON11S_MESH_MAX_PEERS', 'REASON11S_MESH_CAPABILITY_POLICY_VIOLATION', 'REASON11S_MESH_CLOSE_RCVD', 'REASON11S_MESH_MAX_RETRIES', 'REASON11S_MESH_CONFIRM_TIMEOUT', 'REASON11S_MESH_INVALID_GTK', 'REASON11S_MESH_INCONSISTENT_PARAMETERS', 'REASON11S_MESH_INVALID_SECURITY_CAPABILITY', 'REASON11S_RESERVED'])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::dot11sPathSelectionProtocol [enumeration]
    module.add_enum('dot11sPathSelectionProtocol', ['PROTOCOL_HWMP'])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::dot11sPathSelectionMetric [enumeration]
    module.add_enum('dot11sPathSelectionMetric', ['METRIC_AIRTIME'])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::dot11sCongestionControlMode [enumeration]
    module.add_enum('dot11sCongestionControlMode', ['CONGESTION_SIGNALING', 'CONGESTION_NULL'])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::dot11sSynchronizationProtocolIdentifier [enumeration]
    module.add_enum('dot11sSynchronizationProtocolIdentifier', ['SYNC_NEIGHBOUR_OFFSET', 'SYNC_NULL'])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::dot11sAuthenticationProtocol [enumeration]
    module.add_enum('dot11sAuthenticationProtocol', ['AUTH_NULL', 'AUTH_SAE', 'AUTH_IEEE'])
    ## ie-dot11s-preq.h (module 'mesh'): ns3::dot11s::DestinationAddressUnit [class]
    module.add_class('DestinationAddressUnit', parent=root_module['ns3::SimpleRefCount< ns3::dot11s::DestinationAddressUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit> >'])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability [class]
    module.add_class('Dot11sMeshCapability')
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::HwmpProtocol [class]
    module.add_class('HwmpProtocol', parent=root_module['ns3::MeshL2RoutingProtocol'])
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::HwmpProtocol::FailedDestination [struct]
    module.add_class('FailedDestination', outer_class=root_module['ns3::dot11s::HwmpProtocol'])
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable [class]
    module.add_class('HwmpRtable', parent=root_module['ns3::Object'])
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult [struct]
    module.add_class('LookupResult', outer_class=root_module['ns3::dot11s::HwmpRtable'])
    typehandlers.add_type_alias('std::vector< std::pair< unsigned int, ns3::Mac48Address > >', 'ns3::dot11s::HwmpRtable::PrecursorList')
    typehandlers.add_type_alias('std::vector< std::pair< unsigned int, ns3::Mac48Address > >*', 'ns3::dot11s::HwmpRtable::PrecursorList*')
    typehandlers.add_type_alias('std::vector< std::pair< unsigned int, ns3::Mac48Address > >&', 'ns3::dot11s::HwmpRtable::PrecursorList&')
    ## ie-dot11s-beacon-timing.h (module 'mesh'): ns3::dot11s::IeBeaconTiming [class]
    module.add_class('IeBeaconTiming', parent=root_module['ns3::WifiInformationElement'])
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::dot11s::IeBeaconTimingUnit > >', 'ns3::dot11s::IeBeaconTiming::NeighboursTimingUnitsList')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::dot11s::IeBeaconTimingUnit > >*', 'ns3::dot11s::IeBeaconTiming::NeighboursTimingUnitsList*')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::dot11s::IeBeaconTimingUnit > >&', 'ns3::dot11s::IeBeaconTiming::NeighboursTimingUnitsList&')
    ## ie-dot11s-beacon-timing.h (module 'mesh'): ns3::dot11s::IeBeaconTimingUnit [class]
    module.add_class('IeBeaconTimingUnit', parent=root_module['ns3::SimpleRefCount< ns3::dot11s::IeBeaconTimingUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit> >'])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::IeConfiguration [class]
    module.add_class('IeConfiguration', parent=root_module['ns3::WifiInformationElement'])
    ## ie-dot11s-metric-report.h (module 'mesh'): ns3::dot11s::IeLinkMetricReport [class]
    module.add_class('IeLinkMetricReport', parent=root_module['ns3::WifiInformationElement'])
    ## ie-dot11s-id.h (module 'mesh'): ns3::dot11s::IeMeshId [class]
    module.add_class('IeMeshId', parent=root_module['ns3::WifiInformationElement'])
    ## ie-dot11s-peer-management.h (module 'mesh'): ns3::dot11s::IePeerManagement [class]
    module.add_class('IePeerManagement', parent=root_module['ns3::WifiInformationElement'])
    ## ie-dot11s-peer-management.h (module 'mesh'): ns3::dot11s::IePeerManagement::Subtype [enumeration]
    module.add_enum('Subtype', ['PEER_OPEN', 'PEER_CONFIRM', 'PEER_CLOSE'], outer_class=root_module['ns3::dot11s::IePeerManagement'])
    ## ie-dot11s-peering-protocol.h (module 'mesh'): ns3::dot11s::IePeeringProtocol [class]
    module.add_class('IePeeringProtocol', parent=root_module['ns3::WifiInformationElement'])
    ## ie-dot11s-perr.h (module 'mesh'): ns3::dot11s::IePerr [class]
    module.add_class('IePerr', parent=root_module['ns3::WifiInformationElement'])
    ## ie-dot11s-prep.h (module 'mesh'): ns3::dot11s::IePrep [class]
    module.add_class('IePrep', parent=root_module['ns3::WifiInformationElement'])
    ## ie-dot11s-preq.h (module 'mesh'): ns3::dot11s::IePreq [class]
    module.add_class('IePreq', parent=root_module['ns3::WifiInformationElement'])
    ## ie-dot11s-rann.h (module 'mesh'): ns3::dot11s::IeRann [class]
    module.add_class('IeRann', parent=root_module['ns3::WifiInformationElement'])
    ## dot11s-mac-header.h (module 'mesh'): ns3::dot11s::MeshHeader [class]
    module.add_class('MeshHeader', parent=root_module['ns3::Header'])
    ## peer-link.h (module 'mesh'): ns3::dot11s::PeerLink [class]
    module.add_class('PeerLink', parent=root_module['ns3::Object'])
    ## peer-link.h (module 'mesh'): ns3::dot11s::PeerLink::PeerState [enumeration]
    module.add_enum('PeerState', ['IDLE', 'OPN_SNT', 'CNF_RCVD', 'OPN_RCVD', 'ESTAB', 'HOLDING'], outer_class=root_module['ns3::dot11s::PeerLink'])
    typehandlers.add_type_alias('ns3::Callback< void, unsigned int, ns3::Mac48Address, ns3::Mac48Address, ns3::dot11s::PeerLink::PeerState, ns3::dot11s::PeerLink::PeerState, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::dot11s::PeerLink::SignalStatusCallback')
    typehandlers.add_type_alias('ns3::Callback< void, unsigned int, ns3::Mac48Address, ns3::Mac48Address, ns3::dot11s::PeerLink::PeerState, ns3::dot11s::PeerLink::PeerState, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::dot11s::PeerLink::SignalStatusCallback*')
    typehandlers.add_type_alias('ns3::Callback< void, unsigned int, ns3::Mac48Address, ns3::Mac48Address, ns3::dot11s::PeerLink::PeerState, ns3::dot11s::PeerLink::PeerState, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::dot11s::PeerLink::SignalStatusCallback&')
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkCloseStart [class]
    module.add_class('PeerLinkCloseStart', parent=root_module['ns3::Header'])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields [struct]
    module.add_class('PlinkCloseStartFields', outer_class=root_module['ns3::dot11s::PeerLinkCloseStart'])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart [class]
    module.add_class('PeerLinkConfirmStart', parent=root_module['ns3::Header'])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields [struct]
    module.add_class('PlinkConfirmStartFields', outer_class=root_module['ns3::dot11s::PeerLinkConfirmStart'])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart [class]
    module.add_class('PeerLinkOpenStart', parent=root_module['ns3::Header'])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields [struct]
    module.add_class('PlinkOpenStartFields', outer_class=root_module['ns3::dot11s::PeerLinkOpenStart'])
    ## peer-management-protocol.h (module 'mesh'): ns3::dot11s::PeerManagementProtocol [class]
    module.add_class('PeerManagementProtocol', parent=root_module['ns3::Object'])
    typehandlers.add_type_alias('void ( * ) ( ns3::Mac48Address, ns3::Mac48Address const )', 'ns3::dot11s::PeerManagementProtocol::LinkOpenCloseTracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Mac48Address, ns3::Mac48Address const )*', 'ns3::dot11s::PeerManagementProtocol::LinkOpenCloseTracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Mac48Address, ns3::Mac48Address const )&', 'ns3::dot11s::PeerManagementProtocol::LinkOpenCloseTracedCallback&')
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::RouteChange [struct]
    module.add_class('RouteChange')
    module.add_container('std::vector< std::pair< unsigned int, ns3::Mac48Address > >', 'std::pair< unsigned int, ns3::Mac48Address >', container_type='vector')
    module.add_container('ns3::dot11s::HwmpRtable::PrecursorList', 'std::pair< unsigned int, ns3::Mac48Address >', container_type='vector')
    module.add_container('std::vector< ns3::dot11s::HwmpProtocol::FailedDestination >', 'ns3::dot11s::HwmpProtocol::FailedDestination', container_type='vector')
    module.add_container('std::vector< ns3::Ptr< ns3::dot11s::IeBeaconTimingUnit > >', 'ns3::Ptr< ns3::dot11s::IeBeaconTimingUnit >', container_type='vector')
    module.add_container('ns3::dot11s::IeBeaconTiming::NeighboursTimingUnitsList', 'ns3::Ptr< ns3::dot11s::IeBeaconTimingUnit >', container_type='vector')
    module.add_container('std::vector< ns3::Ptr< ns3::dot11s::DestinationAddressUnit > >', 'ns3::Ptr< ns3::dot11s::DestinationAddressUnit >', container_type='vector')
    module.add_container('std::vector< ns3::Ptr< ns3::dot11s::PeerLink > >', 'ns3::Ptr< ns3::dot11s::PeerLink >', container_type='vector')
    module.add_container('std::vector< ns3::Mac48Address >', 'ns3::Mac48Address', container_type='vector')

def register_types_ns3_flame(module):
    root_module = module.get_root()
    
    ## flame-header.h (module 'mesh'): ns3::flame::FlameHeader [class]
    module.add_class('FlameHeader', parent=root_module['ns3::Header'])
    ## flame-protocol.h (module 'mesh'): ns3::flame::FlameProtocol [class]
    module.add_class('FlameProtocol', parent=root_module['ns3::MeshL2RoutingProtocol'])
    ## flame-protocol-mac.h (module 'mesh'): ns3::flame::FlameProtocolMac [class]
    module.add_class('FlameProtocolMac', parent=root_module['ns3::MeshWifiInterfaceMacPlugin'])
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable [class]
    module.add_class('FlameRtable', parent=root_module['ns3::Object'])
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::LookupResult [struct]
    module.add_class('LookupResult', outer_class=root_module['ns3::flame::FlameRtable'])
    ## flame-protocol.h (module 'mesh'): ns3::flame::FlameTag [class]
    module.add_class('FlameTag', parent=root_module['ns3::Tag'])

def register_types_ns3_internal(module):
    root_module = module.get_root()
    

def register_methods(root_module):
    register_Ns3Address_methods(root_module, root_module['ns3::Address'])
    register_Ns3AttributeConstructionList_methods(root_module, root_module['ns3::AttributeConstructionList'])
    register_Ns3AttributeConstructionListItem_methods(root_module, root_module['ns3::AttributeConstructionList::Item'])
    register_Ns3BlockAckReqType_methods(root_module, root_module['ns3::BlockAckReqType'])
    register_Ns3BlockAckType_methods(root_module, root_module['ns3::BlockAckType'])
    register_Ns3Buffer_methods(root_module, root_module['ns3::Buffer'])
    register_Ns3BufferIterator_methods(root_module, root_module['ns3::Buffer::Iterator'])
    register_Ns3ByteTagIterator_methods(root_module, root_module['ns3::ByteTagIterator'])
    register_Ns3ByteTagIteratorItem_methods(root_module, root_module['ns3::ByteTagIterator::Item'])
    register_Ns3ByteTagList_methods(root_module, root_module['ns3::ByteTagList'])
    register_Ns3ByteTagListIterator_methods(root_module, root_module['ns3::ByteTagList::Iterator'])
    register_Ns3ByteTagListIteratorItem_methods(root_module, root_module['ns3::ByteTagList::Iterator::Item'])
    register_Ns3CallbackBase_methods(root_module, root_module['ns3::CallbackBase'])
    register_Ns3CapabilityInformation_methods(root_module, root_module['ns3::CapabilityInformation'])
    register_Ns3DataRate_methods(root_module, root_module['ns3::DataRate'])
    register_Ns3DefaultDeleter__Ns3AttributeAccessor_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeAccessor >'])
    register_Ns3DefaultDeleter__Ns3AttributeChecker_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeChecker >'])
    register_Ns3DefaultDeleter__Ns3AttributeValue_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeValue >'])
    register_Ns3DefaultDeleter__Ns3CallbackImplBase_methods(root_module, root_module['ns3::DefaultDeleter< ns3::CallbackImplBase >'])
    register_Ns3DefaultDeleter__Ns3EventImpl_methods(root_module, root_module['ns3::DefaultDeleter< ns3::EventImpl >'])
    register_Ns3DefaultDeleter__Ns3HashImplementation_methods(root_module, root_module['ns3::DefaultDeleter< ns3::Hash::Implementation >'])
    register_Ns3DefaultDeleter__Ns3MeshWifiInterfaceMacPlugin_methods(root_module, root_module['ns3::DefaultDeleter< ns3::MeshWifiInterfaceMacPlugin >'])
    register_Ns3DefaultDeleter__Ns3NixVector_methods(root_module, root_module['ns3::DefaultDeleter< ns3::NixVector >'])
    register_Ns3DefaultDeleter__Ns3Packet_methods(root_module, root_module['ns3::DefaultDeleter< ns3::Packet >'])
    register_Ns3DefaultDeleter__Ns3TraceSourceAccessor_methods(root_module, root_module['ns3::DefaultDeleter< ns3::TraceSourceAccessor >'])
    register_Ns3DefaultDeleter__Ns3WifiInformationElement_methods(root_module, root_module['ns3::DefaultDeleter< ns3::WifiInformationElement >'])
    register_Ns3DefaultDeleter__Ns3Dot11sDestinationAddressUnit_methods(root_module, root_module['ns3::DefaultDeleter< ns3::dot11s::DestinationAddressUnit >'])
    register_Ns3DefaultDeleter__Ns3Dot11sIeBeaconTimingUnit_methods(root_module, root_module['ns3::DefaultDeleter< ns3::dot11s::IeBeaconTimingUnit >'])
    register_Ns3EventId_methods(root_module, root_module['ns3::EventId'])
    register_Ns3Hasher_methods(root_module, root_module['ns3::Hasher'])
    register_Ns3Ipv4Address_methods(root_module, root_module['ns3::Ipv4Address'])
    register_Ns3Ipv4AddressHash_methods(root_module, root_module['ns3::Ipv4AddressHash'])
    register_Ns3Ipv4Mask_methods(root_module, root_module['ns3::Ipv4Mask'])
    register_Ns3Ipv6Address_methods(root_module, root_module['ns3::Ipv6Address'])
    register_Ns3Ipv6AddressHash_methods(root_module, root_module['ns3::Ipv6AddressHash'])
    register_Ns3Ipv6Prefix_methods(root_module, root_module['ns3::Ipv6Prefix'])
    register_Ns3LogComponent_methods(root_module, root_module['ns3::LogComponent'])
    register_Ns3Mac48Address_methods(root_module, root_module['ns3::Mac48Address'])
    register_Ns3Mac8Address_methods(root_module, root_module['ns3::Mac8Address'])
    register_Ns3MeshHelper_methods(root_module, root_module['ns3::MeshHelper'])
    register_Ns3MeshWifiBeacon_methods(root_module, root_module['ns3::MeshWifiBeacon'])
    register_Ns3ObjectBase_methods(root_module, root_module['ns3::ObjectBase'])
    register_Ns3ObjectDeleter_methods(root_module, root_module['ns3::ObjectDeleter'])
    register_Ns3ObjectFactory_methods(root_module, root_module['ns3::ObjectFactory'])
    register_Ns3PacketMetadata_methods(root_module, root_module['ns3::PacketMetadata'])
    register_Ns3PacketMetadataItem_methods(root_module, root_module['ns3::PacketMetadata::Item'])
    register_Ns3PacketMetadataItemIterator_methods(root_module, root_module['ns3::PacketMetadata::ItemIterator'])
    register_Ns3PacketTagIterator_methods(root_module, root_module['ns3::PacketTagIterator'])
    register_Ns3PacketTagIteratorItem_methods(root_module, root_module['ns3::PacketTagIterator::Item'])
    register_Ns3PacketTagList_methods(root_module, root_module['ns3::PacketTagList'])
    register_Ns3PacketTagListTagData_methods(root_module, root_module['ns3::PacketTagList::TagData'])
    register_Ns3ParameterLogger_methods(root_module, root_module['ns3::ParameterLogger'])
    register_Ns3SimpleRefCount__Ns3Object_Ns3ObjectBase_Ns3ObjectDeleter_methods(root_module, root_module['ns3::SimpleRefCount< ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter >'])
    register_Ns3StatusCode_methods(root_module, root_module['ns3::StatusCode'])
    register_Ns3Tag_methods(root_module, root_module['ns3::Tag'])
    register_Ns3TagBuffer_methods(root_module, root_module['ns3::TagBuffer'])
    register_Ns3Time_methods(root_module, root_module['ns3::Time'])
    register_Ns3TimeWithUnit_methods(root_module, root_module['ns3::TimeWithUnit'])
    register_Ns3TypeId_methods(root_module, root_module['ns3::TypeId'])
    register_Ns3TypeIdAttributeInformation_methods(root_module, root_module['ns3::TypeId::AttributeInformation'])
    register_Ns3TypeIdTraceSourceInformation_methods(root_module, root_module['ns3::TypeId::TraceSourceInformation'])
    register_Ns3WifiAc_methods(root_module, root_module['ns3::WifiAc'])
    register_Ns3WifiAddressHash_methods(root_module, root_module['ns3::WifiAddressHash'])
    register_Ns3WifiAddressTidHash_methods(root_module, root_module['ns3::WifiAddressTidHash'])
    register_Ns3WifiMode_methods(root_module, root_module['ns3::WifiMode'])
    register_Ns3WifiModeFactory_methods(root_module, root_module['ns3::WifiModeFactory'])
    register_Ns3WifiRemoteStation_methods(root_module, root_module['ns3::WifiRemoteStation'])
    register_Ns3WifiRemoteStationInfo_methods(root_module, root_module['ns3::WifiRemoteStationInfo'])
    register_Ns3WifiRemoteStationState_methods(root_module, root_module['ns3::WifiRemoteStationState'])
    register_Ns3Empty_methods(root_module, root_module['ns3::empty'])
    register_Ns3Int64x64_t_methods(root_module, root_module['ns3::int64x64_t'])
    register_Ns3Chunk_methods(root_module, root_module['ns3::Chunk'])
    register_Ns3Header_methods(root_module, root_module['ns3::Header'])
    register_Ns3MgtAddBaRequestHeader_methods(root_module, root_module['ns3::MgtAddBaRequestHeader'])
    register_Ns3MgtAddBaResponseHeader_methods(root_module, root_module['ns3::MgtAddBaResponseHeader'])
    register_Ns3MgtAssocRequestHeader_methods(root_module, root_module['ns3::MgtAssocRequestHeader'])
    register_Ns3MgtAssocResponseHeader_methods(root_module, root_module['ns3::MgtAssocResponseHeader'])
    register_Ns3MgtDelBaHeader_methods(root_module, root_module['ns3::MgtDelBaHeader'])
    register_Ns3MgtProbeRequestHeader_methods(root_module, root_module['ns3::MgtProbeRequestHeader'])
    register_Ns3MgtProbeResponseHeader_methods(root_module, root_module['ns3::MgtProbeResponseHeader'])
    register_Ns3MgtReassocRequestHeader_methods(root_module, root_module['ns3::MgtReassocRequestHeader'])
    register_Ns3Object_methods(root_module, root_module['ns3::Object'])
    register_Ns3ObjectAggregateIterator_methods(root_module, root_module['ns3::Object::AggregateIterator'])
    register_Ns3RandomVariableStream_methods(root_module, root_module['ns3::RandomVariableStream'])
    register_Ns3SequentialRandomVariable_methods(root_module, root_module['ns3::SequentialRandomVariable'])
    register_Ns3SimpleRefCount__Ns3AttributeAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeAccessor__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeAccessor, ns3::empty, ns3::DefaultDeleter<ns3::AttributeAccessor> >'])
    register_Ns3SimpleRefCount__Ns3AttributeChecker_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeChecker__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeChecker, ns3::empty, ns3::DefaultDeleter<ns3::AttributeChecker> >'])
    register_Ns3SimpleRefCount__Ns3AttributeValue_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeValue__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeValue, ns3::empty, ns3::DefaultDeleter<ns3::AttributeValue> >'])
    register_Ns3SimpleRefCount__Ns3CallbackImplBase_Ns3Empty_Ns3DefaultDeleter__lt__ns3CallbackImplBase__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter<ns3::CallbackImplBase> >'])
    register_Ns3SimpleRefCount__Ns3EventImpl_Ns3Empty_Ns3DefaultDeleter__lt__ns3EventImpl__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl> >'])
    register_Ns3SimpleRefCount__Ns3HashImplementation_Ns3Empty_Ns3DefaultDeleter__lt__ns3HashImplementation__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::Hash::Implementation, ns3::empty, ns3::DefaultDeleter<ns3::Hash::Implementation> >'])
    register_Ns3SimpleRefCount__Ns3MeshWifiInterfaceMacPlugin_Ns3Empty_Ns3DefaultDeleter__lt__ns3MeshWifiInterfaceMacPlugin__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::MeshWifiInterfaceMacPlugin, ns3::empty, ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin> >'])
    register_Ns3SimpleRefCount__Ns3NixVector_Ns3Empty_Ns3DefaultDeleter__lt__ns3NixVector__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector> >'])
    register_Ns3SimpleRefCount__Ns3Packet_Ns3Empty_Ns3DefaultDeleter__lt__ns3Packet__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet> >'])
    register_Ns3SimpleRefCount__Ns3TraceSourceAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3TraceSourceAccessor__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor> >'])
    register_Ns3SimpleRefCount__Ns3WifiInformationElement_Ns3Empty_Ns3DefaultDeleter__lt__ns3WifiInformationElement__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::WifiInformationElement, ns3::empty, ns3::DefaultDeleter<ns3::WifiInformationElement> >'])
    register_Ns3SimpleRefCount__Ns3Dot11sDestinationAddressUnit_Ns3Empty_Ns3DefaultDeleter__lt__ns3Dot11sDestinationAddressUnit__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::dot11s::DestinationAddressUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit> >'])
    register_Ns3SimpleRefCount__Ns3Dot11sIeBeaconTimingUnit_Ns3Empty_Ns3DefaultDeleter__lt__ns3Dot11sIeBeaconTimingUnit__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::dot11s::IeBeaconTimingUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit> >'])
    register_Ns3TraceSourceAccessor_methods(root_module, root_module['ns3::TraceSourceAccessor'])
    register_Ns3Trailer_methods(root_module, root_module['ns3::Trailer'])
    register_Ns3TriangularRandomVariable_methods(root_module, root_module['ns3::TriangularRandomVariable'])
    register_Ns3UniformRandomVariable_methods(root_module, root_module['ns3::UniformRandomVariable'])
    register_Ns3WeibullRandomVariable_methods(root_module, root_module['ns3::WeibullRandomVariable'])
    register_Ns3WifiActionHeader_methods(root_module, root_module['ns3::WifiActionHeader'])
    register_Ns3WifiActionHeaderActionValue_methods(root_module, root_module['ns3::WifiActionHeader::ActionValue'])
    register_Ns3WifiInformationElement_methods(root_module, root_module['ns3::WifiInformationElement'])
    register_Ns3WifiInformationElementVector_methods(root_module, root_module['ns3::WifiInformationElementVector'])
    register_Ns3WifiMac_methods(root_module, root_module['ns3::WifiMac'])
    register_Ns3WifiRemoteStationManager_methods(root_module, root_module['ns3::WifiRemoteStationManager'])
    register_Ns3ZetaRandomVariable_methods(root_module, root_module['ns3::ZetaRandomVariable'])
    register_Ns3ZipfRandomVariable_methods(root_module, root_module['ns3::ZipfRandomVariable'])
    register_Ns3AttributeAccessor_methods(root_module, root_module['ns3::AttributeAccessor'])
    register_Ns3AttributeChecker_methods(root_module, root_module['ns3::AttributeChecker'])
    register_Ns3AttributeValue_methods(root_module, root_module['ns3::AttributeValue'])
    register_Ns3BooleanChecker_methods(root_module, root_module['ns3::BooleanChecker'])
    register_Ns3BooleanValue_methods(root_module, root_module['ns3::BooleanValue'])
    register_Ns3CallbackChecker_methods(root_module, root_module['ns3::CallbackChecker'])
    register_Ns3CallbackImplBase_methods(root_module, root_module['ns3::CallbackImplBase'])
    register_Ns3CallbackValue_methods(root_module, root_module['ns3::CallbackValue'])
    register_Ns3Channel_methods(root_module, root_module['ns3::Channel'])
    register_Ns3ConstantRandomVariable_methods(root_module, root_module['ns3::ConstantRandomVariable'])
    register_Ns3DataRateChecker_methods(root_module, root_module['ns3::DataRateChecker'])
    register_Ns3DataRateValue_methods(root_module, root_module['ns3::DataRateValue'])
    register_Ns3DeterministicRandomVariable_methods(root_module, root_module['ns3::DeterministicRandomVariable'])
    register_Ns3DoubleValue_methods(root_module, root_module['ns3::DoubleValue'])
    register_Ns3DsssParameterSet_methods(root_module, root_module['ns3::DsssParameterSet'])
    register_Ns3EdcaParameterSet_methods(root_module, root_module['ns3::EdcaParameterSet'])
    register_Ns3EmpiricalRandomVariable_methods(root_module, root_module['ns3::EmpiricalRandomVariable'])
    register_Ns3EmptyAttributeAccessor_methods(root_module, root_module['ns3::EmptyAttributeAccessor'])
    register_Ns3EmptyAttributeChecker_methods(root_module, root_module['ns3::EmptyAttributeChecker'])
    register_Ns3EmptyAttributeValue_methods(root_module, root_module['ns3::EmptyAttributeValue'])
    register_Ns3EnumChecker_methods(root_module, root_module['ns3::EnumChecker'])
    register_Ns3EnumValue_methods(root_module, root_module['ns3::EnumValue'])
    register_Ns3ErlangRandomVariable_methods(root_module, root_module['ns3::ErlangRandomVariable'])
    register_Ns3ErpInformation_methods(root_module, root_module['ns3::ErpInformation'])
    register_Ns3EventImpl_methods(root_module, root_module['ns3::EventImpl'])
    register_Ns3ExponentialRandomVariable_methods(root_module, root_module['ns3::ExponentialRandomVariable'])
    register_Ns3ExtendedCapabilities_methods(root_module, root_module['ns3::ExtendedCapabilities'])
    register_Ns3ExtendedSupportedRatesIE_methods(root_module, root_module['ns3::ExtendedSupportedRatesIE'])
    register_Ns3GammaRandomVariable_methods(root_module, root_module['ns3::GammaRandomVariable'])
    register_Ns3HeCapabilities_methods(root_module, root_module['ns3::HeCapabilities'])
    register_Ns3HeOperation_methods(root_module, root_module['ns3::HeOperation'])
    register_Ns3HtCapabilities_methods(root_module, root_module['ns3::HtCapabilities'])
    register_Ns3HtOperation_methods(root_module, root_module['ns3::HtOperation'])
    register_Ns3IntegerValue_methods(root_module, root_module['ns3::IntegerValue'])
    register_Ns3Ipv4AddressChecker_methods(root_module, root_module['ns3::Ipv4AddressChecker'])
    register_Ns3Ipv4AddressValue_methods(root_module, root_module['ns3::Ipv4AddressValue'])
    register_Ns3Ipv4MaskChecker_methods(root_module, root_module['ns3::Ipv4MaskChecker'])
    register_Ns3Ipv4MaskValue_methods(root_module, root_module['ns3::Ipv4MaskValue'])
    register_Ns3Ipv6AddressChecker_methods(root_module, root_module['ns3::Ipv6AddressChecker'])
    register_Ns3Ipv6AddressValue_methods(root_module, root_module['ns3::Ipv6AddressValue'])
    register_Ns3Ipv6PrefixChecker_methods(root_module, root_module['ns3::Ipv6PrefixChecker'])
    register_Ns3Ipv6PrefixValue_methods(root_module, root_module['ns3::Ipv6PrefixValue'])
    register_Ns3LogNormalRandomVariable_methods(root_module, root_module['ns3::LogNormalRandomVariable'])
    register_Ns3Mac48AddressChecker_methods(root_module, root_module['ns3::Mac48AddressChecker'])
    register_Ns3Mac48AddressValue_methods(root_module, root_module['ns3::Mac48AddressValue'])
    register_Ns3MeshInformationElementVector_methods(root_module, root_module['ns3::MeshInformationElementVector'])
    register_Ns3MeshL2RoutingProtocol_methods(root_module, root_module['ns3::MeshL2RoutingProtocol'])
    register_Ns3MeshStack_methods(root_module, root_module['ns3::MeshStack'])
    register_Ns3MeshWifiInterfaceMac_methods(root_module, root_module['ns3::MeshWifiInterfaceMac'])
    register_Ns3MeshWifiInterfaceMacPlugin_methods(root_module, root_module['ns3::MeshWifiInterfaceMacPlugin'])
    register_Ns3MgtBeaconHeader_methods(root_module, root_module['ns3::MgtBeaconHeader'])
    register_Ns3MuEdcaParameterSet_methods(root_module, root_module['ns3::MuEdcaParameterSet'])
    register_Ns3NetDevice_methods(root_module, root_module['ns3::NetDevice'])
    register_Ns3NixVector_methods(root_module, root_module['ns3::NixVector'])
    register_Ns3Node_methods(root_module, root_module['ns3::Node'])
    register_Ns3NormalRandomVariable_methods(root_module, root_module['ns3::NormalRandomVariable'])
    register_Ns3ObjectFactoryChecker_methods(root_module, root_module['ns3::ObjectFactoryChecker'])
    register_Ns3ObjectFactoryValue_methods(root_module, root_module['ns3::ObjectFactoryValue'])
    register_Ns3Packet_methods(root_module, root_module['ns3::Packet'])
    register_Ns3ParetoRandomVariable_methods(root_module, root_module['ns3::ParetoRandomVariable'])
    register_Ns3Ssid_methods(root_module, root_module['ns3::Ssid'])
    register_Ns3SsidChecker_methods(root_module, root_module['ns3::SsidChecker'])
    register_Ns3SsidValue_methods(root_module, root_module['ns3::SsidValue'])
    register_Ns3SupportedRates_methods(root_module, root_module['ns3::SupportedRates'])
    register_Ns3TimeValue_methods(root_module, root_module['ns3::TimeValue'])
    register_Ns3TypeIdChecker_methods(root_module, root_module['ns3::TypeIdChecker'])
    register_Ns3TypeIdValue_methods(root_module, root_module['ns3::TypeIdValue'])
    register_Ns3UintegerValue_methods(root_module, root_module['ns3::UintegerValue'])
    register_Ns3VhtCapabilities_methods(root_module, root_module['ns3::VhtCapabilities'])
    register_Ns3VhtOperation_methods(root_module, root_module['ns3::VhtOperation'])
    register_Ns3WifiModeChecker_methods(root_module, root_module['ns3::WifiModeChecker'])
    register_Ns3WifiModeValue_methods(root_module, root_module['ns3::WifiModeValue'])
    register_Ns3AddressChecker_methods(root_module, root_module['ns3::AddressChecker'])
    register_Ns3AddressValue_methods(root_module, root_module['ns3::AddressValue'])
    register_Ns3BridgeChannel_methods(root_module, root_module['ns3::BridgeChannel'])
    register_Ns3CallbackImpl__Bool_Const_ns3WifiTxVector___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< bool, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Bool_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Ptr__lt__const_ns3Packet__gt___Unsigned_short_Const_ns3Address___amp___Const_ns3Address___amp___Ns3NetDevicePacketType_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Bool_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Ptr__lt__const_ns3Packet__gt___Unsigned_short_Const_ns3Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Ns3ObjectBase___star___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Unsigned_long_Const_ns3WifiTxVector___amp___Unsigned_short_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< unsigned long long, const ns3::WifiTxVector &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Unsigned_long_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Unsigned_short_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Bool_Ns3Ptr__lt__ns3Packet__gt___Ns3Mac48Address_Ns3Mac48Address_Unsigned_short_Unsigned_int_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, bool, ns3::Ptr<ns3::Packet>, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Const_ns3WifiMacHeader___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, const ns3::WifiMacHeader &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Mac48Address_Ns3Mac48Address_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Mac48Address_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3WifiMacQueueItem__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Ptr__lt__const_ns3Packet__gt___Unsigned_short_Const_ns3Address___amp___Const_ns3Address___amp___Ns3NetDevicePacketType_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Time_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Time, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3WifiMacDropReason_Ns3Ptr__lt__const_ns3WifiMacQueueItem__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::WifiMacDropReason, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Dot11sRouteChange_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::dot11s::RouteChange, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Unsigned_char_Ns3Ptr__lt__const_ns3WifiMacQueueItem__gt___Const_ns3WifiTxVector___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, unsigned char, ns3::Ptr<const ns3::WifiMacQueueItem>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Unsigned_char_Ns3Ptr__lt__const_ns3WifiPsdu__gt___Const_ns3WifiTxVector___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, unsigned char, ns3::Ptr<const ns3::WifiPsdu>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Unsigned_char_StdUnordered_map__lt__unsigned_short__ns3Ptr__lt__ns3WifiPsdu__gt____stdHash__lt__unsigned_short__gt____stdEqual_to__lt__unsigned_short__gt____stdAllocator__lt__stdPair__lt__const_unsigned_short__ns3Ptr__lt__ns3WifiPsdu__gt____gt____gt____gt_____star___Const_stdSet__lt__ns3Mac48Address__stdLess__lt__ns3Mac48Address__gt____stdAllocator__lt__ns3Mac48Address__gt____gt_____star___Unsigned_long_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, unsigned char, std::unordered_map<unsigned short, ns3::Ptr<ns3::WifiPsdu>, std::hash<unsigned short>, std::equal_to<unsigned short>, std::allocator<std::pair<const unsigned short, ns3::Ptr<ns3::WifiPsdu>>>> *, const std::set<ns3::Mac48Address, std::less<ns3::Mac48Address>, std::allocator<ns3::Mac48Address>> *, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3Dot11sStack_methods(root_module, root_module['ns3::Dot11sStack'])
    register_Ns3FlameStack_methods(root_module, root_module['ns3::FlameStack'])
    register_Ns3MeshPointDevice_methods(root_module, root_module['ns3::MeshPointDevice'])
    register_Ns3HashImplementation_methods(root_module, root_module['ns3::Hash::Implementation'])
    register_Ns3HashFunctionFnv1a_methods(root_module, root_module['ns3::Hash::Function::Fnv1a'])
    register_Ns3HashFunctionHash32_methods(root_module, root_module['ns3::Hash::Function::Hash32'])
    register_Ns3HashFunctionHash64_methods(root_module, root_module['ns3::Hash::Function::Hash64'])
    register_Ns3HashFunctionMurmur3_methods(root_module, root_module['ns3::Hash::Function::Murmur3'])
    register_Ns3Dot11sDestinationAddressUnit_methods(root_module, root_module['ns3::dot11s::DestinationAddressUnit'])
    register_Ns3Dot11sDot11sMeshCapability_methods(root_module, root_module['ns3::dot11s::Dot11sMeshCapability'])
    register_Ns3Dot11sHwmpProtocol_methods(root_module, root_module['ns3::dot11s::HwmpProtocol'])
    register_Ns3Dot11sHwmpProtocolFailedDestination_methods(root_module, root_module['ns3::dot11s::HwmpProtocol::FailedDestination'])
    register_Ns3Dot11sHwmpRtable_methods(root_module, root_module['ns3::dot11s::HwmpRtable'])
    register_Ns3Dot11sHwmpRtableLookupResult_methods(root_module, root_module['ns3::dot11s::HwmpRtable::LookupResult'])
    register_Ns3Dot11sIeBeaconTiming_methods(root_module, root_module['ns3::dot11s::IeBeaconTiming'])
    register_Ns3Dot11sIeBeaconTimingUnit_methods(root_module, root_module['ns3::dot11s::IeBeaconTimingUnit'])
    register_Ns3Dot11sIeConfiguration_methods(root_module, root_module['ns3::dot11s::IeConfiguration'])
    register_Ns3Dot11sIeLinkMetricReport_methods(root_module, root_module['ns3::dot11s::IeLinkMetricReport'])
    register_Ns3Dot11sIeMeshId_methods(root_module, root_module['ns3::dot11s::IeMeshId'])
    register_Ns3Dot11sIePeerManagement_methods(root_module, root_module['ns3::dot11s::IePeerManagement'])
    register_Ns3Dot11sIePeeringProtocol_methods(root_module, root_module['ns3::dot11s::IePeeringProtocol'])
    register_Ns3Dot11sIePerr_methods(root_module, root_module['ns3::dot11s::IePerr'])
    register_Ns3Dot11sIePrep_methods(root_module, root_module['ns3::dot11s::IePrep'])
    register_Ns3Dot11sIePreq_methods(root_module, root_module['ns3::dot11s::IePreq'])
    register_Ns3Dot11sIeRann_methods(root_module, root_module['ns3::dot11s::IeRann'])
    register_Ns3Dot11sMeshHeader_methods(root_module, root_module['ns3::dot11s::MeshHeader'])
    register_Ns3Dot11sPeerLink_methods(root_module, root_module['ns3::dot11s::PeerLink'])
    register_Ns3Dot11sPeerLinkCloseStart_methods(root_module, root_module['ns3::dot11s::PeerLinkCloseStart'])
    register_Ns3Dot11sPeerLinkCloseStartPlinkCloseStartFields_methods(root_module, root_module['ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields'])
    register_Ns3Dot11sPeerLinkConfirmStart_methods(root_module, root_module['ns3::dot11s::PeerLinkConfirmStart'])
    register_Ns3Dot11sPeerLinkConfirmStartPlinkConfirmStartFields_methods(root_module, root_module['ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields'])
    register_Ns3Dot11sPeerLinkOpenStart_methods(root_module, root_module['ns3::dot11s::PeerLinkOpenStart'])
    register_Ns3Dot11sPeerLinkOpenStartPlinkOpenStartFields_methods(root_module, root_module['ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields'])
    register_Ns3Dot11sPeerManagementProtocol_methods(root_module, root_module['ns3::dot11s::PeerManagementProtocol'])
    register_Ns3Dot11sRouteChange_methods(root_module, root_module['ns3::dot11s::RouteChange'])
    register_Ns3FlameFlameHeader_methods(root_module, root_module['ns3::flame::FlameHeader'])
    register_Ns3FlameFlameProtocol_methods(root_module, root_module['ns3::flame::FlameProtocol'])
    register_Ns3FlameFlameProtocolMac_methods(root_module, root_module['ns3::flame::FlameProtocolMac'])
    register_Ns3FlameFlameRtable_methods(root_module, root_module['ns3::flame::FlameRtable'])
    register_Ns3FlameFlameRtableLookupResult_methods(root_module, root_module['ns3::flame::FlameRtable::LookupResult'])
    register_Ns3FlameFlameTag_methods(root_module, root_module['ns3::flame::FlameTag'])
    return

def register_Ns3Address_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<')
    cls.add_output_stream_operator()
    ## address.h (module 'network'): ns3::Address::Address() [constructor]
    cls.add_constructor([])
    ## address.h (module 'network'): ns3::Address::Address(uint8_t type, uint8_t const * buffer, uint8_t len) [constructor]
    cls.add_constructor([param('uint8_t', 'type'), param('uint8_t const *', 'buffer'), param('uint8_t', 'len')])
    ## address.h (module 'network'): ns3::Address::Address(ns3::Address const & address) [constructor]
    cls.add_constructor([param('ns3::Address const &', 'address')])
    ## address.h (module 'network'): bool ns3::Address::CheckCompatible(uint8_t type, uint8_t len) const [member function]
    cls.add_method('CheckCompatible', 
                   'bool', 
                   [param('uint8_t', 'type'), param('uint8_t', 'len')], 
                   is_const=True)
    ## address.h (module 'network'): uint32_t ns3::Address::CopyAllFrom(uint8_t const * buffer, uint8_t len) [member function]
    cls.add_method('CopyAllFrom', 
                   'uint32_t', 
                   [param('uint8_t const *', 'buffer'), param('uint8_t', 'len')])
    ## address.h (module 'network'): uint32_t ns3::Address::CopyAllTo(uint8_t * buffer, uint8_t len) const [member function]
    cls.add_method('CopyAllTo', 
                   'uint32_t', 
                   [param('uint8_t *', 'buffer'), param('uint8_t', 'len')], 
                   is_const=True)
    ## address.h (module 'network'): uint32_t ns3::Address::CopyFrom(uint8_t const * buffer, uint8_t len) [member function]
    cls.add_method('CopyFrom', 
                   'uint32_t', 
                   [param('uint8_t const *', 'buffer'), param('uint8_t', 'len')])
    ## address.h (module 'network'): uint32_t ns3::Address::CopyTo(uint8_t * buffer) const [member function]
    cls.add_method('CopyTo', 
                   'uint32_t', 
                   [param('uint8_t *', 'buffer')], 
                   is_const=True)
    ## address.h (module 'network'): void ns3::Address::Deserialize(ns3::TagBuffer buffer) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'buffer')])
    ## address.h (module 'network'): uint8_t ns3::Address::GetLength() const [member function]
    cls.add_method('GetLength', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## address.h (module 'network'): uint32_t ns3::Address::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## address.h (module 'network'): bool ns3::Address::IsInvalid() const [member function]
    cls.add_method('IsInvalid', 
                   'bool', 
                   [], 
                   is_const=True)
    ## address.h (module 'network'): bool ns3::Address::IsMatchingType(uint8_t type) const [member function]
    cls.add_method('IsMatchingType', 
                   'bool', 
                   [param('uint8_t', 'type')], 
                   is_const=True)
    ## address.h (module 'network'): static uint8_t ns3::Address::Register() [member function]
    cls.add_method('Register', 
                   'uint8_t', 
                   [], 
                   is_static=True)
    ## address.h (module 'network'): void ns3::Address::Serialize(ns3::TagBuffer buffer) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'buffer')], 
                   is_const=True)
    return

def register_Ns3AttributeConstructionList_methods(root_module, cls):
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::AttributeConstructionList(ns3::AttributeConstructionList const & arg0) [constructor]
    cls.add_constructor([param('ns3::AttributeConstructionList const &', 'arg0')])
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::AttributeConstructionList() [constructor]
    cls.add_constructor([])
    ## attribute-construction-list.h (module 'core'): void ns3::AttributeConstructionList::Add(std::string name, ns3::Ptr<const ns3::AttributeChecker> checker, ns3::Ptr<ns3::AttributeValue> value) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('std::string', 'name'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker'), param('ns3::Ptr< ns3::AttributeValue >', 'value')])
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::CIterator ns3::AttributeConstructionList::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::AttributeConstructionList::CIterator', 
                   [], 
                   is_const=True)
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::CIterator ns3::AttributeConstructionList::End() const [member function]
    cls.add_method('End', 
                   'ns3::AttributeConstructionList::CIterator', 
                   [], 
                   is_const=True)
    ## attribute-construction-list.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::AttributeConstructionList::Find(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('Find', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True)
    return

def register_Ns3AttributeConstructionListItem_methods(root_module, cls):
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::Item::Item() [constructor]
    cls.add_constructor([])
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::Item::Item(ns3::AttributeConstructionList::Item const & arg0) [constructor]
    cls.add_constructor([param('ns3::AttributeConstructionList::Item const &', 'arg0')])
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::Item::checker [variable]
    cls.add_instance_attribute('checker', 'ns3::Ptr< ns3::AttributeChecker const >', is_const=False)
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::Item::name [variable]
    cls.add_instance_attribute('name', 'std::string', is_const=False)
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::Item::value [variable]
    cls.add_instance_attribute('value', 'ns3::Ptr< ns3::AttributeValue >', is_const=False)
    return

def register_Ns3BlockAckReqType_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckReqType::BlockAckReqType(ns3::BlockAckReqType const & arg0) [constructor]
    cls.add_constructor([param('ns3::BlockAckReqType const &', 'arg0')])
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckReqType::BlockAckReqType() [constructor]
    cls.add_constructor([])
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckReqType::BlockAckReqType(ns3::BlockAckReqType::Variant v) [constructor]
    cls.add_constructor([param('ns3::BlockAckReqType::Variant', 'v')])
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckReqType::BlockAckReqType(ns3::BlockAckReqType::Variant v, uint8_t nSeqControls) [constructor]
    cls.add_constructor([param('ns3::BlockAckReqType::Variant', 'v'), param('uint8_t', 'nSeqControls')])
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckReqType::m_nSeqControls [variable]
    cls.add_instance_attribute('m_nSeqControls', 'uint8_t', is_const=False)
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckReqType::m_variant [variable]
    cls.add_instance_attribute('m_variant', 'ns3::BlockAckReqType::Variant', is_const=False)
    return

def register_Ns3BlockAckType_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckType::BlockAckType(ns3::BlockAckType const & arg0) [constructor]
    cls.add_constructor([param('ns3::BlockAckType const &', 'arg0')])
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckType::BlockAckType() [constructor]
    cls.add_constructor([])
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckType::BlockAckType(ns3::BlockAckType::Variant v) [constructor]
    cls.add_constructor([param('ns3::BlockAckType::Variant', 'v')])
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckType::BlockAckType(ns3::BlockAckType::Variant v, std::vector<unsigned char, std::allocator<unsigned char>> l) [constructor]
    cls.add_constructor([param('ns3::BlockAckType::Variant', 'v'), param('std::vector< unsigned char >', 'l')])
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckType::m_bitmapLen [variable]
    cls.add_instance_attribute('m_bitmapLen', 'std::vector< unsigned char >', is_const=False)
    ## block-ack-type.h (module 'wifi'): ns3::BlockAckType::m_variant [variable]
    cls.add_instance_attribute('m_variant', 'ns3::BlockAckType::Variant', is_const=False)
    return

def register_Ns3Buffer_methods(root_module, cls):
    ## buffer.h (module 'network'): ns3::Buffer::Buffer(ns3::Buffer const & o) [constructor]
    cls.add_constructor([param('ns3::Buffer const &', 'o')])
    ## buffer.h (module 'network'): ns3::Buffer::Buffer() [constructor]
    cls.add_constructor([])
    ## buffer.h (module 'network'): ns3::Buffer::Buffer(uint32_t dataSize) [constructor]
    cls.add_constructor([param('uint32_t', 'dataSize')])
    ## buffer.h (module 'network'): ns3::Buffer::Buffer(uint32_t dataSize, bool initialize) [constructor]
    cls.add_constructor([param('uint32_t', 'dataSize'), param('bool', 'initialize')])
    ## buffer.h (module 'network'): void ns3::Buffer::AddAtEnd(uint32_t end) [member function]
    cls.add_method('AddAtEnd', 
                   'void', 
                   [param('uint32_t', 'end')])
    ## buffer.h (module 'network'): void ns3::Buffer::AddAtEnd(ns3::Buffer const & o) [member function]
    cls.add_method('AddAtEnd', 
                   'void', 
                   [param('ns3::Buffer const &', 'o')])
    ## buffer.h (module 'network'): void ns3::Buffer::AddAtStart(uint32_t start) [member function]
    cls.add_method('AddAtStart', 
                   'void', 
                   [param('uint32_t', 'start')])
    ## buffer.h (module 'network'): ns3::Buffer::Iterator ns3::Buffer::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::Buffer::Iterator', 
                   [], 
                   is_const=True)
    ## buffer.h (module 'network'): void ns3::Buffer::CopyData(std::ostream * os, uint32_t size) const [member function]
    cls.add_method('CopyData', 
                   'void', 
                   [param('std::ostream *', 'os'), param('uint32_t', 'size')], 
                   is_const=True)
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::CopyData(uint8_t * buffer, uint32_t size) const [member function]
    cls.add_method('CopyData', 
                   'uint32_t', 
                   [param('uint8_t *', 'buffer'), param('uint32_t', 'size')], 
                   is_const=True)
    ## buffer.h (module 'network'): ns3::Buffer ns3::Buffer::CreateFragment(uint32_t start, uint32_t length) const [member function]
    cls.add_method('CreateFragment', 
                   'ns3::Buffer', 
                   [param('uint32_t', 'start'), param('uint32_t', 'length')], 
                   is_const=True)
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::Deserialize(uint8_t const * buffer, uint32_t size) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('uint8_t const *', 'buffer'), param('uint32_t', 'size')])
    ## buffer.h (module 'network'): ns3::Buffer::Iterator ns3::Buffer::End() const [member function]
    cls.add_method('End', 
                   'ns3::Buffer::Iterator', 
                   [], 
                   is_const=True)
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## buffer.h (module 'network'): uint8_t const * ns3::Buffer::PeekData() const [member function]
    cls.add_method('PeekData', 
                   'uint8_t const *', 
                   [], 
                   is_const=True)
    ## buffer.h (module 'network'): void ns3::Buffer::RemoveAtEnd(uint32_t end) [member function]
    cls.add_method('RemoveAtEnd', 
                   'void', 
                   [param('uint32_t', 'end')])
    ## buffer.h (module 'network'): void ns3::Buffer::RemoveAtStart(uint32_t start) [member function]
    cls.add_method('RemoveAtStart', 
                   'void', 
                   [param('uint32_t', 'start')])
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::Serialize(uint8_t * buffer, uint32_t maxSize) const [member function]
    cls.add_method('Serialize', 
                   'uint32_t', 
                   [param('uint8_t *', 'buffer'), param('uint32_t', 'maxSize')], 
                   is_const=True)
    return

def register_Ns3BufferIterator_methods(root_module, cls):
    ## buffer.h (module 'network'): ns3::Buffer::Iterator::Iterator(ns3::Buffer::Iterator const & arg0) [constructor]
    cls.add_constructor([param('ns3::Buffer::Iterator const &', 'arg0')])
    ## buffer.h (module 'network'): ns3::Buffer::Iterator::Iterator() [constructor]
    cls.add_constructor([])
    ## buffer.h (module 'network'): uint16_t ns3::Buffer::Iterator::CalculateIpChecksum(uint16_t size) [member function]
    cls.add_method('CalculateIpChecksum', 
                   'uint16_t', 
                   [param('uint16_t', 'size')])
    ## buffer.h (module 'network'): uint16_t ns3::Buffer::Iterator::CalculateIpChecksum(uint16_t size, uint32_t initialChecksum) [member function]
    cls.add_method('CalculateIpChecksum', 
                   'uint16_t', 
                   [param('uint16_t', 'size'), param('uint32_t', 'initialChecksum')])
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::Iterator::GetDistanceFrom(ns3::Buffer::Iterator const & o) const [member function]
    cls.add_method('GetDistanceFrom', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator const &', 'o')], 
                   is_const=True)
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::Iterator::GetRemainingSize() const [member function]
    cls.add_method('GetRemainingSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::Iterator::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## buffer.h (module 'network'): bool ns3::Buffer::Iterator::IsEnd() const [member function]
    cls.add_method('IsEnd', 
                   'bool', 
                   [], 
                   is_const=True)
    ## buffer.h (module 'network'): bool ns3::Buffer::Iterator::IsStart() const [member function]
    cls.add_method('IsStart', 
                   'bool', 
                   [], 
                   is_const=True)
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::Next() [member function]
    cls.add_method('Next', 
                   'void', 
                   [])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::Next(uint32_t delta) [member function]
    cls.add_method('Next', 
                   'void', 
                   [param('uint32_t', 'delta')])
    ## buffer.h (module 'network'): uint8_t ns3::Buffer::Iterator::PeekU8() [member function]
    cls.add_method('PeekU8', 
                   'uint8_t', 
                   [])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::Prev() [member function]
    cls.add_method('Prev', 
                   'void', 
                   [])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::Prev(uint32_t delta) [member function]
    cls.add_method('Prev', 
                   'void', 
                   [param('uint32_t', 'delta')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::Read(uint8_t * buffer, uint32_t size) [member function]
    cls.add_method('Read', 
                   'void', 
                   [param('uint8_t *', 'buffer'), param('uint32_t', 'size')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::Read(ns3::Buffer::Iterator start, uint32_t size) [member function]
    cls.add_method('Read', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint32_t', 'size')])
    ## buffer.h (module 'network'): uint16_t ns3::Buffer::Iterator::ReadLsbtohU16() [member function]
    cls.add_method('ReadLsbtohU16', 
                   'uint16_t', 
                   [])
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::Iterator::ReadLsbtohU32() [member function]
    cls.add_method('ReadLsbtohU32', 
                   'uint32_t', 
                   [])
    ## buffer.h (module 'network'): uint64_t ns3::Buffer::Iterator::ReadLsbtohU64() [member function]
    cls.add_method('ReadLsbtohU64', 
                   'uint64_t', 
                   [])
    ## buffer.h (module 'network'): uint16_t ns3::Buffer::Iterator::ReadNtohU16() [member function]
    cls.add_method('ReadNtohU16', 
                   'uint16_t', 
                   [])
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::Iterator::ReadNtohU32() [member function]
    cls.add_method('ReadNtohU32', 
                   'uint32_t', 
                   [])
    ## buffer.h (module 'network'): uint64_t ns3::Buffer::Iterator::ReadNtohU64() [member function]
    cls.add_method('ReadNtohU64', 
                   'uint64_t', 
                   [])
    ## buffer.h (module 'network'): uint16_t ns3::Buffer::Iterator::ReadU16() [member function]
    cls.add_method('ReadU16', 
                   'uint16_t', 
                   [])
    ## buffer.h (module 'network'): uint32_t ns3::Buffer::Iterator::ReadU32() [member function]
    cls.add_method('ReadU32', 
                   'uint32_t', 
                   [])
    ## buffer.h (module 'network'): uint64_t ns3::Buffer::Iterator::ReadU64() [member function]
    cls.add_method('ReadU64', 
                   'uint64_t', 
                   [])
    ## buffer.h (module 'network'): uint8_t ns3::Buffer::Iterator::ReadU8() [member function]
    cls.add_method('ReadU8', 
                   'uint8_t', 
                   [])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::Write(uint8_t const * buffer, uint32_t size) [member function]
    cls.add_method('Write', 
                   'void', 
                   [param('uint8_t const *', 'buffer'), param('uint32_t', 'size')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::Write(ns3::Buffer::Iterator start, ns3::Buffer::Iterator end) [member function]
    cls.add_method('Write', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start'), param('ns3::Buffer::Iterator', 'end')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteHtolsbU16(uint16_t data) [member function]
    cls.add_method('WriteHtolsbU16', 
                   'void', 
                   [param('uint16_t', 'data')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteHtolsbU32(uint32_t data) [member function]
    cls.add_method('WriteHtolsbU32', 
                   'void', 
                   [param('uint32_t', 'data')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteHtolsbU64(uint64_t data) [member function]
    cls.add_method('WriteHtolsbU64', 
                   'void', 
                   [param('uint64_t', 'data')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteHtonU16(uint16_t data) [member function]
    cls.add_method('WriteHtonU16', 
                   'void', 
                   [param('uint16_t', 'data')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteHtonU32(uint32_t data) [member function]
    cls.add_method('WriteHtonU32', 
                   'void', 
                   [param('uint32_t', 'data')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteHtonU64(uint64_t data) [member function]
    cls.add_method('WriteHtonU64', 
                   'void', 
                   [param('uint64_t', 'data')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteU16(uint16_t data) [member function]
    cls.add_method('WriteU16', 
                   'void', 
                   [param('uint16_t', 'data')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteU32(uint32_t data) [member function]
    cls.add_method('WriteU32', 
                   'void', 
                   [param('uint32_t', 'data')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteU64(uint64_t data) [member function]
    cls.add_method('WriteU64', 
                   'void', 
                   [param('uint64_t', 'data')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteU8(uint8_t data) [member function]
    cls.add_method('WriteU8', 
                   'void', 
                   [param('uint8_t', 'data')])
    ## buffer.h (module 'network'): void ns3::Buffer::Iterator::WriteU8(uint8_t data, uint32_t len) [member function]
    cls.add_method('WriteU8', 
                   'void', 
                   [param('uint8_t', 'data'), param('uint32_t', 'len')])
    return

def register_Ns3ByteTagIterator_methods(root_module, cls):
    ## packet.h (module 'network'): ns3::ByteTagIterator::ByteTagIterator(ns3::ByteTagIterator const & arg0) [constructor]
    cls.add_constructor([param('ns3::ByteTagIterator const &', 'arg0')])
    ## packet.h (module 'network'): bool ns3::ByteTagIterator::HasNext() const [member function]
    cls.add_method('HasNext', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): ns3::ByteTagIterator::Item ns3::ByteTagIterator::Next() [member function]
    cls.add_method('Next', 
                   'ns3::ByteTagIterator::Item', 
                   [])
    return

def register_Ns3ByteTagIteratorItem_methods(root_module, cls):
    ## packet.h (module 'network'): ns3::ByteTagIterator::Item::Item(ns3::ByteTagIterator::Item const & arg0) [constructor]
    cls.add_constructor([param('ns3::ByteTagIterator::Item const &', 'arg0')])
    ## packet.h (module 'network'): uint32_t ns3::ByteTagIterator::Item::GetEnd() const [member function]
    cls.add_method('GetEnd', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): uint32_t ns3::ByteTagIterator::Item::GetStart() const [member function]
    cls.add_method('GetStart', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): void ns3::ByteTagIterator::Item::GetTag(ns3::Tag & tag) const [member function]
    cls.add_method('GetTag', 
                   'void', 
                   [param('ns3::Tag &', 'tag')], 
                   is_const=True)
    ## packet.h (module 'network'): ns3::TypeId ns3::ByteTagIterator::Item::GetTypeId() const [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True)
    return

def register_Ns3ByteTagList_methods(root_module, cls):
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::ByteTagList() [constructor]
    cls.add_constructor([])
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::ByteTagList(ns3::ByteTagList const & o) [constructor]
    cls.add_constructor([param('ns3::ByteTagList const &', 'o')])
    ## byte-tag-list.h (module 'network'): ns3::TagBuffer ns3::ByteTagList::Add(ns3::TypeId tid, uint32_t bufferSize, int32_t start, int32_t end) [member function]
    cls.add_method('Add', 
                   'ns3::TagBuffer', 
                   [param('ns3::TypeId', 'tid'), param('uint32_t', 'bufferSize'), param('int32_t', 'start'), param('int32_t', 'end')])
    ## byte-tag-list.h (module 'network'): void ns3::ByteTagList::Add(ns3::ByteTagList const & o) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::ByteTagList const &', 'o')])
    ## byte-tag-list.h (module 'network'): void ns3::ByteTagList::AddAtEnd(int32_t appendOffset) [member function]
    cls.add_method('AddAtEnd', 
                   'void', 
                   [param('int32_t', 'appendOffset')])
    ## byte-tag-list.h (module 'network'): void ns3::ByteTagList::AddAtStart(int32_t prependOffset) [member function]
    cls.add_method('AddAtStart', 
                   'void', 
                   [param('int32_t', 'prependOffset')])
    ## byte-tag-list.h (module 'network'): void ns3::ByteTagList::Adjust(int32_t adjustment) [member function]
    cls.add_method('Adjust', 
                   'void', 
                   [param('int32_t', 'adjustment')])
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator ns3::ByteTagList::Begin(int32_t offsetStart, int32_t offsetEnd) const [member function]
    cls.add_method('Begin', 
                   'ns3::ByteTagList::Iterator', 
                   [param('int32_t', 'offsetStart'), param('int32_t', 'offsetEnd')], 
                   is_const=True)
    ## byte-tag-list.h (module 'network'): uint32_t ns3::ByteTagList::Deserialize(uint32_t const * buffer, uint32_t size) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('uint32_t const *', 'buffer'), param('uint32_t', 'size')])
    ## byte-tag-list.h (module 'network'): uint32_t ns3::ByteTagList::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## byte-tag-list.h (module 'network'): void ns3::ByteTagList::RemoveAll() [member function]
    cls.add_method('RemoveAll', 
                   'void', 
                   [])
    ## byte-tag-list.h (module 'network'): uint32_t ns3::ByteTagList::Serialize(uint32_t * buffer, uint32_t maxSize) const [member function]
    cls.add_method('Serialize', 
                   'uint32_t', 
                   [param('uint32_t *', 'buffer'), param('uint32_t', 'maxSize')], 
                   is_const=True)
    return

def register_Ns3ByteTagListIterator_methods(root_module, cls):
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Iterator(ns3::ByteTagList::Iterator const & arg0) [constructor]
    cls.add_constructor([param('ns3::ByteTagList::Iterator const &', 'arg0')])
    ## byte-tag-list.h (module 'network'): uint32_t ns3::ByteTagList::Iterator::GetOffsetStart() const [member function]
    cls.add_method('GetOffsetStart', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## byte-tag-list.h (module 'network'): bool ns3::ByteTagList::Iterator::HasNext() const [member function]
    cls.add_method('HasNext', 
                   'bool', 
                   [], 
                   is_const=True)
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Item ns3::ByteTagList::Iterator::Next() [member function]
    cls.add_method('Next', 
                   'ns3::ByteTagList::Iterator::Item', 
                   [])
    return

def register_Ns3ByteTagListIteratorItem_methods(root_module, cls):
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Item::Item(ns3::ByteTagList::Iterator::Item const & arg0) [constructor]
    cls.add_constructor([param('ns3::ByteTagList::Iterator::Item const &', 'arg0')])
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Item::Item(ns3::TagBuffer buf) [constructor]
    cls.add_constructor([param('ns3::TagBuffer', 'buf')])
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Item::buf [variable]
    cls.add_instance_attribute('buf', 'ns3::TagBuffer', is_const=False)
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Item::end [variable]
    cls.add_instance_attribute('end', 'int32_t', is_const=False)
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Item::size [variable]
    cls.add_instance_attribute('size', 'uint32_t', is_const=False)
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Item::start [variable]
    cls.add_instance_attribute('start', 'int32_t', is_const=False)
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Item::tid [variable]
    cls.add_instance_attribute('tid', 'ns3::TypeId', is_const=False)
    return

def register_Ns3CallbackBase_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackBase::CallbackBase(ns3::CallbackBase const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackBase const &', 'arg0')])
    ## callback.h (module 'core'): ns3::CallbackBase::CallbackBase() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::Ptr<ns3::CallbackImplBase> ns3::CallbackBase::GetImpl() const [member function]
    cls.add_method('GetImpl', 
                   'ns3::Ptr< ns3::CallbackImplBase >', 
                   [], 
                   is_const=True)
    ## callback.h (module 'core'): ns3::CallbackBase::CallbackBase(ns3::Ptr<ns3::CallbackImplBase> impl) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::CallbackImplBase >', 'impl')], 
                        visibility='protected')
    return

def register_Ns3CapabilityInformation_methods(root_module, cls):
    ## capability-information.h (module 'wifi'): ns3::CapabilityInformation::CapabilityInformation(ns3::CapabilityInformation const & arg0) [constructor]
    cls.add_constructor([param('ns3::CapabilityInformation const &', 'arg0')])
    ## capability-information.h (module 'wifi'): ns3::CapabilityInformation::CapabilityInformation() [constructor]
    cls.add_constructor([])
    ## capability-information.h (module 'wifi'): ns3::Buffer::Iterator ns3::CapabilityInformation::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## capability-information.h (module 'wifi'): uint32_t ns3::CapabilityInformation::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## capability-information.h (module 'wifi'): bool ns3::CapabilityInformation::IsCfPollable() const [member function]
    cls.add_method('IsCfPollable', 
                   'bool', 
                   [], 
                   is_const=True)
    ## capability-information.h (module 'wifi'): bool ns3::CapabilityInformation::IsEss() const [member function]
    cls.add_method('IsEss', 
                   'bool', 
                   [], 
                   is_const=True)
    ## capability-information.h (module 'wifi'): bool ns3::CapabilityInformation::IsIbss() const [member function]
    cls.add_method('IsIbss', 
                   'bool', 
                   [], 
                   is_const=True)
    ## capability-information.h (module 'wifi'): bool ns3::CapabilityInformation::IsShortPreamble() const [member function]
    cls.add_method('IsShortPreamble', 
                   'bool', 
                   [], 
                   is_const=True)
    ## capability-information.h (module 'wifi'): bool ns3::CapabilityInformation::IsShortSlotTime() const [member function]
    cls.add_method('IsShortSlotTime', 
                   'bool', 
                   [], 
                   is_const=True)
    ## capability-information.h (module 'wifi'): ns3::Buffer::Iterator ns3::CapabilityInformation::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True)
    ## capability-information.h (module 'wifi'): void ns3::CapabilityInformation::SetCfPollable() [member function]
    cls.add_method('SetCfPollable', 
                   'void', 
                   [])
    ## capability-information.h (module 'wifi'): void ns3::CapabilityInformation::SetEss() [member function]
    cls.add_method('SetEss', 
                   'void', 
                   [])
    ## capability-information.h (module 'wifi'): void ns3::CapabilityInformation::SetIbss() [member function]
    cls.add_method('SetIbss', 
                   'void', 
                   [])
    ## capability-information.h (module 'wifi'): void ns3::CapabilityInformation::SetShortPreamble(bool shortPreamble) [member function]
    cls.add_method('SetShortPreamble', 
                   'void', 
                   [param('bool', 'shortPreamble')])
    ## capability-information.h (module 'wifi'): void ns3::CapabilityInformation::SetShortSlotTime(bool shortSlotTime) [member function]
    cls.add_method('SetShortSlotTime', 
                   'void', 
                   [param('bool', 'shortSlotTime')])
    return

def register_Ns3DataRate_methods(root_module, cls):
    cls.add_output_stream_operator()
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_numeric_operator('*', root_module['ns3::DataRate'], root_module['ns3::DataRate'], param('double', 'right'))
    cls.add_binary_numeric_operator('*', root_module['ns3::DataRate'], root_module['ns3::DataRate'], param('uint64_t', 'right'))
    cls.add_inplace_numeric_operator('*=', param('double', 'right'))
    cls.add_inplace_numeric_operator('*=', param('uint64_t', 'right'))
    cls.add_binary_numeric_operator('+', root_module['ns3::DataRate'], root_module['ns3::DataRate'], param('ns3::DataRate', 'right'))
    cls.add_inplace_numeric_operator('+=', param('ns3::DataRate', 'right'))
    cls.add_binary_numeric_operator('-', root_module['ns3::DataRate'], root_module['ns3::DataRate'], param('ns3::DataRate', 'right'))
    cls.add_inplace_numeric_operator('-=', param('ns3::DataRate', 'right'))
    cls.add_binary_comparison_operator('<')
    cls.add_binary_comparison_operator('<=')
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('>')
    cls.add_binary_comparison_operator('>=')
    ## data-rate.h (module 'network'): ns3::DataRate::DataRate(ns3::DataRate const & arg0) [constructor]
    cls.add_constructor([param('ns3::DataRate const &', 'arg0')])
    ## data-rate.h (module 'network'): ns3::DataRate::DataRate() [constructor]
    cls.add_constructor([])
    ## data-rate.h (module 'network'): ns3::DataRate::DataRate(uint64_t bps) [constructor]
    cls.add_constructor([param('uint64_t', 'bps')])
    ## data-rate.h (module 'network'): ns3::DataRate::DataRate(std::string rate) [constructor]
    cls.add_constructor([param('std::string', 'rate')])
    ## data-rate.h (module 'network'): ns3::Time ns3::DataRate::CalculateBitsTxTime(uint32_t bits) const [member function]
    cls.add_method('CalculateBitsTxTime', 
                   'ns3::Time', 
                   [param('uint32_t', 'bits')], 
                   is_const=True)
    ## data-rate.h (module 'network'): ns3::Time ns3::DataRate::CalculateBytesTxTime(uint32_t bytes) const [member function]
    cls.add_method('CalculateBytesTxTime', 
                   'ns3::Time', 
                   [param('uint32_t', 'bytes')], 
                   is_const=True)
    ## data-rate.h (module 'network'): uint64_t ns3::DataRate::GetBitRate() const [member function]
    cls.add_method('GetBitRate', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    return

def register_Ns3DefaultDeleter__Ns3AttributeAccessor_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::AttributeAccessor>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::AttributeAccessor>::DefaultDeleter(ns3::DefaultDeleter<ns3::AttributeAccessor> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::AttributeAccessor > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::AttributeAccessor>::Delete(ns3::AttributeAccessor * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::AttributeAccessor *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3AttributeChecker_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::AttributeChecker>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::AttributeChecker>::DefaultDeleter(ns3::DefaultDeleter<ns3::AttributeChecker> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::AttributeChecker > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::AttributeChecker>::Delete(ns3::AttributeChecker * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::AttributeChecker *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3AttributeValue_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::AttributeValue>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::AttributeValue>::DefaultDeleter(ns3::DefaultDeleter<ns3::AttributeValue> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::AttributeValue > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::AttributeValue>::Delete(ns3::AttributeValue * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::AttributeValue *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3CallbackImplBase_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::CallbackImplBase>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::CallbackImplBase>::DefaultDeleter(ns3::DefaultDeleter<ns3::CallbackImplBase> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::CallbackImplBase > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::CallbackImplBase>::Delete(ns3::CallbackImplBase * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::CallbackImplBase *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3EventImpl_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::EventImpl>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::EventImpl>::DefaultDeleter(ns3::DefaultDeleter<ns3::EventImpl> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::EventImpl > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::EventImpl>::Delete(ns3::EventImpl * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::EventImpl *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3HashImplementation_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::Hash::Implementation>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::Hash::Implementation>::DefaultDeleter(ns3::DefaultDeleter<ns3::Hash::Implementation> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::Hash::Implementation > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::Hash::Implementation>::Delete(ns3::Hash::Implementation * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::Hash::Implementation *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3MeshWifiInterfaceMacPlugin_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin>::DefaultDeleter(ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::MeshWifiInterfaceMacPlugin > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin>::Delete(ns3::MeshWifiInterfaceMacPlugin * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::MeshWifiInterfaceMacPlugin *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3NixVector_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::NixVector>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::NixVector>::DefaultDeleter(ns3::DefaultDeleter<ns3::NixVector> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::NixVector > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::NixVector>::Delete(ns3::NixVector * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::NixVector *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3Packet_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::Packet>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::Packet>::DefaultDeleter(ns3::DefaultDeleter<ns3::Packet> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::Packet > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::Packet>::Delete(ns3::Packet * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::Packet *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3TraceSourceAccessor_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::TraceSourceAccessor>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::TraceSourceAccessor>::DefaultDeleter(ns3::DefaultDeleter<ns3::TraceSourceAccessor> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::TraceSourceAccessor > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::TraceSourceAccessor>::Delete(ns3::TraceSourceAccessor * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::TraceSourceAccessor *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3WifiInformationElement_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::WifiInformationElement>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::WifiInformationElement>::DefaultDeleter(ns3::DefaultDeleter<ns3::WifiInformationElement> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::WifiInformationElement > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::WifiInformationElement>::Delete(ns3::WifiInformationElement * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::WifiInformationElement *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3Dot11sDestinationAddressUnit_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit>::DefaultDeleter(ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::dot11s::DestinationAddressUnit > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit>::Delete(ns3::dot11s::DestinationAddressUnit * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::dot11s::DestinationAddressUnit *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3Dot11sIeBeaconTimingUnit_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit>::DefaultDeleter(ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::dot11s::IeBeaconTimingUnit > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit>::Delete(ns3::dot11s::IeBeaconTimingUnit * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::dot11s::IeBeaconTimingUnit *', 'object')], 
                   is_static=True)
    return

def register_Ns3EventId_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<')
    ## event-id.h (module 'core'): ns3::EventId::EventId(ns3::EventId const & arg0) [constructor]
    cls.add_constructor([param('ns3::EventId const &', 'arg0')])
    ## event-id.h (module 'core'): ns3::EventId::EventId() [constructor]
    cls.add_constructor([])
    ## event-id.h (module 'core'): ns3::EventId::EventId(ns3::Ptr<ns3::EventImpl> const & impl, uint64_t ts, uint32_t context, uint32_t uid) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::EventImpl > const &', 'impl'), param('uint64_t', 'ts'), param('uint32_t', 'context'), param('uint32_t', 'uid')])
    ## event-id.h (module 'core'): void ns3::EventId::Cancel() [member function]
    cls.add_method('Cancel', 
                   'void', 
                   [])
    ## event-id.h (module 'core'): uint32_t ns3::EventId::GetContext() const [member function]
    cls.add_method('GetContext', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## event-id.h (module 'core'): uint64_t ns3::EventId::GetTs() const [member function]
    cls.add_method('GetTs', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## event-id.h (module 'core'): uint32_t ns3::EventId::GetUid() const [member function]
    cls.add_method('GetUid', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## event-id.h (module 'core'): bool ns3::EventId::IsExpired() const [member function]
    cls.add_method('IsExpired', 
                   'bool', 
                   [], 
                   is_const=True)
    ## event-id.h (module 'core'): bool ns3::EventId::IsRunning() const [member function]
    cls.add_method('IsRunning', 
                   'bool', 
                   [], 
                   is_const=True)
    ## event-id.h (module 'core'): ns3::EventImpl * ns3::EventId::PeekEventImpl() const [member function]
    cls.add_method('PeekEventImpl', 
                   'ns3::EventImpl *', 
                   [], 
                   is_const=True)
    ## event-id.h (module 'core'): void ns3::EventId::Remove() [member function]
    cls.add_method('Remove', 
                   'void', 
                   [])
    return

def register_Ns3Hasher_methods(root_module, cls):
    ## hash.h (module 'core'): ns3::Hasher::Hasher(ns3::Hasher const & arg0) [constructor]
    cls.add_constructor([param('ns3::Hasher const &', 'arg0')])
    ## hash.h (module 'core'): ns3::Hasher::Hasher() [constructor]
    cls.add_constructor([])
    ## hash.h (module 'core'): ns3::Hasher::Hasher(ns3::Ptr<ns3::Hash::Implementation> hp) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::Hash::Implementation >', 'hp')])
    ## hash.h (module 'core'): uint32_t ns3::Hasher::GetHash32(char const * buffer, std::size_t const size) [member function]
    cls.add_method('GetHash32', 
                   'uint32_t', 
                   [param('char const *', 'buffer'), param('std::size_t const', 'size')])
    ## hash.h (module 'core'): uint32_t ns3::Hasher::GetHash32(std::string const s) [member function]
    cls.add_method('GetHash32', 
                   'uint32_t', 
                   [param('std::string const', 's')])
    ## hash.h (module 'core'): uint64_t ns3::Hasher::GetHash64(char const * buffer, std::size_t const size) [member function]
    cls.add_method('GetHash64', 
                   'uint64_t', 
                   [param('char const *', 'buffer'), param('std::size_t const', 'size')])
    ## hash.h (module 'core'): uint64_t ns3::Hasher::GetHash64(std::string const s) [member function]
    cls.add_method('GetHash64', 
                   'uint64_t', 
                   [param('std::string const', 's')])
    ## hash.h (module 'core'): ns3::Hasher & ns3::Hasher::clear() [member function]
    cls.add_method('clear', 
                   'ns3::Hasher &', 
                   [])
    return

def register_Ns3Ipv4Address_methods(root_module, cls):
    cls.add_output_stream_operator()
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<')
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address::Ipv4Address(ns3::Ipv4Address const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv4Address const &', 'arg0')])
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address::Ipv4Address() [constructor]
    cls.add_constructor([])
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address::Ipv4Address(uint32_t address) [constructor]
    cls.add_constructor([param('uint32_t', 'address')])
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address::Ipv4Address(char const * address) [constructor]
    cls.add_constructor([param('char const *', 'address')])
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address ns3::Ipv4Address::CombineMask(ns3::Ipv4Mask const & mask) const [member function]
    cls.add_method('CombineMask', 
                   'ns3::Ipv4Address', 
                   [param('ns3::Ipv4Mask const &', 'mask')], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): static ns3::Ipv4Address ns3::Ipv4Address::ConvertFrom(ns3::Address const & address) [member function]
    cls.add_method('ConvertFrom', 
                   'ns3::Ipv4Address', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## ipv4-address.h (module 'network'): static ns3::Ipv4Address ns3::Ipv4Address::Deserialize(uint8_t const * buf) [member function]
    cls.add_method('Deserialize', 
                   'ns3::Ipv4Address', 
                   [param('uint8_t const *', 'buf')], 
                   is_static=True)
    ## ipv4-address.h (module 'network'): uint32_t ns3::Ipv4Address::Get() const [member function]
    cls.add_method('Get', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): static ns3::Ipv4Address ns3::Ipv4Address::GetAny() [member function]
    cls.add_method('GetAny', 
                   'ns3::Ipv4Address', 
                   [], 
                   is_static=True)
    ## ipv4-address.h (module 'network'): static ns3::Ipv4Address ns3::Ipv4Address::GetBroadcast() [member function]
    cls.add_method('GetBroadcast', 
                   'ns3::Ipv4Address', 
                   [], 
                   is_static=True)
    ## ipv4-address.h (module 'network'): static ns3::Ipv4Address ns3::Ipv4Address::GetLoopback() [member function]
    cls.add_method('GetLoopback', 
                   'ns3::Ipv4Address', 
                   [], 
                   is_static=True)
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address ns3::Ipv4Address::GetSubnetDirectedBroadcast(ns3::Ipv4Mask const & mask) const [member function]
    cls.add_method('GetSubnetDirectedBroadcast', 
                   'ns3::Ipv4Address', 
                   [param('ns3::Ipv4Mask const &', 'mask')], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): static ns3::Ipv4Address ns3::Ipv4Address::GetZero() [member function]
    cls.add_method('GetZero', 
                   'ns3::Ipv4Address', 
                   [], 
                   is_static=True)
    ## ipv4-address.h (module 'network'): bool ns3::Ipv4Address::IsAny() const [member function]
    cls.add_method('IsAny', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): bool ns3::Ipv4Address::IsBroadcast() const [member function]
    cls.add_method('IsBroadcast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): bool ns3::Ipv4Address::IsInitialized() const [member function]
    cls.add_method('IsInitialized', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): bool ns3::Ipv4Address::IsLocalMulticast() const [member function]
    cls.add_method('IsLocalMulticast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): bool ns3::Ipv4Address::IsLocalhost() const [member function]
    cls.add_method('IsLocalhost', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): static bool ns3::Ipv4Address::IsMatchingType(ns3::Address const & address) [member function]
    cls.add_method('IsMatchingType', 
                   'bool', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## ipv4-address.h (module 'network'): bool ns3::Ipv4Address::IsMulticast() const [member function]
    cls.add_method('IsMulticast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): bool ns3::Ipv4Address::IsSubnetDirectedBroadcast(ns3::Ipv4Mask const & mask) const [member function]
    cls.add_method('IsSubnetDirectedBroadcast', 
                   'bool', 
                   [param('ns3::Ipv4Mask const &', 'mask')], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): void ns3::Ipv4Address::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): void ns3::Ipv4Address::Serialize(uint8_t * buf) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('uint8_t *', 'buf')], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): void ns3::Ipv4Address::Set(uint32_t address) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('uint32_t', 'address')])
    ## ipv4-address.h (module 'network'): void ns3::Ipv4Address::Set(char const * address) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('char const *', 'address')])
    return

def register_Ns3Ipv4AddressHash_methods(root_module, cls):
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressHash::Ipv4AddressHash() [constructor]
    cls.add_constructor([])
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressHash::Ipv4AddressHash(ns3::Ipv4AddressHash const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv4AddressHash const &', 'arg0')])
    ## ipv4-address.h (module 'network'): size_t ns3::Ipv4AddressHash::operator()(ns3::Ipv4Address const & x) const [member operator]
    cls.add_method('operator()', 
                   'size_t', 
                   [param('ns3::Ipv4Address const &', 'x')], 
                   custom_name='__call__', is_const=True)
    return

def register_Ns3Ipv4Mask_methods(root_module, cls):
    cls.add_output_stream_operator()
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    ## ipv4-address.h (module 'network'): ns3::Ipv4Mask::Ipv4Mask(ns3::Ipv4Mask const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv4Mask const &', 'arg0')])
    ## ipv4-address.h (module 'network'): ns3::Ipv4Mask::Ipv4Mask() [constructor]
    cls.add_constructor([])
    ## ipv4-address.h (module 'network'): ns3::Ipv4Mask::Ipv4Mask(uint32_t mask) [constructor]
    cls.add_constructor([param('uint32_t', 'mask')])
    ## ipv4-address.h (module 'network'): ns3::Ipv4Mask::Ipv4Mask(char const * mask) [constructor]
    cls.add_constructor([param('char const *', 'mask')])
    ## ipv4-address.h (module 'network'): uint32_t ns3::Ipv4Mask::Get() const [member function]
    cls.add_method('Get', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): uint32_t ns3::Ipv4Mask::GetInverse() const [member function]
    cls.add_method('GetInverse', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): static ns3::Ipv4Mask ns3::Ipv4Mask::GetLoopback() [member function]
    cls.add_method('GetLoopback', 
                   'ns3::Ipv4Mask', 
                   [], 
                   is_static=True)
    ## ipv4-address.h (module 'network'): static ns3::Ipv4Mask ns3::Ipv4Mask::GetOnes() [member function]
    cls.add_method('GetOnes', 
                   'ns3::Ipv4Mask', 
                   [], 
                   is_static=True)
    ## ipv4-address.h (module 'network'): uint16_t ns3::Ipv4Mask::GetPrefixLength() const [member function]
    cls.add_method('GetPrefixLength', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): static ns3::Ipv4Mask ns3::Ipv4Mask::GetZero() [member function]
    cls.add_method('GetZero', 
                   'ns3::Ipv4Mask', 
                   [], 
                   is_static=True)
    ## ipv4-address.h (module 'network'): bool ns3::Ipv4Mask::IsMatch(ns3::Ipv4Address a, ns3::Ipv4Address b) const [member function]
    cls.add_method('IsMatch', 
                   'bool', 
                   [param('ns3::Ipv4Address', 'a'), param('ns3::Ipv4Address', 'b')], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): void ns3::Ipv4Mask::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): void ns3::Ipv4Mask::Set(uint32_t mask) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('uint32_t', 'mask')])
    return

def register_Ns3Ipv6Address_methods(root_module, cls):
    cls.add_output_stream_operator()
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<')
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address::Ipv6Address() [constructor]
    cls.add_constructor([])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address::Ipv6Address(char const * address) [constructor]
    cls.add_constructor([param('char const *', 'address')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address::Ipv6Address(uint8_t * address) [constructor]
    cls.add_constructor([param('uint8_t *', 'address')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address::Ipv6Address(ns3::Ipv6Address const & addr) [constructor]
    cls.add_constructor([param('ns3::Ipv6Address const &', 'addr')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address::Ipv6Address(ns3::Ipv6Address const * addr) [constructor]
    cls.add_constructor([param('ns3::Ipv6Address const *', 'addr')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address ns3::Ipv6Address::CombinePrefix(ns3::Ipv6Prefix const & prefix) const [member function]
    cls.add_method('CombinePrefix', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Ipv6Prefix const &', 'prefix')], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::ConvertFrom(ns3::Address const & address) [member function]
    cls.add_method('ConvertFrom', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::Deserialize(uint8_t const * buf) [member function]
    cls.add_method('Deserialize', 
                   'ns3::Ipv6Address', 
                   [param('uint8_t const *', 'buf')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::GetAllHostsMulticast() [member function]
    cls.add_method('GetAllHostsMulticast', 
                   'ns3::Ipv6Address', 
                   [], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::GetAllNodesMulticast() [member function]
    cls.add_method('GetAllNodesMulticast', 
                   'ns3::Ipv6Address', 
                   [], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::GetAllRoutersMulticast() [member function]
    cls.add_method('GetAllRoutersMulticast', 
                   'ns3::Ipv6Address', 
                   [], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::GetAny() [member function]
    cls.add_method('GetAny', 
                   'ns3::Ipv6Address', 
                   [], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): void ns3::Ipv6Address::GetBytes(uint8_t * buf) const [member function]
    cls.add_method('GetBytes', 
                   'void', 
                   [param('uint8_t *', 'buf')], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): ns3::Ipv4Address ns3::Ipv6Address::GetIpv4MappedAddress() const [member function]
    cls.add_method('GetIpv4MappedAddress', 
                   'ns3::Ipv4Address', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::GetLoopback() [member function]
    cls.add_method('GetLoopback', 
                   'ns3::Ipv6Address', 
                   [], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::GetOnes() [member function]
    cls.add_method('GetOnes', 
                   'ns3::Ipv6Address', 
                   [], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::GetZero() [member function]
    cls.add_method('GetZero', 
                   'ns3::Ipv6Address', 
                   [], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::HasPrefix(ns3::Ipv6Prefix const & prefix) const [member function]
    cls.add_method('HasPrefix', 
                   'bool', 
                   [param('ns3::Ipv6Prefix const &', 'prefix')], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsAllNodesMulticast() const [member function]
    cls.add_method('IsAllNodesMulticast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsAllRoutersMulticast() const [member function]
    cls.add_method('IsAllRoutersMulticast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsAny() const [member function]
    cls.add_method('IsAny', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsDocumentation() const [member function]
    cls.add_method('IsDocumentation', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsInitialized() const [member function]
    cls.add_method('IsInitialized', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsIpv4MappedAddress() const [member function]
    cls.add_method('IsIpv4MappedAddress', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsLinkLocal() const [member function]
    cls.add_method('IsLinkLocal', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsLinkLocalMulticast() const [member function]
    cls.add_method('IsLinkLocalMulticast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsLocalhost() const [member function]
    cls.add_method('IsLocalhost', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): static bool ns3::Ipv6Address::IsMatchingType(ns3::Address const & address) [member function]
    cls.add_method('IsMatchingType', 
                   'bool', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsMulticast() const [member function]
    cls.add_method('IsMulticast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Address::IsSolicitedMulticast() const [member function]
    cls.add_method('IsSolicitedMulticast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredAddress(ns3::Address addr, ns3::Ipv6Address prefix) [member function]
    cls.add_method('MakeAutoconfiguredAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Address', 'addr'), param('ns3::Ipv6Address', 'prefix')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredAddress(ns3::Address addr, ns3::Ipv6Prefix prefix) [member function]
    cls.add_method('MakeAutoconfiguredAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Address', 'addr'), param('ns3::Ipv6Prefix', 'prefix')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredAddress(ns3::Mac16Address addr, ns3::Ipv6Address prefix) [member function]
    cls.add_method('MakeAutoconfiguredAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Mac16Address', 'addr'), param('ns3::Ipv6Address', 'prefix')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredAddress(ns3::Mac48Address addr, ns3::Ipv6Address prefix) [member function]
    cls.add_method('MakeAutoconfiguredAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Mac48Address', 'addr'), param('ns3::Ipv6Address', 'prefix')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredAddress(ns3::Mac64Address addr, ns3::Ipv6Address prefix) [member function]
    cls.add_method('MakeAutoconfiguredAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Mac64Address', 'addr'), param('ns3::Ipv6Address', 'prefix')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredAddress(ns3::Mac8Address addr, ns3::Ipv6Address prefix) [member function]
    cls.add_method('MakeAutoconfiguredAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Mac8Address', 'addr'), param('ns3::Ipv6Address', 'prefix')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredLinkLocalAddress(ns3::Address mac) [member function]
    cls.add_method('MakeAutoconfiguredLinkLocalAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Address', 'mac')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredLinkLocalAddress(ns3::Mac16Address mac) [member function]
    cls.add_method('MakeAutoconfiguredLinkLocalAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Mac16Address', 'mac')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredLinkLocalAddress(ns3::Mac48Address mac) [member function]
    cls.add_method('MakeAutoconfiguredLinkLocalAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Mac48Address', 'mac')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredLinkLocalAddress(ns3::Mac64Address mac) [member function]
    cls.add_method('MakeAutoconfiguredLinkLocalAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Mac64Address', 'mac')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeAutoconfiguredLinkLocalAddress(ns3::Mac8Address mac) [member function]
    cls.add_method('MakeAutoconfiguredLinkLocalAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Mac8Address', 'mac')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeIpv4MappedAddress(ns3::Ipv4Address addr) [member function]
    cls.add_method('MakeIpv4MappedAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Ipv4Address', 'addr')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Address ns3::Ipv6Address::MakeSolicitedAddress(ns3::Ipv6Address addr) [member function]
    cls.add_method('MakeSolicitedAddress', 
                   'ns3::Ipv6Address', 
                   [param('ns3::Ipv6Address', 'addr')], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): void ns3::Ipv6Address::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): void ns3::Ipv6Address::Serialize(uint8_t * buf) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('uint8_t *', 'buf')], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): void ns3::Ipv6Address::Set(char const * address) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('char const *', 'address')])
    ## ipv6-address.h (module 'network'): void ns3::Ipv6Address::Set(uint8_t * address) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('uint8_t *', 'address')])
    return

def register_Ns3Ipv6AddressHash_methods(root_module, cls):
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressHash::Ipv6AddressHash() [constructor]
    cls.add_constructor([])
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressHash::Ipv6AddressHash(ns3::Ipv6AddressHash const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv6AddressHash const &', 'arg0')])
    ## ipv6-address.h (module 'network'): size_t ns3::Ipv6AddressHash::operator()(ns3::Ipv6Address const & x) const [member operator]
    cls.add_method('operator()', 
                   'size_t', 
                   [param('ns3::Ipv6Address const &', 'x')], 
                   custom_name='__call__', is_const=True)
    return

def register_Ns3Ipv6Prefix_methods(root_module, cls):
    cls.add_output_stream_operator()
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix::Ipv6Prefix() [constructor]
    cls.add_constructor([])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix::Ipv6Prefix(uint8_t * prefix) [constructor]
    cls.add_constructor([param('uint8_t *', 'prefix')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix::Ipv6Prefix(char const * prefix) [constructor]
    cls.add_constructor([param('char const *', 'prefix')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix::Ipv6Prefix(uint8_t * prefix, uint8_t prefixLength) [constructor]
    cls.add_constructor([param('uint8_t *', 'prefix'), param('uint8_t', 'prefixLength')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix::Ipv6Prefix(char const * prefix, uint8_t prefixLength) [constructor]
    cls.add_constructor([param('char const *', 'prefix'), param('uint8_t', 'prefixLength')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix::Ipv6Prefix(uint8_t prefix) [constructor]
    cls.add_constructor([param('uint8_t', 'prefix')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix::Ipv6Prefix(ns3::Ipv6Prefix const & prefix) [constructor]
    cls.add_constructor([param('ns3::Ipv6Prefix const &', 'prefix')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix::Ipv6Prefix(ns3::Ipv6Prefix const * prefix) [constructor]
    cls.add_constructor([param('ns3::Ipv6Prefix const *', 'prefix')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address ns3::Ipv6Prefix::ConvertToIpv6Address() const [member function]
    cls.add_method('ConvertToIpv6Address', 
                   'ns3::Ipv6Address', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): void ns3::Ipv6Prefix::GetBytes(uint8_t * buf) const [member function]
    cls.add_method('GetBytes', 
                   'void', 
                   [param('uint8_t *', 'buf')], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Prefix ns3::Ipv6Prefix::GetLoopback() [member function]
    cls.add_method('GetLoopback', 
                   'ns3::Ipv6Prefix', 
                   [], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): uint8_t ns3::Ipv6Prefix::GetMinimumPrefixLength() const [member function]
    cls.add_method('GetMinimumPrefixLength', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Prefix ns3::Ipv6Prefix::GetOnes() [member function]
    cls.add_method('GetOnes', 
                   'ns3::Ipv6Prefix', 
                   [], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): uint8_t ns3::Ipv6Prefix::GetPrefixLength() const [member function]
    cls.add_method('GetPrefixLength', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): static ns3::Ipv6Prefix ns3::Ipv6Prefix::GetZero() [member function]
    cls.add_method('GetZero', 
                   'ns3::Ipv6Prefix', 
                   [], 
                   is_static=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6Prefix::IsMatch(ns3::Ipv6Address a, ns3::Ipv6Address b) const [member function]
    cls.add_method('IsMatch', 
                   'bool', 
                   [param('ns3::Ipv6Address', 'a'), param('ns3::Ipv6Address', 'b')], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): void ns3::Ipv6Prefix::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): void ns3::Ipv6Prefix::SetPrefixLength(uint8_t prefixLength) [member function]
    cls.add_method('SetPrefixLength', 
                   'void', 
                   [param('uint8_t', 'prefixLength')])
    return

def register_Ns3LogComponent_methods(root_module, cls):
    ## log.h (module 'core'): ns3::LogComponent::LogComponent(ns3::LogComponent const & arg0) [constructor]
    cls.add_constructor([param('ns3::LogComponent const &', 'arg0')])
    ## log.h (module 'core'): ns3::LogComponent::LogComponent(std::string const & name, std::string const & file, ns3::LogLevel const mask=::ns3::LogLevel::LOG_NONE) [constructor]
    cls.add_constructor([param('std::string const &', 'name'), param('std::string const &', 'file'), param('ns3::LogLevel const', 'mask', default_value='::ns3::LogLevel::LOG_NONE')])
    ## log.h (module 'core'): void ns3::LogComponent::Disable(ns3::LogLevel const level) [member function]
    cls.add_method('Disable', 
                   'void', 
                   [param('ns3::LogLevel const', 'level')])
    ## log.h (module 'core'): void ns3::LogComponent::Enable(ns3::LogLevel const level) [member function]
    cls.add_method('Enable', 
                   'void', 
                   [param('ns3::LogLevel const', 'level')])
    ## log.h (module 'core'): std::string ns3::LogComponent::File() const [member function]
    cls.add_method('File', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## log.h (module 'core'): static ns3::LogComponent::ComponentList * ns3::LogComponent::GetComponentList() [member function]
    cls.add_method('GetComponentList', 
                   'ns3::LogComponent::ComponentList *', 
                   [], 
                   is_static=True)
    ## log.h (module 'core'): static std::string ns3::LogComponent::GetLevelLabel(ns3::LogLevel const level) [member function]
    cls.add_method('GetLevelLabel', 
                   'std::string', 
                   [param('ns3::LogLevel const', 'level')], 
                   is_static=True)
    ## log.h (module 'core'): bool ns3::LogComponent::IsEnabled(ns3::LogLevel const level) const [member function]
    cls.add_method('IsEnabled', 
                   'bool', 
                   [param('ns3::LogLevel const', 'level')], 
                   is_const=True)
    ## log.h (module 'core'): bool ns3::LogComponent::IsNoneEnabled() const [member function]
    cls.add_method('IsNoneEnabled', 
                   'bool', 
                   [], 
                   is_const=True)
    ## log.h (module 'core'): char const * ns3::LogComponent::Name() const [member function]
    cls.add_method('Name', 
                   'char const *', 
                   [], 
                   is_const=True)
    ## log.h (module 'core'): void ns3::LogComponent::SetMask(ns3::LogLevel const level) [member function]
    cls.add_method('SetMask', 
                   'void', 
                   [param('ns3::LogLevel const', 'level')])
    return

def register_Ns3Mac48Address_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<')
    cls.add_output_stream_operator()
    ## mac48-address.h (module 'network'): ns3::Mac48Address::Mac48Address(ns3::Mac48Address const & arg0) [constructor]
    cls.add_constructor([param('ns3::Mac48Address const &', 'arg0')])
    ## mac48-address.h (module 'network'): ns3::Mac48Address::Mac48Address() [constructor]
    cls.add_constructor([])
    ## mac48-address.h (module 'network'): ns3::Mac48Address::Mac48Address(char const * str) [constructor]
    cls.add_constructor([param('char const *', 'str')])
    ## mac48-address.h (module 'network'): static ns3::Mac48Address ns3::Mac48Address::Allocate() [member function]
    cls.add_method('Allocate', 
                   'ns3::Mac48Address', 
                   [], 
                   is_static=True)
    ## mac48-address.h (module 'network'): static ns3::Mac48Address ns3::Mac48Address::ConvertFrom(ns3::Address const & address) [member function]
    cls.add_method('ConvertFrom', 
                   'ns3::Mac48Address', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## mac48-address.h (module 'network'): void ns3::Mac48Address::CopyFrom(uint8_t const * buffer) [member function]
    cls.add_method('CopyFrom', 
                   'void', 
                   [param('uint8_t const *', 'buffer')])
    ## mac48-address.h (module 'network'): void ns3::Mac48Address::CopyTo(uint8_t * buffer) const [member function]
    cls.add_method('CopyTo', 
                   'void', 
                   [param('uint8_t *', 'buffer')], 
                   is_const=True)
    ## mac48-address.h (module 'network'): static ns3::Mac48Address ns3::Mac48Address::GetBroadcast() [member function]
    cls.add_method('GetBroadcast', 
                   'ns3::Mac48Address', 
                   [], 
                   is_static=True)
    ## mac48-address.h (module 'network'): static ns3::Mac48Address ns3::Mac48Address::GetMulticast(ns3::Ipv4Address address) [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Mac48Address', 
                   [param('ns3::Ipv4Address', 'address')], 
                   is_static=True)
    ## mac48-address.h (module 'network'): static ns3::Mac48Address ns3::Mac48Address::GetMulticast(ns3::Ipv6Address address) [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Mac48Address', 
                   [param('ns3::Ipv6Address', 'address')], 
                   is_static=True)
    ## mac48-address.h (module 'network'): static ns3::Mac48Address ns3::Mac48Address::GetMulticast6Prefix() [member function]
    cls.add_method('GetMulticast6Prefix', 
                   'ns3::Mac48Address', 
                   [], 
                   is_static=True)
    ## mac48-address.h (module 'network'): static ns3::Mac48Address ns3::Mac48Address::GetMulticastPrefix() [member function]
    cls.add_method('GetMulticastPrefix', 
                   'ns3::Mac48Address', 
                   [], 
                   is_static=True)
    ## mac48-address.h (module 'network'): bool ns3::Mac48Address::IsBroadcast() const [member function]
    cls.add_method('IsBroadcast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## mac48-address.h (module 'network'): bool ns3::Mac48Address::IsGroup() const [member function]
    cls.add_method('IsGroup', 
                   'bool', 
                   [], 
                   is_const=True)
    ## mac48-address.h (module 'network'): static bool ns3::Mac48Address::IsMatchingType(ns3::Address const & address) [member function]
    cls.add_method('IsMatchingType', 
                   'bool', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    return

def register_Ns3Mac8Address_methods(root_module, cls):
    cls.add_binary_comparison_operator('<')
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_output_stream_operator()
    ## mac8-address.h (module 'network'): ns3::Mac8Address::Mac8Address(ns3::Mac8Address const & arg0) [constructor]
    cls.add_constructor([param('ns3::Mac8Address const &', 'arg0')])
    ## mac8-address.h (module 'network'): ns3::Mac8Address::Mac8Address() [constructor]
    cls.add_constructor([])
    ## mac8-address.h (module 'network'): ns3::Mac8Address::Mac8Address(uint8_t addr) [constructor]
    cls.add_constructor([param('uint8_t', 'addr')])
    ## mac8-address.h (module 'network'): static ns3::Mac8Address ns3::Mac8Address::Allocate() [member function]
    cls.add_method('Allocate', 
                   'ns3::Mac8Address', 
                   [], 
                   is_static=True)
    ## mac8-address.h (module 'network'): static ns3::Mac8Address ns3::Mac8Address::ConvertFrom(ns3::Address const & address) [member function]
    cls.add_method('ConvertFrom', 
                   'ns3::Mac8Address', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## mac8-address.h (module 'network'): void ns3::Mac8Address::CopyFrom(uint8_t const * pBuffer) [member function]
    cls.add_method('CopyFrom', 
                   'void', 
                   [param('uint8_t const *', 'pBuffer')])
    ## mac8-address.h (module 'network'): void ns3::Mac8Address::CopyTo(uint8_t * pBuffer) const [member function]
    cls.add_method('CopyTo', 
                   'void', 
                   [param('uint8_t *', 'pBuffer')], 
                   is_const=True)
    ## mac8-address.h (module 'network'): static ns3::Mac8Address ns3::Mac8Address::GetBroadcast() [member function]
    cls.add_method('GetBroadcast', 
                   'ns3::Mac8Address', 
                   [], 
                   is_static=True)
    ## mac8-address.h (module 'network'): static bool ns3::Mac8Address::IsMatchingType(ns3::Address const & address) [member function]
    cls.add_method('IsMatchingType', 
                   'bool', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    return

def register_Ns3MeshHelper_methods(root_module, cls):
    ## mesh-helper.h (module 'mesh'): ns3::MeshHelper::MeshHelper(ns3::MeshHelper const & arg0) [constructor]
    cls.add_constructor([param('ns3::MeshHelper const &', 'arg0')])
    ## mesh-helper.h (module 'mesh'): ns3::MeshHelper::MeshHelper() [constructor]
    cls.add_constructor([])
    ## mesh-helper.h (module 'mesh'): int64_t ns3::MeshHelper::AssignStreams(ns3::NetDeviceContainer c, int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('ns3::NetDeviceContainer', 'c'), param('int64_t', 'stream')])
    ## mesh-helper.h (module 'mesh'): static ns3::MeshHelper ns3::MeshHelper::Default() [member function]
    cls.add_method('Default', 
                   'ns3::MeshHelper', 
                   [], 
                   is_static=True)
    ## mesh-helper.h (module 'mesh'): static void ns3::MeshHelper::EnableLogComponents() [member function]
    cls.add_method('EnableLogComponents', 
                   'void', 
                   [], 
                   is_static=True)
    ## mesh-helper.h (module 'mesh'): ns3::NetDeviceContainer ns3::MeshHelper::Install(ns3::WifiPhyHelper const & phyHelper, ns3::NodeContainer c) const [member function]
    cls.add_method('Install', 
                   'ns3::NetDeviceContainer', 
                   [param('ns3::WifiPhyHelper const &', 'phyHelper'), param('ns3::NodeContainer', 'c')], 
                   is_const=True)
    ## mesh-helper.h (module 'mesh'): void ns3::MeshHelper::Report(ns3::Ptr<ns3::NetDevice> const & device, std::ostream & os) [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('ns3::Ptr< ns3::NetDevice > const &', 'device'), param('std::ostream &', 'os')])
    ## mesh-helper.h (module 'mesh'): void ns3::MeshHelper::ResetStats(ns3::Ptr<ns3::NetDevice> const & device) [member function]
    cls.add_method('ResetStats', 
                   'void', 
                   [param('ns3::Ptr< ns3::NetDevice > const &', 'device')])
    ## mesh-helper.h (module 'mesh'): void ns3::MeshHelper::SetMacType(std::string n0="", ns3::AttributeValue const & v0=ns3::EmptyAttributeValue(), std::string n1="", ns3::AttributeValue const & v1=ns3::EmptyAttributeValue(), std::string n2="", ns3::AttributeValue const & v2=ns3::EmptyAttributeValue(), std::string n3="", ns3::AttributeValue const & v3=ns3::EmptyAttributeValue(), std::string n4="", ns3::AttributeValue const & v4=ns3::EmptyAttributeValue(), std::string n5="", ns3::AttributeValue const & v5=ns3::EmptyAttributeValue(), std::string n6="", ns3::AttributeValue const & v6=ns3::EmptyAttributeValue(), std::string n7="", ns3::AttributeValue const & v7=ns3::EmptyAttributeValue()) [member function]
    cls.add_method('SetMacType', 
                   'void', 
                   [param('std::string', 'n0', default_value='""'), param('ns3::AttributeValue const &', 'v0', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n1', default_value='""'), param('ns3::AttributeValue const &', 'v1', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n2', default_value='""'), param('ns3::AttributeValue const &', 'v2', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n3', default_value='""'), param('ns3::AttributeValue const &', 'v3', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n4', default_value='""'), param('ns3::AttributeValue const &', 'v4', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n5', default_value='""'), param('ns3::AttributeValue const &', 'v5', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n6', default_value='""'), param('ns3::AttributeValue const &', 'v6', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n7', default_value='""'), param('ns3::AttributeValue const &', 'v7', default_value='ns3::EmptyAttributeValue()')])
    ## mesh-helper.h (module 'mesh'): void ns3::MeshHelper::SetNumberOfInterfaces(uint32_t nInterfaces) [member function]
    cls.add_method('SetNumberOfInterfaces', 
                   'void', 
                   [param('uint32_t', 'nInterfaces')])
    ## mesh-helper.h (module 'mesh'): void ns3::MeshHelper::SetRemoteStationManager(std::string type, std::string n0="", ns3::AttributeValue const & v0=ns3::EmptyAttributeValue(), std::string n1="", ns3::AttributeValue const & v1=ns3::EmptyAttributeValue(), std::string n2="", ns3::AttributeValue const & v2=ns3::EmptyAttributeValue(), std::string n3="", ns3::AttributeValue const & v3=ns3::EmptyAttributeValue(), std::string n4="", ns3::AttributeValue const & v4=ns3::EmptyAttributeValue(), std::string n5="", ns3::AttributeValue const & v5=ns3::EmptyAttributeValue(), std::string n6="", ns3::AttributeValue const & v6=ns3::EmptyAttributeValue(), std::string n7="", ns3::AttributeValue const & v7=ns3::EmptyAttributeValue()) [member function]
    cls.add_method('SetRemoteStationManager', 
                   'void', 
                   [param('std::string', 'type'), param('std::string', 'n0', default_value='""'), param('ns3::AttributeValue const &', 'v0', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n1', default_value='""'), param('ns3::AttributeValue const &', 'v1', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n2', default_value='""'), param('ns3::AttributeValue const &', 'v2', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n3', default_value='""'), param('ns3::AttributeValue const &', 'v3', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n4', default_value='""'), param('ns3::AttributeValue const &', 'v4', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n5', default_value='""'), param('ns3::AttributeValue const &', 'v5', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n6', default_value='""'), param('ns3::AttributeValue const &', 'v6', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n7', default_value='""'), param('ns3::AttributeValue const &', 'v7', default_value='ns3::EmptyAttributeValue()')])
    ## mesh-helper.h (module 'mesh'): void ns3::MeshHelper::SetSpreadInterfaceChannels(ns3::MeshHelper::ChannelPolicy policy) [member function]
    cls.add_method('SetSpreadInterfaceChannels', 
                   'void', 
                   [param('ns3::MeshHelper::ChannelPolicy', 'policy')])
    ## mesh-helper.h (module 'mesh'): void ns3::MeshHelper::SetStackInstaller(std::string type, std::string n0="", ns3::AttributeValue const & v0=ns3::EmptyAttributeValue(), std::string n1="", ns3::AttributeValue const & v1=ns3::EmptyAttributeValue(), std::string n2="", ns3::AttributeValue const & v2=ns3::EmptyAttributeValue(), std::string n3="", ns3::AttributeValue const & v3=ns3::EmptyAttributeValue(), std::string n4="", ns3::AttributeValue const & v4=ns3::EmptyAttributeValue(), std::string n5="", ns3::AttributeValue const & v5=ns3::EmptyAttributeValue(), std::string n6="", ns3::AttributeValue const & v6=ns3::EmptyAttributeValue(), std::string n7="", ns3::AttributeValue const & v7=ns3::EmptyAttributeValue()) [member function]
    cls.add_method('SetStackInstaller', 
                   'void', 
                   [param('std::string', 'type'), param('std::string', 'n0', default_value='""'), param('ns3::AttributeValue const &', 'v0', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n1', default_value='""'), param('ns3::AttributeValue const &', 'v1', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n2', default_value='""'), param('ns3::AttributeValue const &', 'v2', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n3', default_value='""'), param('ns3::AttributeValue const &', 'v3', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n4', default_value='""'), param('ns3::AttributeValue const &', 'v4', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n5', default_value='""'), param('ns3::AttributeValue const &', 'v5', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n6', default_value='""'), param('ns3::AttributeValue const &', 'v6', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n7', default_value='""'), param('ns3::AttributeValue const &', 'v7', default_value='ns3::EmptyAttributeValue()')])
    ## mesh-helper.h (module 'mesh'): void ns3::MeshHelper::SetStandard(ns3::WifiStandard standard) [member function]
    cls.add_method('SetStandard', 
                   'void', 
                   [param('ns3::WifiStandard', 'standard')])
    return

def register_Ns3MeshWifiBeacon_methods(root_module, cls):
    ## mesh-wifi-beacon.h (module 'mesh'): ns3::MeshWifiBeacon::MeshWifiBeacon(ns3::MeshWifiBeacon const & arg0) [constructor]
    cls.add_constructor([param('ns3::MeshWifiBeacon const &', 'arg0')])
    ## mesh-wifi-beacon.h (module 'mesh'): ns3::MeshWifiBeacon::MeshWifiBeacon(ns3::Ssid ssid, ns3::SupportedRates rates, uint64_t us) [constructor]
    cls.add_constructor([param('ns3::Ssid', 'ssid'), param('ns3::SupportedRates', 'rates'), param('uint64_t', 'us')])
    ## mesh-wifi-beacon.h (module 'mesh'): void ns3::MeshWifiBeacon::AddInformationElement(ns3::Ptr<ns3::WifiInformationElement> ie) [member function]
    cls.add_method('AddInformationElement', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiInformationElement >', 'ie')])
    ## mesh-wifi-beacon.h (module 'mesh'): ns3::MgtBeaconHeader ns3::MeshWifiBeacon::BeaconHeader() const [member function]
    cls.add_method('BeaconHeader', 
                   'ns3::MgtBeaconHeader', 
                   [], 
                   is_const=True)
    ## mesh-wifi-beacon.h (module 'mesh'): ns3::WifiMacHeader ns3::MeshWifiBeacon::CreateHeader(ns3::Mac48Address address, ns3::Mac48Address mpAddress) [member function]
    cls.add_method('CreateHeader', 
                   'ns3::WifiMacHeader', 
                   [param('ns3::Mac48Address', 'address'), param('ns3::Mac48Address', 'mpAddress')])
    ## mesh-wifi-beacon.h (module 'mesh'): ns3::Ptr<ns3::Packet> ns3::MeshWifiBeacon::CreatePacket() [member function]
    cls.add_method('CreatePacket', 
                   'ns3::Ptr< ns3::Packet >', 
                   [])
    ## mesh-wifi-beacon.h (module 'mesh'): ns3::Time ns3::MeshWifiBeacon::GetBeaconInterval() const [member function]
    cls.add_method('GetBeaconInterval', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    return

def register_Ns3ObjectBase_methods(root_module, cls):
    ## object-base.h (module 'core'): ns3::ObjectBase::ObjectBase() [constructor]
    cls.add_constructor([])
    ## object-base.h (module 'core'): ns3::ObjectBase::ObjectBase(ns3::ObjectBase const & arg0) [constructor]
    cls.add_constructor([param('ns3::ObjectBase const &', 'arg0')])
    ## object-base.h (module 'core'): void ns3::ObjectBase::GetAttribute(std::string name, ns3::AttributeValue & value) const [member function]
    cls.add_method('GetAttribute', 
                   'void', 
                   [param('std::string', 'name'), param('ns3::AttributeValue &', 'value')], 
                   is_const=True)
    ## object-base.h (module 'core'): bool ns3::ObjectBase::GetAttributeFailSafe(std::string name, ns3::AttributeValue & value) const [member function]
    cls.add_method('GetAttributeFailSafe', 
                   'bool', 
                   [param('std::string', 'name'), param('ns3::AttributeValue &', 'value')], 
                   is_const=True)
    ## object-base.h (module 'core'): ns3::TypeId ns3::ObjectBase::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## object-base.h (module 'core'): static ns3::TypeId ns3::ObjectBase::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## object-base.h (module 'core'): void ns3::ObjectBase::SetAttribute(std::string name, ns3::AttributeValue const & value) [member function]
    cls.add_method('SetAttribute', 
                   'void', 
                   [param('std::string', 'name'), param('ns3::AttributeValue const &', 'value')])
    ## object-base.h (module 'core'): bool ns3::ObjectBase::SetAttributeFailSafe(std::string name, ns3::AttributeValue const & value) [member function]
    cls.add_method('SetAttributeFailSafe', 
                   'bool', 
                   [param('std::string', 'name'), param('ns3::AttributeValue const &', 'value')])
    ## object-base.h (module 'core'): bool ns3::ObjectBase::TraceConnect(std::string name, std::string context, ns3::CallbackBase const & cb) [member function]
    cls.add_method('TraceConnect', 
                   'bool', 
                   [param('std::string', 'name'), param('std::string', 'context'), param('ns3::CallbackBase const &', 'cb')])
    ## object-base.h (module 'core'): bool ns3::ObjectBase::TraceConnectWithoutContext(std::string name, ns3::CallbackBase const & cb) [member function]
    cls.add_method('TraceConnectWithoutContext', 
                   'bool', 
                   [param('std::string', 'name'), param('ns3::CallbackBase const &', 'cb')])
    ## object-base.h (module 'core'): bool ns3::ObjectBase::TraceDisconnect(std::string name, std::string context, ns3::CallbackBase const & cb) [member function]
    cls.add_method('TraceDisconnect', 
                   'bool', 
                   [param('std::string', 'name'), param('std::string', 'context'), param('ns3::CallbackBase const &', 'cb')])
    ## object-base.h (module 'core'): bool ns3::ObjectBase::TraceDisconnectWithoutContext(std::string name, ns3::CallbackBase const & cb) [member function]
    cls.add_method('TraceDisconnectWithoutContext', 
                   'bool', 
                   [param('std::string', 'name'), param('ns3::CallbackBase const &', 'cb')])
    ## object-base.h (module 'core'): void ns3::ObjectBase::ConstructSelf(ns3::AttributeConstructionList const & attributes) [member function]
    cls.add_method('ConstructSelf', 
                   'void', 
                   [param('ns3::AttributeConstructionList const &', 'attributes')], 
                   visibility='protected')
    ## object-base.h (module 'core'): void ns3::ObjectBase::NotifyConstructionCompleted() [member function]
    cls.add_method('NotifyConstructionCompleted', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3ObjectDeleter_methods(root_module, cls):
    ## object.h (module 'core'): ns3::ObjectDeleter::ObjectDeleter() [constructor]
    cls.add_constructor([])
    ## object.h (module 'core'): ns3::ObjectDeleter::ObjectDeleter(ns3::ObjectDeleter const & arg0) [constructor]
    cls.add_constructor([param('ns3::ObjectDeleter const &', 'arg0')])
    ## object.h (module 'core'): static void ns3::ObjectDeleter::Delete(ns3::Object * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::Object *', 'object')], 
                   is_static=True)
    return

def register_Ns3ObjectFactory_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## object-factory.h (module 'core'): ns3::ObjectFactory::ObjectFactory(ns3::ObjectFactory const & arg0) [constructor]
    cls.add_constructor([param('ns3::ObjectFactory const &', 'arg0')])
    ## object-factory.h (module 'core'): ns3::ObjectFactory::ObjectFactory() [constructor]
    cls.add_constructor([])
    ## object-factory.h (module 'core'): ns3::ObjectFactory::ObjectFactory(std::string const & typeId) [constructor]
    cls.add_constructor([param('std::string const &', 'typeId')])
    ## object-factory.h (module 'core'): ns3::Ptr<ns3::Object> ns3::ObjectFactory::Create() const [member function]
    cls.add_method('Create', 
                   'ns3::Ptr< ns3::Object >', 
                   [], 
                   is_const=True)
    ## object-factory.h (module 'core'): ns3::TypeId ns3::ObjectFactory::GetTypeId() const [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True)
    ## object-factory.h (module 'core'): bool ns3::ObjectFactory::IsTypeIdSet() const [member function]
    cls.add_method('IsTypeIdSet', 
                   'bool', 
                   [], 
                   is_const=True)
    ## object-factory.h (module 'core'): void ns3::ObjectFactory::Set() [member function]
    cls.add_method('Set', 
                   'void', 
                   [])
    ## object-factory.h (module 'core'): void ns3::ObjectFactory::SetTypeId(ns3::TypeId tid) [member function]
    cls.add_method('SetTypeId', 
                   'void', 
                   [param('ns3::TypeId', 'tid')])
    ## object-factory.h (module 'core'): void ns3::ObjectFactory::SetTypeId(char const * tid) [member function]
    cls.add_method('SetTypeId', 
                   'void', 
                   [param('char const *', 'tid')])
    ## object-factory.h (module 'core'): void ns3::ObjectFactory::SetTypeId(std::string tid) [member function]
    cls.add_method('SetTypeId', 
                   'void', 
                   [param('std::string', 'tid')])
    return

def register_Ns3PacketMetadata_methods(root_module, cls):
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::PacketMetadata(uint64_t uid, uint32_t size) [constructor]
    cls.add_constructor([param('uint64_t', 'uid'), param('uint32_t', 'size')])
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::PacketMetadata(ns3::PacketMetadata const & o) [constructor]
    cls.add_constructor([param('ns3::PacketMetadata const &', 'o')])
    ## packet-metadata.h (module 'network'): void ns3::PacketMetadata::AddAtEnd(ns3::PacketMetadata const & o) [member function]
    cls.add_method('AddAtEnd', 
                   'void', 
                   [param('ns3::PacketMetadata const &', 'o')])
    ## packet-metadata.h (module 'network'): void ns3::PacketMetadata::AddHeader(ns3::Header const & header, uint32_t size) [member function]
    cls.add_method('AddHeader', 
                   'void', 
                   [param('ns3::Header const &', 'header'), param('uint32_t', 'size')])
    ## packet-metadata.h (module 'network'): void ns3::PacketMetadata::AddPaddingAtEnd(uint32_t end) [member function]
    cls.add_method('AddPaddingAtEnd', 
                   'void', 
                   [param('uint32_t', 'end')])
    ## packet-metadata.h (module 'network'): void ns3::PacketMetadata::AddTrailer(ns3::Trailer const & trailer, uint32_t size) [member function]
    cls.add_method('AddTrailer', 
                   'void', 
                   [param('ns3::Trailer const &', 'trailer'), param('uint32_t', 'size')])
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::ItemIterator ns3::PacketMetadata::BeginItem(ns3::Buffer buffer) const [member function]
    cls.add_method('BeginItem', 
                   'ns3::PacketMetadata::ItemIterator', 
                   [param('ns3::Buffer', 'buffer')], 
                   is_const=True)
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata ns3::PacketMetadata::CreateFragment(uint32_t start, uint32_t end) const [member function]
    cls.add_method('CreateFragment', 
                   'ns3::PacketMetadata', 
                   [param('uint32_t', 'start'), param('uint32_t', 'end')], 
                   is_const=True)
    ## packet-metadata.h (module 'network'): uint32_t ns3::PacketMetadata::Deserialize(uint8_t const * buffer, uint32_t size) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('uint8_t const *', 'buffer'), param('uint32_t', 'size')])
    ## packet-metadata.h (module 'network'): static void ns3::PacketMetadata::Enable() [member function]
    cls.add_method('Enable', 
                   'void', 
                   [], 
                   is_static=True)
    ## packet-metadata.h (module 'network'): static void ns3::PacketMetadata::EnableChecking() [member function]
    cls.add_method('EnableChecking', 
                   'void', 
                   [], 
                   is_static=True)
    ## packet-metadata.h (module 'network'): uint32_t ns3::PacketMetadata::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packet-metadata.h (module 'network'): uint64_t ns3::PacketMetadata::GetUid() const [member function]
    cls.add_method('GetUid', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## packet-metadata.h (module 'network'): void ns3::PacketMetadata::RemoveAtEnd(uint32_t end) [member function]
    cls.add_method('RemoveAtEnd', 
                   'void', 
                   [param('uint32_t', 'end')])
    ## packet-metadata.h (module 'network'): void ns3::PacketMetadata::RemoveAtStart(uint32_t start) [member function]
    cls.add_method('RemoveAtStart', 
                   'void', 
                   [param('uint32_t', 'start')])
    ## packet-metadata.h (module 'network'): void ns3::PacketMetadata::RemoveHeader(ns3::Header const & header, uint32_t size) [member function]
    cls.add_method('RemoveHeader', 
                   'void', 
                   [param('ns3::Header const &', 'header'), param('uint32_t', 'size')])
    ## packet-metadata.h (module 'network'): void ns3::PacketMetadata::RemoveTrailer(ns3::Trailer const & trailer, uint32_t size) [member function]
    cls.add_method('RemoveTrailer', 
                   'void', 
                   [param('ns3::Trailer const &', 'trailer'), param('uint32_t', 'size')])
    ## packet-metadata.h (module 'network'): uint32_t ns3::PacketMetadata::Serialize(uint8_t * buffer, uint32_t maxSize) const [member function]
    cls.add_method('Serialize', 
                   'uint32_t', 
                   [param('uint8_t *', 'buffer'), param('uint32_t', 'maxSize')], 
                   is_const=True)
    return

def register_Ns3PacketMetadataItem_methods(root_module, cls):
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::Item() [constructor]
    cls.add_constructor([])
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::Item(ns3::PacketMetadata::Item const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketMetadata::Item const &', 'arg0')])
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::current [variable]
    cls.add_instance_attribute('current', 'ns3::Buffer::Iterator', is_const=False)
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::currentSize [variable]
    cls.add_instance_attribute('currentSize', 'uint32_t', is_const=False)
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::currentTrimedFromEnd [variable]
    cls.add_instance_attribute('currentTrimedFromEnd', 'uint32_t', is_const=False)
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::currentTrimedFromStart [variable]
    cls.add_instance_attribute('currentTrimedFromStart', 'uint32_t', is_const=False)
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::isFragment [variable]
    cls.add_instance_attribute('isFragment', 'bool', is_const=False)
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::tid [variable]
    cls.add_instance_attribute('tid', 'ns3::TypeId', is_const=False)
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::type [variable]
    cls.add_instance_attribute('type', 'ns3::PacketMetadata::Item::ItemType', is_const=False)
    return

def register_Ns3PacketMetadataItemIterator_methods(root_module, cls):
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::ItemIterator::ItemIterator(ns3::PacketMetadata::ItemIterator const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketMetadata::ItemIterator const &', 'arg0')])
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::ItemIterator::ItemIterator(ns3::PacketMetadata const * metadata, ns3::Buffer buffer) [constructor]
    cls.add_constructor([param('ns3::PacketMetadata const *', 'metadata'), param('ns3::Buffer', 'buffer')])
    ## packet-metadata.h (module 'network'): bool ns3::PacketMetadata::ItemIterator::HasNext() const [member function]
    cls.add_method('HasNext', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item ns3::PacketMetadata::ItemIterator::Next() [member function]
    cls.add_method('Next', 
                   'ns3::PacketMetadata::Item', 
                   [])
    return

def register_Ns3PacketTagIterator_methods(root_module, cls):
    ## packet.h (module 'network'): ns3::PacketTagIterator::PacketTagIterator(ns3::PacketTagIterator const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketTagIterator const &', 'arg0')])
    ## packet.h (module 'network'): bool ns3::PacketTagIterator::HasNext() const [member function]
    cls.add_method('HasNext', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): ns3::PacketTagIterator::Item ns3::PacketTagIterator::Next() [member function]
    cls.add_method('Next', 
                   'ns3::PacketTagIterator::Item', 
                   [])
    return

def register_Ns3PacketTagIteratorItem_methods(root_module, cls):
    ## packet.h (module 'network'): ns3::PacketTagIterator::Item::Item(ns3::PacketTagIterator::Item const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketTagIterator::Item const &', 'arg0')])
    ## packet.h (module 'network'): void ns3::PacketTagIterator::Item::GetTag(ns3::Tag & tag) const [member function]
    cls.add_method('GetTag', 
                   'void', 
                   [param('ns3::Tag &', 'tag')], 
                   is_const=True)
    ## packet.h (module 'network'): ns3::TypeId ns3::PacketTagIterator::Item::GetTypeId() const [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True)
    return

def register_Ns3PacketTagList_methods(root_module, cls):
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::PacketTagList() [constructor]
    cls.add_constructor([])
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::PacketTagList(ns3::PacketTagList const & o) [constructor]
    cls.add_constructor([param('ns3::PacketTagList const &', 'o')])
    ## packet-tag-list.h (module 'network'): void ns3::PacketTagList::Add(ns3::Tag const & tag) const [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::Tag const &', 'tag')], 
                   is_const=True)
    ## packet-tag-list.h (module 'network'): uint32_t ns3::PacketTagList::Deserialize(uint32_t const * buffer, uint32_t size) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('uint32_t const *', 'buffer'), param('uint32_t', 'size')])
    ## packet-tag-list.h (module 'network'): uint32_t ns3::PacketTagList::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::TagData const * ns3::PacketTagList::Head() const [member function]
    cls.add_method('Head', 
                   'ns3::PacketTagList::TagData const *', 
                   [], 
                   is_const=True)
    ## packet-tag-list.h (module 'network'): bool ns3::PacketTagList::Peek(ns3::Tag & tag) const [member function]
    cls.add_method('Peek', 
                   'bool', 
                   [param('ns3::Tag &', 'tag')], 
                   is_const=True)
    ## packet-tag-list.h (module 'network'): bool ns3::PacketTagList::Remove(ns3::Tag & tag) [member function]
    cls.add_method('Remove', 
                   'bool', 
                   [param('ns3::Tag &', 'tag')])
    ## packet-tag-list.h (module 'network'): void ns3::PacketTagList::RemoveAll() [member function]
    cls.add_method('RemoveAll', 
                   'void', 
                   [])
    ## packet-tag-list.h (module 'network'): bool ns3::PacketTagList::Replace(ns3::Tag & tag) [member function]
    cls.add_method('Replace', 
                   'bool', 
                   [param('ns3::Tag &', 'tag')])
    ## packet-tag-list.h (module 'network'): uint32_t ns3::PacketTagList::Serialize(uint32_t * buffer, uint32_t maxSize) const [member function]
    cls.add_method('Serialize', 
                   'uint32_t', 
                   [param('uint32_t *', 'buffer'), param('uint32_t', 'maxSize')], 
                   is_const=True)
    return

def register_Ns3PacketTagListTagData_methods(root_module, cls):
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::TagData::TagData() [constructor]
    cls.add_constructor([])
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::TagData::TagData(ns3::PacketTagList::TagData const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketTagList::TagData const &', 'arg0')])
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::TagData::count [variable]
    cls.add_instance_attribute('count', 'uint32_t', is_const=False)
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::TagData::data [variable]
    cls.add_instance_attribute('data', 'uint8_t [ 1 ]', is_const=False)
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::TagData::next [variable]
    cls.add_instance_attribute('next', 'ns3::PacketTagList::TagData *', is_const=False)
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::TagData::size [variable]
    cls.add_instance_attribute('size', 'uint32_t', is_const=False)
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::TagData::tid [variable]
    cls.add_instance_attribute('tid', 'ns3::TypeId', is_const=False)
    return

def register_Ns3ParameterLogger_methods(root_module, cls):
    ## log.h (module 'core'): ns3::ParameterLogger::ParameterLogger(ns3::ParameterLogger const & arg0) [constructor]
    cls.add_constructor([param('ns3::ParameterLogger const &', 'arg0')])
    ## log.h (module 'core'): ns3::ParameterLogger::ParameterLogger(std::ostream & os) [constructor]
    cls.add_constructor([param('std::ostream &', 'os')])
    return

def register_Ns3SimpleRefCount__Ns3Object_Ns3ObjectBase_Ns3ObjectDeleter_methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter>::SimpleRefCount(ns3::SimpleRefCount<ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter > const &', 'o')])
    return

def register_Ns3StatusCode_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## status-code.h (module 'wifi'): ns3::StatusCode::StatusCode(ns3::StatusCode const & arg0) [constructor]
    cls.add_constructor([param('ns3::StatusCode const &', 'arg0')])
    ## status-code.h (module 'wifi'): ns3::StatusCode::StatusCode() [constructor]
    cls.add_constructor([])
    ## status-code.h (module 'wifi'): ns3::Buffer::Iterator ns3::StatusCode::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## status-code.h (module 'wifi'): uint32_t ns3::StatusCode::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## status-code.h (module 'wifi'): bool ns3::StatusCode::IsSuccess() const [member function]
    cls.add_method('IsSuccess', 
                   'bool', 
                   [], 
                   is_const=True)
    ## status-code.h (module 'wifi'): ns3::Buffer::Iterator ns3::StatusCode::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True)
    ## status-code.h (module 'wifi'): void ns3::StatusCode::SetFailure() [member function]
    cls.add_method('SetFailure', 
                   'void', 
                   [])
    ## status-code.h (module 'wifi'): void ns3::StatusCode::SetSuccess() [member function]
    cls.add_method('SetSuccess', 
                   'void', 
                   [])
    return

def register_Ns3Tag_methods(root_module, cls):
    ## tag.h (module 'network'): ns3::Tag::Tag() [constructor]
    cls.add_constructor([])
    ## tag.h (module 'network'): ns3::Tag::Tag(ns3::Tag const & arg0) [constructor]
    cls.add_constructor([param('ns3::Tag const &', 'arg0')])
    ## tag.h (module 'network'): void ns3::Tag::Deserialize(ns3::TagBuffer i) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_pure_virtual=True, is_virtual=True)
    ## tag.h (module 'network'): uint32_t ns3::Tag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## tag.h (module 'network'): static ns3::TypeId ns3::Tag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## tag.h (module 'network'): void ns3::Tag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## tag.h (module 'network'): void ns3::Tag::Serialize(ns3::TagBuffer i) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3TagBuffer_methods(root_module, cls):
    ## tag-buffer.h (module 'network'): ns3::TagBuffer::TagBuffer(ns3::TagBuffer const & arg0) [constructor]
    cls.add_constructor([param('ns3::TagBuffer const &', 'arg0')])
    ## tag-buffer.h (module 'network'): ns3::TagBuffer::TagBuffer(uint8_t * start, uint8_t * end) [constructor]
    cls.add_constructor([param('uint8_t *', 'start'), param('uint8_t *', 'end')])
    ## tag-buffer.h (module 'network'): void ns3::TagBuffer::CopyFrom(ns3::TagBuffer o) [member function]
    cls.add_method('CopyFrom', 
                   'void', 
                   [param('ns3::TagBuffer', 'o')])
    ## tag-buffer.h (module 'network'): void ns3::TagBuffer::Read(uint8_t * buffer, uint32_t size) [member function]
    cls.add_method('Read', 
                   'void', 
                   [param('uint8_t *', 'buffer'), param('uint32_t', 'size')])
    ## tag-buffer.h (module 'network'): double ns3::TagBuffer::ReadDouble() [member function]
    cls.add_method('ReadDouble', 
                   'double', 
                   [])
    ## tag-buffer.h (module 'network'): uint16_t ns3::TagBuffer::ReadU16() [member function]
    cls.add_method('ReadU16', 
                   'uint16_t', 
                   [])
    ## tag-buffer.h (module 'network'): uint32_t ns3::TagBuffer::ReadU32() [member function]
    cls.add_method('ReadU32', 
                   'uint32_t', 
                   [])
    ## tag-buffer.h (module 'network'): uint64_t ns3::TagBuffer::ReadU64() [member function]
    cls.add_method('ReadU64', 
                   'uint64_t', 
                   [])
    ## tag-buffer.h (module 'network'): uint8_t ns3::TagBuffer::ReadU8() [member function]
    cls.add_method('ReadU8', 
                   'uint8_t', 
                   [])
    ## tag-buffer.h (module 'network'): void ns3::TagBuffer::TrimAtEnd(uint32_t trim) [member function]
    cls.add_method('TrimAtEnd', 
                   'void', 
                   [param('uint32_t', 'trim')])
    ## tag-buffer.h (module 'network'): void ns3::TagBuffer::Write(uint8_t const * buffer, uint32_t size) [member function]
    cls.add_method('Write', 
                   'void', 
                   [param('uint8_t const *', 'buffer'), param('uint32_t', 'size')])
    ## tag-buffer.h (module 'network'): void ns3::TagBuffer::WriteDouble(double v) [member function]
    cls.add_method('WriteDouble', 
                   'void', 
                   [param('double', 'v')])
    ## tag-buffer.h (module 'network'): void ns3::TagBuffer::WriteU16(uint16_t v) [member function]
    cls.add_method('WriteU16', 
                   'void', 
                   [param('uint16_t', 'v')])
    ## tag-buffer.h (module 'network'): void ns3::TagBuffer::WriteU32(uint32_t v) [member function]
    cls.add_method('WriteU32', 
                   'void', 
                   [param('uint32_t', 'v')])
    ## tag-buffer.h (module 'network'): void ns3::TagBuffer::WriteU64(uint64_t v) [member function]
    cls.add_method('WriteU64', 
                   'void', 
                   [param('uint64_t', 'v')])
    ## tag-buffer.h (module 'network'): void ns3::TagBuffer::WriteU8(uint8_t v) [member function]
    cls.add_method('WriteU8', 
                   'void', 
                   [param('uint8_t', 'v')])
    return

def register_Ns3Time_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<=')
    cls.add_binary_comparison_operator('>=')
    cls.add_binary_comparison_operator('<')
    cls.add_binary_comparison_operator('>')
    cls.add_binary_numeric_operator('+', root_module['ns3::Time'], root_module['ns3::Time'], param('ns3::Time const &', 'right'))
    cls.add_binary_numeric_operator('-', root_module['ns3::Time'], root_module['ns3::Time'], param('ns3::Time const &', 'right'))
    cls.add_binary_numeric_operator('*', root_module['ns3::Time'], root_module['ns3::Time'], param('ns3::int64x64_t const &', 'right'))
    cls.add_binary_numeric_operator('/', root_module['ns3::int64x64_t'], root_module['ns3::Time'], param('ns3::Time const &', 'right'))
    cls.add_binary_numeric_operator('/', root_module['ns3::Time'], root_module['ns3::Time'], param('ns3::int64x64_t const &', 'right'))
    cls.add_inplace_numeric_operator('+=', param('ns3::Time const &', 'right'))
    cls.add_inplace_numeric_operator('-=', param('ns3::Time const &', 'right'))
    cls.add_output_stream_operator()
    ## nstime.h (module 'core'): ns3::Time::Time() [constructor]
    cls.add_constructor([])
    ## nstime.h (module 'core'): ns3::Time::Time(ns3::Time const & o) [constructor]
    cls.add_constructor([param('ns3::Time const &', 'o')])
    ## nstime.h (module 'core'): ns3::Time::Time(double v) [constructor]
    cls.add_constructor([param('double', 'v')])
    ## nstime.h (module 'core'): ns3::Time::Time(int v) [constructor]
    cls.add_constructor([param('int', 'v')])
    ## nstime.h (module 'core'): ns3::Time::Time(long int v) [constructor]
    cls.add_constructor([param('long int', 'v')])
    ## nstime.h (module 'core'): ns3::Time::Time(long long int v) [constructor]
    cls.add_constructor([param('long long int', 'v')])
    ## nstime.h (module 'core'): ns3::Time::Time(unsigned int v) [constructor]
    cls.add_constructor([param('unsigned int', 'v')])
    ## nstime.h (module 'core'): ns3::Time::Time(long unsigned int v) [constructor]
    cls.add_constructor([param('long unsigned int', 'v')])
    ## nstime.h (module 'core'): ns3::Time::Time(long long unsigned int v) [constructor]
    cls.add_constructor([param('long long unsigned int', 'v')])
    ## nstime.h (module 'core'): ns3::Time::Time(ns3::int64x64_t const & v) [constructor]
    cls.add_constructor([param('ns3::int64x64_t const &', 'v')])
    ## nstime.h (module 'core'): ns3::Time::Time(std::string const & s) [constructor]
    cls.add_constructor([param('std::string const &', 's')])
    ## nstime.h (module 'core'): ns3::TimeWithUnit ns3::Time::As(ns3::Time::Unit const unit=::ns3::Time::Unit::AUTO) const [member function]
    cls.add_method('As', 
                   'ns3::TimeWithUnit', 
                   [param('ns3::Time::Unit const', 'unit', default_value='::ns3::Time::Unit::AUTO')], 
                   is_const=True)
    ## nstime.h (module 'core'): int ns3::Time::Compare(ns3::Time const & o) const [member function]
    cls.add_method('Compare', 
                   'int', 
                   [param('ns3::Time const &', 'o')], 
                   is_const=True)
    ## nstime.h (module 'core'): static ns3::Time ns3::Time::From(ns3::int64x64_t const & value) [member function]
    cls.add_method('From', 
                   'ns3::Time', 
                   [param('ns3::int64x64_t const &', 'value')], 
                   is_static=True)
    ## nstime.h (module 'core'): static ns3::Time ns3::Time::From(ns3::int64x64_t const & value, ns3::Time::Unit unit) [member function]
    cls.add_method('From', 
                   'ns3::Time', 
                   [param('ns3::int64x64_t const &', 'value'), param('ns3::Time::Unit', 'unit')], 
                   is_static=True)
    ## nstime.h (module 'core'): static ns3::Time ns3::Time::FromDouble(double value, ns3::Time::Unit unit) [member function]
    cls.add_method('FromDouble', 
                   'ns3::Time', 
                   [param('double', 'value'), param('ns3::Time::Unit', 'unit')], 
                   is_static=True)
    ## nstime.h (module 'core'): static ns3::Time ns3::Time::FromInteger(uint64_t value, ns3::Time::Unit unit) [member function]
    cls.add_method('FromInteger', 
                   'ns3::Time', 
                   [param('uint64_t', 'value'), param('ns3::Time::Unit', 'unit')], 
                   is_static=True)
    ## nstime.h (module 'core'): double ns3::Time::GetDays() const [member function]
    cls.add_method('GetDays', 
                   'double', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): double ns3::Time::GetDouble() const [member function]
    cls.add_method('GetDouble', 
                   'double', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): int64_t ns3::Time::GetFemtoSeconds() const [member function]
    cls.add_method('GetFemtoSeconds', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): double ns3::Time::GetHours() const [member function]
    cls.add_method('GetHours', 
                   'double', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): int64_t ns3::Time::GetInteger() const [member function]
    cls.add_method('GetInteger', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): int64_t ns3::Time::GetMicroSeconds() const [member function]
    cls.add_method('GetMicroSeconds', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): int64_t ns3::Time::GetMilliSeconds() const [member function]
    cls.add_method('GetMilliSeconds', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): double ns3::Time::GetMinutes() const [member function]
    cls.add_method('GetMinutes', 
                   'double', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): int64_t ns3::Time::GetNanoSeconds() const [member function]
    cls.add_method('GetNanoSeconds', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): int64_t ns3::Time::GetPicoSeconds() const [member function]
    cls.add_method('GetPicoSeconds', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): static ns3::Time::Unit ns3::Time::GetResolution() [member function]
    cls.add_method('GetResolution', 
                   'ns3::Time::Unit', 
                   [], 
                   is_static=True)
    ## nstime.h (module 'core'): double ns3::Time::GetSeconds() const [member function]
    cls.add_method('GetSeconds', 
                   'double', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): int64_t ns3::Time::GetTimeStep() const [member function]
    cls.add_method('GetTimeStep', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): double ns3::Time::GetYears() const [member function]
    cls.add_method('GetYears', 
                   'double', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): bool ns3::Time::IsNegative() const [member function]
    cls.add_method('IsNegative', 
                   'bool', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): bool ns3::Time::IsPositive() const [member function]
    cls.add_method('IsPositive', 
                   'bool', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): bool ns3::Time::IsStrictlyNegative() const [member function]
    cls.add_method('IsStrictlyNegative', 
                   'bool', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): bool ns3::Time::IsStrictlyPositive() const [member function]
    cls.add_method('IsStrictlyPositive', 
                   'bool', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): bool ns3::Time::IsZero() const [member function]
    cls.add_method('IsZero', 
                   'bool', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): static ns3::Time ns3::Time::Max() [member function]
    cls.add_method('Max', 
                   'ns3::Time', 
                   [], 
                   is_static=True)
    ## nstime.h (module 'core'): static ns3::Time ns3::Time::Min() [member function]
    cls.add_method('Min', 
                   'ns3::Time', 
                   [], 
                   is_static=True)
    ## nstime.h (module 'core'): ns3::Time ns3::Time::RoundTo(ns3::Time::Unit unit) const [member function]
    cls.add_method('RoundTo', 
                   'ns3::Time', 
                   [param('ns3::Time::Unit', 'unit')], 
                   is_const=True)
    ## nstime.h (module 'core'): static void ns3::Time::SetResolution(ns3::Time::Unit resolution) [member function]
    cls.add_method('SetResolution', 
                   'void', 
                   [param('ns3::Time::Unit', 'resolution')], 
                   is_static=True)
    ## nstime.h (module 'core'): static bool ns3::Time::StaticInit() [member function]
    cls.add_method('StaticInit', 
                   'bool', 
                   [], 
                   is_static=True)
    ## nstime.h (module 'core'): ns3::int64x64_t ns3::Time::To(ns3::Time::Unit unit) const [member function]
    cls.add_method('To', 
                   'ns3::int64x64_t', 
                   [param('ns3::Time::Unit', 'unit')], 
                   is_const=True)
    ## nstime.h (module 'core'): double ns3::Time::ToDouble(ns3::Time::Unit unit) const [member function]
    cls.add_method('ToDouble', 
                   'double', 
                   [param('ns3::Time::Unit', 'unit')], 
                   is_const=True)
    ## nstime.h (module 'core'): int64_t ns3::Time::ToInteger(ns3::Time::Unit unit) const [member function]
    cls.add_method('ToInteger', 
                   'int64_t', 
                   [param('ns3::Time::Unit', 'unit')], 
                   is_const=True)
    return

def register_Ns3TimeWithUnit_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## nstime.h (module 'core'): ns3::TimeWithUnit::TimeWithUnit(ns3::TimeWithUnit const & arg0) [constructor]
    cls.add_constructor([param('ns3::TimeWithUnit const &', 'arg0')])
    ## nstime.h (module 'core'): ns3::TimeWithUnit::TimeWithUnit(ns3::Time const time, ns3::Time::Unit const unit) [constructor]
    cls.add_constructor([param('ns3::Time const', 'time'), param('ns3::Time::Unit const', 'unit')])
    return

def register_Ns3TypeId_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_output_stream_operator()
    cls.add_binary_comparison_operator('<')
    ## type-id.h (module 'core'): ns3::TypeId::TypeId(std::string const & name) [constructor]
    cls.add_constructor([param('std::string const &', 'name')])
    ## type-id.h (module 'core'): ns3::TypeId::TypeId() [constructor]
    cls.add_constructor([])
    ## type-id.h (module 'core'): ns3::TypeId::TypeId(ns3::TypeId const & o) [constructor]
    cls.add_constructor([param('ns3::TypeId const &', 'o')])
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeId::AddAttribute(std::string name, std::string help, ns3::AttributeValue const & initialValue, ns3::Ptr<const ns3::AttributeAccessor> accessor, ns3::Ptr<const ns3::AttributeChecker> checker, ns3::TypeId::SupportLevel supportLevel=::ns3::TypeId::SupportLevel::SUPPORTED, std::string const & supportMsg="") [member function]
    cls.add_method('AddAttribute', 
                   'ns3::TypeId', 
                   [param('std::string', 'name'), param('std::string', 'help'), param('ns3::AttributeValue const &', 'initialValue'), param('ns3::Ptr< ns3::AttributeAccessor const >', 'accessor'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker'), param('ns3::TypeId::SupportLevel', 'supportLevel', default_value='::ns3::TypeId::SupportLevel::SUPPORTED'), param('std::string const &', 'supportMsg', default_value='""')])
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeId::AddAttribute(std::string name, std::string help, uint32_t flags, ns3::AttributeValue const & initialValue, ns3::Ptr<const ns3::AttributeAccessor> accessor, ns3::Ptr<const ns3::AttributeChecker> checker, ns3::TypeId::SupportLevel supportLevel=::ns3::TypeId::SupportLevel::SUPPORTED, std::string const & supportMsg="") [member function]
    cls.add_method('AddAttribute', 
                   'ns3::TypeId', 
                   [param('std::string', 'name'), param('std::string', 'help'), param('uint32_t', 'flags'), param('ns3::AttributeValue const &', 'initialValue'), param('ns3::Ptr< ns3::AttributeAccessor const >', 'accessor'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker'), param('ns3::TypeId::SupportLevel', 'supportLevel', default_value='::ns3::TypeId::SupportLevel::SUPPORTED'), param('std::string const &', 'supportMsg', default_value='""')])
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeId::AddTraceSource(std::string name, std::string help, ns3::Ptr<const ns3::TraceSourceAccessor> accessor, std::string callback, ns3::TypeId::SupportLevel supportLevel=::ns3::TypeId::SupportLevel::SUPPORTED, std::string const & supportMsg="") [member function]
    cls.add_method('AddTraceSource', 
                   'ns3::TypeId', 
                   [param('std::string', 'name'), param('std::string', 'help'), param('ns3::Ptr< ns3::TraceSourceAccessor const >', 'accessor'), param('std::string', 'callback'), param('ns3::TypeId::SupportLevel', 'supportLevel', default_value='::ns3::TypeId::SupportLevel::SUPPORTED'), param('std::string const &', 'supportMsg', default_value='""')])
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation ns3::TypeId::GetAttribute(std::size_t i) const [member function]
    cls.add_method('GetAttribute', 
                   'ns3::TypeId::AttributeInformation', 
                   [param('std::size_t', 'i')], 
                   is_const=True)
    ## type-id.h (module 'core'): std::string ns3::TypeId::GetAttributeFullName(std::size_t i) const [member function]
    cls.add_method('GetAttributeFullName', 
                   'std::string', 
                   [param('std::size_t', 'i')], 
                   is_const=True)
    ## type-id.h (module 'core'): std::size_t ns3::TypeId::GetAttributeN() const [member function]
    cls.add_method('GetAttributeN', 
                   'std::size_t', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): ns3::Callback<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> ns3::TypeId::GetConstructor() const [member function]
    cls.add_method('GetConstructor', 
                   'ns3::Callback< ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): std::string ns3::TypeId::GetGroupName() const [member function]
    cls.add_method('GetGroupName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): ns3::TypeId::hash_t ns3::TypeId::GetHash() const [member function]
    cls.add_method('GetHash', 
                   'ns3::TypeId::hash_t', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): std::string ns3::TypeId::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeId::GetParent() const [member function]
    cls.add_method('GetParent', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): static ns3::TypeId ns3::TypeId::GetRegistered(uint16_t i) [member function]
    cls.add_method('GetRegistered', 
                   'ns3::TypeId', 
                   [param('uint16_t', 'i')], 
                   is_static=True)
    ## type-id.h (module 'core'): static uint16_t ns3::TypeId::GetRegisteredN() [member function]
    cls.add_method('GetRegisteredN', 
                   'uint16_t', 
                   [], 
                   is_static=True)
    ## type-id.h (module 'core'): std::size_t ns3::TypeId::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'std::size_t', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): ns3::TypeId::TraceSourceInformation ns3::TypeId::GetTraceSource(std::size_t i) const [member function]
    cls.add_method('GetTraceSource', 
                   'ns3::TypeId::TraceSourceInformation', 
                   [param('std::size_t', 'i')], 
                   is_const=True)
    ## type-id.h (module 'core'): std::size_t ns3::TypeId::GetTraceSourceN() const [member function]
    cls.add_method('GetTraceSourceN', 
                   'std::size_t', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): uint16_t ns3::TypeId::GetUid() const [member function]
    cls.add_method('GetUid', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): bool ns3::TypeId::HasConstructor() const [member function]
    cls.add_method('HasConstructor', 
                   'bool', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): bool ns3::TypeId::HasParent() const [member function]
    cls.add_method('HasParent', 
                   'bool', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeId::HideFromDocumentation() [member function]
    cls.add_method('HideFromDocumentation', 
                   'ns3::TypeId', 
                   [])
    ## type-id.h (module 'core'): bool ns3::TypeId::IsChildOf(ns3::TypeId other) const [member function]
    cls.add_method('IsChildOf', 
                   'bool', 
                   [param('ns3::TypeId', 'other')], 
                   is_const=True)
    ## type-id.h (module 'core'): bool ns3::TypeId::LookupAttributeByName(std::string name, ns3::TypeId::AttributeInformation * info) const [member function]
    cls.add_method('LookupAttributeByName', 
                   'bool', 
                   [param('std::string', 'name'), param('ns3::TypeId::AttributeInformation *', 'info', transfer_ownership=False)], 
                   is_const=True)
    ## type-id.h (module 'core'): static ns3::TypeId ns3::TypeId::LookupByHash(ns3::TypeId::hash_t hash) [member function]
    cls.add_method('LookupByHash', 
                   'ns3::TypeId', 
                   [param('uint32_t', 'hash')], 
                   is_static=True)
    ## type-id.h (module 'core'): static bool ns3::TypeId::LookupByHashFailSafe(ns3::TypeId::hash_t hash, ns3::TypeId * tid) [member function]
    cls.add_method('LookupByHashFailSafe', 
                   'bool', 
                   [param('uint32_t', 'hash'), param('ns3::TypeId *', 'tid')], 
                   is_static=True)
    ## type-id.h (module 'core'): static ns3::TypeId ns3::TypeId::LookupByName(std::string name) [member function]
    cls.add_method('LookupByName', 
                   'ns3::TypeId', 
                   [param('std::string', 'name')], 
                   is_static=True)
    ## type-id.h (module 'core'): ns3::Ptr<const ns3::TraceSourceAccessor> ns3::TypeId::LookupTraceSourceByName(std::string name) const [member function]
    cls.add_method('LookupTraceSourceByName', 
                   'ns3::Ptr< ns3::TraceSourceAccessor const >', 
                   [param('std::string', 'name')], 
                   is_const=True)
    ## type-id.h (module 'core'): ns3::Ptr<const ns3::TraceSourceAccessor> ns3::TypeId::LookupTraceSourceByName(std::string name, ns3::TypeId::TraceSourceInformation * info) const [member function]
    cls.add_method('LookupTraceSourceByName', 
                   'ns3::Ptr< ns3::TraceSourceAccessor const >', 
                   [param('std::string', 'name'), param('ns3::TypeId::TraceSourceInformation *', 'info')], 
                   is_const=True)
    ## type-id.h (module 'core'): bool ns3::TypeId::MustHideFromDocumentation() const [member function]
    cls.add_method('MustHideFromDocumentation', 
                   'bool', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): bool ns3::TypeId::SetAttributeInitialValue(std::size_t i, ns3::Ptr<const ns3::AttributeValue> initialValue) [member function]
    cls.add_method('SetAttributeInitialValue', 
                   'bool', 
                   [param('std::size_t', 'i'), param('ns3::Ptr< ns3::AttributeValue const >', 'initialValue')])
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeId::SetGroupName(std::string groupName) [member function]
    cls.add_method('SetGroupName', 
                   'ns3::TypeId', 
                   [param('std::string', 'groupName')])
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeId::SetParent(ns3::TypeId tid) [member function]
    cls.add_method('SetParent', 
                   'ns3::TypeId', 
                   [param('ns3::TypeId', 'tid')])
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeId::SetSize(std::size_t size) [member function]
    cls.add_method('SetSize', 
                   'ns3::TypeId', 
                   [param('std::size_t', 'size')])
    ## type-id.h (module 'core'): void ns3::TypeId::SetUid(uint16_t uid) [member function]
    cls.add_method('SetUid', 
                   'void', 
                   [param('uint16_t', 'uid')])
    return

def register_Ns3TypeIdAttributeInformation_methods(root_module, cls):
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation::AttributeInformation() [constructor]
    cls.add_constructor([])
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation::AttributeInformation(ns3::TypeId::AttributeInformation const & arg0) [constructor]
    cls.add_constructor([param('ns3::TypeId::AttributeInformation const &', 'arg0')])
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation::accessor [variable]
    cls.add_instance_attribute('accessor', 'ns3::Ptr< ns3::AttributeAccessor const >', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation::checker [variable]
    cls.add_instance_attribute('checker', 'ns3::Ptr< ns3::AttributeChecker const >', is_const=False)
    cls.add_instance_attribute('flags', 'uint32_t', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation::help [variable]
    cls.add_instance_attribute('help', 'std::string', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation::initialValue [variable]
    cls.add_instance_attribute('initialValue', 'ns3::Ptr< ns3::AttributeValue const >', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation::name [variable]
    cls.add_instance_attribute('name', 'std::string', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation::originalInitialValue [variable]
    cls.add_instance_attribute('originalInitialValue', 'ns3::Ptr< ns3::AttributeValue const >', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation::supportLevel [variable]
    cls.add_instance_attribute('supportLevel', 'ns3::TypeId::SupportLevel', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::AttributeInformation::supportMsg [variable]
    cls.add_instance_attribute('supportMsg', 'std::string', is_const=False)
    return

def register_Ns3TypeIdTraceSourceInformation_methods(root_module, cls):
    ## type-id.h (module 'core'): ns3::TypeId::TraceSourceInformation::TraceSourceInformation() [constructor]
    cls.add_constructor([])
    ## type-id.h (module 'core'): ns3::TypeId::TraceSourceInformation::TraceSourceInformation(ns3::TypeId::TraceSourceInformation const & arg0) [constructor]
    cls.add_constructor([param('ns3::TypeId::TraceSourceInformation const &', 'arg0')])
    ## type-id.h (module 'core'): ns3::TypeId::TraceSourceInformation::accessor [variable]
    cls.add_instance_attribute('accessor', 'ns3::Ptr< ns3::TraceSourceAccessor const >', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::TraceSourceInformation::callback [variable]
    cls.add_instance_attribute('callback', 'std::string', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::TraceSourceInformation::help [variable]
    cls.add_instance_attribute('help', 'std::string', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::TraceSourceInformation::name [variable]
    cls.add_instance_attribute('name', 'std::string', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::TraceSourceInformation::supportLevel [variable]
    cls.add_instance_attribute('supportLevel', 'ns3::TypeId::SupportLevel', is_const=False)
    ## type-id.h (module 'core'): ns3::TypeId::TraceSourceInformation::supportMsg [variable]
    cls.add_instance_attribute('supportMsg', 'std::string', is_const=False)
    return

def register_Ns3WifiAc_methods(root_module, cls):
    ## qos-utils.h (module 'wifi'): ns3::WifiAc::WifiAc(ns3::WifiAc const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiAc const &', 'arg0')])
    ## qos-utils.h (module 'wifi'): ns3::WifiAc::WifiAc(uint8_t lowTid, uint8_t highTid) [constructor]
    cls.add_constructor([param('uint8_t', 'lowTid'), param('uint8_t', 'highTid')])
    ## qos-utils.h (module 'wifi'): uint8_t ns3::WifiAc::GetHighTid() const [member function]
    cls.add_method('GetHighTid', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## qos-utils.h (module 'wifi'): uint8_t ns3::WifiAc::GetLowTid() const [member function]
    cls.add_method('GetLowTid', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## qos-utils.h (module 'wifi'): uint8_t ns3::WifiAc::GetOtherTid(uint8_t tid) const [member function]
    cls.add_method('GetOtherTid', 
                   'uint8_t', 
                   [param('uint8_t', 'tid')], 
                   is_const=True)
    return

def register_Ns3WifiAddressHash_methods(root_module, cls):
    ## qos-utils.h (module 'wifi'): ns3::WifiAddressHash::WifiAddressHash() [constructor]
    cls.add_constructor([])
    ## qos-utils.h (module 'wifi'): ns3::WifiAddressHash::WifiAddressHash(ns3::WifiAddressHash const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiAddressHash const &', 'arg0')])
    ## qos-utils.h (module 'wifi'): std::size_t ns3::WifiAddressHash::operator()(ns3::Mac48Address const & address) const [member operator]
    cls.add_method('operator()', 
                   'std::size_t', 
                   [param('ns3::Mac48Address const &', 'address')], 
                   custom_name='__call__', is_const=True)
    return

def register_Ns3WifiAddressTidHash_methods(root_module, cls):
    ## qos-utils.h (module 'wifi'): ns3::WifiAddressTidHash::WifiAddressTidHash() [constructor]
    cls.add_constructor([])
    ## qos-utils.h (module 'wifi'): ns3::WifiAddressTidHash::WifiAddressTidHash(ns3::WifiAddressTidHash const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiAddressTidHash const &', 'arg0')])
    ## qos-utils.h (module 'wifi'): std::size_t ns3::WifiAddressTidHash::operator()(ns3::WifiAddressTidPair const & addressTidPair) const [member operator]
    cls.add_method('operator()', 
                   'std::size_t', 
                   [param('ns3::WifiAddressTidPair const &', 'addressTidPair')], 
                   custom_name='__call__', is_const=True)
    return

def register_Ns3WifiMode_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<')
    cls.add_output_stream_operator()
    ## wifi-mode.h (module 'wifi'): ns3::WifiMode::WifiMode(ns3::WifiMode const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiMode const &', 'arg0')])
    ## wifi-mode.h (module 'wifi'): ns3::WifiMode::WifiMode() [constructor]
    cls.add_constructor([])
    ## wifi-mode.h (module 'wifi'): ns3::WifiMode::WifiMode(std::string name) [constructor]
    cls.add_constructor([param('std::string', 'name')])
    ## wifi-mode.h (module 'wifi'): ns3::WifiCodeRate ns3::WifiMode::GetCodeRate() const [member function]
    cls.add_method('GetCodeRate', 
                   'ns3::WifiCodeRate', 
                   [], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): uint16_t ns3::WifiMode::GetConstellationSize() const [member function]
    cls.add_method('GetConstellationSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): uint64_t ns3::WifiMode::GetDataRate(uint16_t channelWidth, uint16_t guardInterval, uint8_t nss) const [member function]
    cls.add_method('GetDataRate', 
                   'uint64_t', 
                   [param('uint16_t', 'channelWidth'), param('uint16_t', 'guardInterval'), param('uint8_t', 'nss')], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): uint64_t ns3::WifiMode::GetDataRate(ns3::WifiTxVector const & txVector, uint16_t staId=65535) const [member function]
    cls.add_method('GetDataRate', 
                   'uint64_t', 
                   [param('ns3::WifiTxVector const &', 'txVector'), param('uint16_t', 'staId', default_value='65535')], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): uint64_t ns3::WifiMode::GetDataRate(uint16_t channelWidth) const [member function]
    cls.add_method('GetDataRate', 
                   'uint64_t', 
                   [param('uint16_t', 'channelWidth')], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): uint8_t ns3::WifiMode::GetMcsValue() const [member function]
    cls.add_method('GetMcsValue', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): ns3::WifiModulationClass ns3::WifiMode::GetModulationClass() const [member function]
    cls.add_method('GetModulationClass', 
                   'ns3::WifiModulationClass', 
                   [], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): uint64_t ns3::WifiMode::GetNonHtReferenceRate() const [member function]
    cls.add_method('GetNonHtReferenceRate', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): uint64_t ns3::WifiMode::GetPhyRate(uint16_t channelWidth, uint16_t guardInterval, uint8_t nss) const [member function]
    cls.add_method('GetPhyRate', 
                   'uint64_t', 
                   [param('uint16_t', 'channelWidth'), param('uint16_t', 'guardInterval'), param('uint8_t', 'nss')], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): uint64_t ns3::WifiMode::GetPhyRate(ns3::WifiTxVector const & txVector, uint16_t staId=65535) const [member function]
    cls.add_method('GetPhyRate', 
                   'uint64_t', 
                   [param('ns3::WifiTxVector const &', 'txVector'), param('uint16_t', 'staId', default_value='65535')], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): uint64_t ns3::WifiMode::GetPhyRate(uint16_t channelWidth) const [member function]
    cls.add_method('GetPhyRate', 
                   'uint64_t', 
                   [param('uint16_t', 'channelWidth')], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): uint32_t ns3::WifiMode::GetUid() const [member function]
    cls.add_method('GetUid', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): std::string ns3::WifiMode::GetUniqueName() const [member function]
    cls.add_method('GetUniqueName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): bool ns3::WifiMode::IsAllowed(uint16_t channelWidth, uint8_t nss) const [member function]
    cls.add_method('IsAllowed', 
                   'bool', 
                   [param('uint16_t', 'channelWidth'), param('uint8_t', 'nss')], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): bool ns3::WifiMode::IsAllowed(ns3::WifiTxVector const & txVector) const [member function]
    cls.add_method('IsAllowed', 
                   'bool', 
                   [param('ns3::WifiTxVector const &', 'txVector')], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): bool ns3::WifiMode::IsHigherCodeRate(ns3::WifiMode mode) const [member function]
    cls.add_method('IsHigherCodeRate', 
                   'bool', 
                   [param('ns3::WifiMode', 'mode')], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): bool ns3::WifiMode::IsHigherDataRate(ns3::WifiMode mode) const [member function]
    cls.add_method('IsHigherDataRate', 
                   'bool', 
                   [param('ns3::WifiMode', 'mode')], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): bool ns3::WifiMode::IsMandatory() const [member function]
    cls.add_method('IsMandatory', 
                   'bool', 
                   [], 
                   is_const=True)
    return

def register_Ns3WifiModeFactory_methods(root_module, cls):
    ## wifi-mode.h (module 'wifi'): ns3::WifiModeFactory::WifiModeFactory(ns3::WifiModeFactory const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiModeFactory const &', 'arg0')])
    ## wifi-mode.h (module 'wifi'): static ns3::WifiMode ns3::WifiModeFactory::CreateWifiMcs(std::string uniqueName, uint8_t mcsValue, ns3::WifiModulationClass modClass, bool isMandatory, ns3::WifiModeFactory::CodeRateCallback codeRateCallback, ns3::WifiModeFactory::ConstellationSizeCallback constellationSizeCallback, ns3::WifiModeFactory::PhyRateCallback phyRateCallback, ns3::WifiModeFactory::DataRateCallback dataRateCallback, ns3::WifiModeFactory::NonHtReferenceRateCallback nonHtReferenceRateCallback, ns3::WifiModeFactory::AllowedCallback isAllowedCallback) [member function]
    cls.add_method('CreateWifiMcs', 
                   'ns3::WifiMode', 
                   [param('std::string', 'uniqueName'), param('uint8_t', 'mcsValue'), param('ns3::WifiModulationClass', 'modClass'), param('bool', 'isMandatory'), param('ns3::Callback< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'codeRateCallback'), param('ns3::Callback< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'constellationSizeCallback'), param('ns3::Callback< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'phyRateCallback'), param('ns3::Callback< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'dataRateCallback'), param('ns3::Callback< unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'nonHtReferenceRateCallback'), param('ns3::Callback< bool, ns3::WifiTxVector const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'isAllowedCallback')], 
                   is_static=True)
    ## wifi-mode.h (module 'wifi'): static ns3::WifiMode ns3::WifiModeFactory::CreateWifiMode(std::string uniqueName, ns3::WifiModulationClass modClass, bool isMandatory, ns3::WifiModeFactory::CodeRateCallback codeRateCallback, ns3::WifiModeFactory::ConstellationSizeCallback constellationSizeCallback, ns3::WifiModeFactory::PhyRateCallback phyRateCallback, ns3::WifiModeFactory::DataRateCallback dataRateCallback, ns3::WifiModeFactory::AllowedCallback isAllowedCallback) [member function]
    cls.add_method('CreateWifiMode', 
                   'ns3::WifiMode', 
                   [param('std::string', 'uniqueName'), param('ns3::WifiModulationClass', 'modClass'), param('bool', 'isMandatory'), param('ns3::Callback< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'codeRateCallback'), param('ns3::Callback< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'constellationSizeCallback'), param('ns3::Callback< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'phyRateCallback'), param('ns3::Callback< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'dataRateCallback'), param('ns3::Callback< bool, ns3::WifiTxVector const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'isAllowedCallback')], 
                   is_static=True)
    return

def register_Ns3WifiRemoteStation_methods(root_module, cls):
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStation::WifiRemoteStation() [constructor]
    cls.add_constructor([])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStation::WifiRemoteStation(ns3::WifiRemoteStation const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiRemoteStation const &', 'arg0')])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStation::m_rssiAndUpdateTimePair [variable]
    cls.add_instance_attribute('m_rssiAndUpdateTimePair', 'std::pair< double, ns3::Time >', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStation::m_state [variable]
    cls.add_instance_attribute('m_state', 'ns3::WifiRemoteStationState *', is_const=False)
    return

def register_Ns3WifiRemoteStationInfo_methods(root_module, cls):
    ## wifi-remote-station-info.h (module 'wifi'): ns3::WifiRemoteStationInfo::WifiRemoteStationInfo(ns3::WifiRemoteStationInfo const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiRemoteStationInfo const &', 'arg0')])
    ## wifi-remote-station-info.h (module 'wifi'): ns3::WifiRemoteStationInfo::WifiRemoteStationInfo() [constructor]
    cls.add_constructor([])
    ## wifi-remote-station-info.h (module 'wifi'): double ns3::WifiRemoteStationInfo::GetFrameErrorRate() const [member function]
    cls.add_method('GetFrameErrorRate', 
                   'double', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-info.h (module 'wifi'): void ns3::WifiRemoteStationInfo::NotifyTxFailed() [member function]
    cls.add_method('NotifyTxFailed', 
                   'void', 
                   [])
    ## wifi-remote-station-info.h (module 'wifi'): void ns3::WifiRemoteStationInfo::NotifyTxSuccess(uint32_t retryCounter) [member function]
    cls.add_method('NotifyTxSuccess', 
                   'void', 
                   [param('uint32_t', 'retryCounter')])
    return

def register_Ns3WifiRemoteStationState_methods(root_module, cls):
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::WifiRemoteStationState() [constructor]
    cls.add_constructor([])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::WifiRemoteStationState(ns3::WifiRemoteStationState const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiRemoteStationState const &', 'arg0')])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_address [variable]
    cls.add_instance_attribute('m_address', 'ns3::Mac48Address', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_aggregation [variable]
    cls.add_instance_attribute('m_aggregation', 'bool', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_aid [variable]
    cls.add_instance_attribute('m_aid', 'uint16_t', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_channelWidth [variable]
    cls.add_instance_attribute('m_channelWidth', 'uint16_t', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_dsssSupported [variable]
    cls.add_instance_attribute('m_dsssSupported', 'bool', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_erpOfdmSupported [variable]
    cls.add_instance_attribute('m_erpOfdmSupported', 'bool', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_guardInterval [variable]
    cls.add_instance_attribute('m_guardInterval', 'uint16_t', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_heCapabilities [variable]
    cls.add_instance_attribute('m_heCapabilities', 'ns3::Ptr< ns3::HeCapabilities const >', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_htCapabilities [variable]
    cls.add_instance_attribute('m_htCapabilities', 'ns3::Ptr< ns3::HtCapabilities const >', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_info [variable]
    cls.add_instance_attribute('m_info', 'ns3::WifiRemoteStationInfo', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_ness [variable]
    cls.add_instance_attribute('m_ness', 'uint8_t', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_ofdmSupported [variable]
    cls.add_instance_attribute('m_ofdmSupported', 'bool', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_operationalMcsSet [variable]
    cls.add_instance_attribute('m_operationalMcsSet', 'ns3::WifiModeList', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_operationalRateSet [variable]
    cls.add_instance_attribute('m_operationalRateSet', 'ns3::WifiModeList', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_qosSupported [variable]
    cls.add_instance_attribute('m_qosSupported', 'bool', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_shortPreamble [variable]
    cls.add_instance_attribute('m_shortPreamble', 'bool', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_shortSlotTime [variable]
    cls.add_instance_attribute('m_shortSlotTime', 'bool', is_const=False)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationState::m_vhtCapabilities [variable]
    cls.add_instance_attribute('m_vhtCapabilities', 'ns3::Ptr< ns3::VhtCapabilities const >', is_const=False)
    return

def register_Ns3Empty_methods(root_module, cls):
    ## empty.h (module 'core'): ns3::empty::empty() [constructor]
    cls.add_constructor([])
    ## empty.h (module 'core'): ns3::empty::empty(ns3::empty const & arg0) [constructor]
    cls.add_constructor([param('ns3::empty const &', 'arg0')])
    return

def register_Ns3Int64x64_t_methods(root_module, cls):
    cls.add_binary_numeric_operator('*', root_module['ns3::Time'], root_module['ns3::int64x64_t'], param('ns3::Time const &', 'right'))
    cls.add_binary_numeric_operator('+', root_module['ns3::int64x64_t'], root_module['ns3::int64x64_t'], param('ns3::int64x64_t const &', 'right'))
    cls.add_binary_numeric_operator('-', root_module['ns3::int64x64_t'], root_module['ns3::int64x64_t'], param('ns3::int64x64_t const &', 'right'))
    cls.add_binary_numeric_operator('*', root_module['ns3::int64x64_t'], root_module['ns3::int64x64_t'], param('ns3::int64x64_t const &', 'right'))
    cls.add_binary_numeric_operator('/', root_module['ns3::int64x64_t'], root_module['ns3::int64x64_t'], param('ns3::int64x64_t const &', 'right'))
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<=')
    cls.add_binary_comparison_operator('>=')
    cls.add_output_stream_operator()
    cls.add_unary_numeric_operator('-')
    cls.add_inplace_numeric_operator('/=', param('ns3::int64x64_t const &', 'right'))
    cls.add_inplace_numeric_operator('*=', param('ns3::int64x64_t const &', 'right'))
    cls.add_inplace_numeric_operator('-=', param('ns3::int64x64_t const &', 'right'))
    cls.add_inplace_numeric_operator('+=', param('ns3::int64x64_t const &', 'right'))
    cls.add_binary_comparison_operator('>')
    cls.add_binary_comparison_operator('<')
    cls.add_binary_comparison_operator('==')
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t() [constructor]
    cls.add_constructor([])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(double const value) [constructor]
    cls.add_constructor([param('double const', 'value')])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(long double const value) [constructor]
    cls.add_constructor([param('long double const', 'value')])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(int const v) [constructor]
    cls.add_constructor([param('int const', 'v')])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(long int const v) [constructor]
    cls.add_constructor([param('long int const', 'v')])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(long long int const v) [constructor]
    cls.add_constructor([param('long long int const', 'v')])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(unsigned int const v) [constructor]
    cls.add_constructor([param('unsigned int const', 'v')])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(long unsigned int const v) [constructor]
    cls.add_constructor([param('long unsigned int const', 'v')])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(long long unsigned int const v) [constructor]
    cls.add_constructor([param('long long unsigned int const', 'v')])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(int128_t const v) [constructor]
    cls.add_constructor([param('int128_t const', 'v')])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(int64_t const hi, uint64_t const lo) [constructor]
    cls.add_constructor([param('int64_t const', 'hi'), param('uint64_t const', 'lo')])
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::int64x64_t(ns3::int64x64_t const & o) [constructor]
    cls.add_constructor([param('ns3::int64x64_t const &', 'o')])
    ## int64x64-128.h (module 'core'): double ns3::int64x64_t::GetDouble() const [member function]
    cls.add_method('GetDouble', 
                   'double', 
                   [], 
                   is_const=True)
    ## int64x64-128.h (module 'core'): int64_t ns3::int64x64_t::GetHigh() const [member function]
    cls.add_method('GetHigh', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## int64x64-128.h (module 'core'): int64_t ns3::int64x64_t::GetInt() const [member function]
    cls.add_method('GetInt', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## int64x64-128.h (module 'core'): uint64_t ns3::int64x64_t::GetLow() const [member function]
    cls.add_method('GetLow', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## int64x64-128.h (module 'core'): static ns3::int64x64_t ns3::int64x64_t::Invert(uint64_t const v) [member function]
    cls.add_method('Invert', 
                   'ns3::int64x64_t', 
                   [param('uint64_t const', 'v')], 
                   is_static=True)
    ## int64x64-128.h (module 'core'): void ns3::int64x64_t::MulByInvert(ns3::int64x64_t const & o) [member function]
    cls.add_method('MulByInvert', 
                   'void', 
                   [param('ns3::int64x64_t const &', 'o')])
    ## int64x64-128.h (module 'core'): int64_t ns3::int64x64_t::Round() const [member function]
    cls.add_method('Round', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::implementation [variable]
    cls.add_static_attribute('implementation', 'ns3::int64x64_t::impl_type const', is_const=True)
    return

def register_Ns3Chunk_methods(root_module, cls):
    ## chunk.h (module 'network'): ns3::Chunk::Chunk() [constructor]
    cls.add_constructor([])
    ## chunk.h (module 'network'): ns3::Chunk::Chunk(ns3::Chunk const & arg0) [constructor]
    cls.add_constructor([param('ns3::Chunk const &', 'arg0')])
    ## chunk.h (module 'network'): uint32_t ns3::Chunk::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_pure_virtual=True, is_virtual=True)
    ## chunk.h (module 'network'): uint32_t ns3::Chunk::Deserialize(ns3::Buffer::Iterator start, ns3::Buffer::Iterator end) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('ns3::Buffer::Iterator', 'end')], 
                   is_virtual=True)
    ## chunk.h (module 'network'): static ns3::TypeId ns3::Chunk::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## chunk.h (module 'network'): void ns3::Chunk::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3Header_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## header.h (module 'network'): ns3::Header::Header() [constructor]
    cls.add_constructor([])
    ## header.h (module 'network'): ns3::Header::Header(ns3::Header const & arg0) [constructor]
    cls.add_constructor([param('ns3::Header const &', 'arg0')])
    ## header.h (module 'network'): uint32_t ns3::Header::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_pure_virtual=True, is_virtual=True)
    ## header.h (module 'network'): uint32_t ns3::Header::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## header.h (module 'network'): static ns3::TypeId ns3::Header::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## header.h (module 'network'): void ns3::Header::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## header.h (module 'network'): void ns3::Header::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3MgtAddBaRequestHeader_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::MgtAddBaRequestHeader::MgtAddBaRequestHeader(ns3::MgtAddBaRequestHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::MgtAddBaRequestHeader const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): ns3::MgtAddBaRequestHeader::MgtAddBaRequestHeader() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtAddBaRequestHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint16_t ns3::MgtAddBaRequestHeader::GetBufferSize() const [member function]
    cls.add_method('GetBufferSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::TypeId ns3::MgtAddBaRequestHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtAddBaRequestHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint16_t ns3::MgtAddBaRequestHeader::GetStartingSequence() const [member function]
    cls.add_method('GetStartingSequence', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): uint8_t ns3::MgtAddBaRequestHeader::GetTid() const [member function]
    cls.add_method('GetTid', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): uint16_t ns3::MgtAddBaRequestHeader::GetTimeout() const [member function]
    cls.add_method('GetTimeout', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): static ns3::TypeId ns3::MgtAddBaRequestHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mgt-headers.h (module 'wifi'): bool ns3::MgtAddBaRequestHeader::IsAmsduSupported() const [member function]
    cls.add_method('IsAmsduSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): bool ns3::MgtAddBaRequestHeader::IsImmediateBlockAck() const [member function]
    cls.add_method('IsImmediateBlockAck', 
                   'bool', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaRequestHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaRequestHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaRequestHeader::SetAmsduSupport(bool supported) [member function]
    cls.add_method('SetAmsduSupport', 
                   'void', 
                   [param('bool', 'supported')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaRequestHeader::SetBufferSize(uint16_t size) [member function]
    cls.add_method('SetBufferSize', 
                   'void', 
                   [param('uint16_t', 'size')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaRequestHeader::SetDelayedBlockAck() [member function]
    cls.add_method('SetDelayedBlockAck', 
                   'void', 
                   [])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaRequestHeader::SetImmediateBlockAck() [member function]
    cls.add_method('SetImmediateBlockAck', 
                   'void', 
                   [])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaRequestHeader::SetStartingSequence(uint16_t seq) [member function]
    cls.add_method('SetStartingSequence', 
                   'void', 
                   [param('uint16_t', 'seq')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaRequestHeader::SetTid(uint8_t tid) [member function]
    cls.add_method('SetTid', 
                   'void', 
                   [param('uint8_t', 'tid')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaRequestHeader::SetTimeout(uint16_t timeout) [member function]
    cls.add_method('SetTimeout', 
                   'void', 
                   [param('uint16_t', 'timeout')])
    return

def register_Ns3MgtAddBaResponseHeader_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::MgtAddBaResponseHeader::MgtAddBaResponseHeader(ns3::MgtAddBaResponseHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::MgtAddBaResponseHeader const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): ns3::MgtAddBaResponseHeader::MgtAddBaResponseHeader() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtAddBaResponseHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint16_t ns3::MgtAddBaResponseHeader::GetBufferSize() const [member function]
    cls.add_method('GetBufferSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::TypeId ns3::MgtAddBaResponseHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtAddBaResponseHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::StatusCode ns3::MgtAddBaResponseHeader::GetStatusCode() const [member function]
    cls.add_method('GetStatusCode', 
                   'ns3::StatusCode', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): uint8_t ns3::MgtAddBaResponseHeader::GetTid() const [member function]
    cls.add_method('GetTid', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): uint16_t ns3::MgtAddBaResponseHeader::GetTimeout() const [member function]
    cls.add_method('GetTimeout', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): static ns3::TypeId ns3::MgtAddBaResponseHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mgt-headers.h (module 'wifi'): bool ns3::MgtAddBaResponseHeader::IsAmsduSupported() const [member function]
    cls.add_method('IsAmsduSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): bool ns3::MgtAddBaResponseHeader::IsImmediateBlockAck() const [member function]
    cls.add_method('IsImmediateBlockAck', 
                   'bool', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaResponseHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaResponseHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaResponseHeader::SetAmsduSupport(bool supported) [member function]
    cls.add_method('SetAmsduSupport', 
                   'void', 
                   [param('bool', 'supported')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaResponseHeader::SetBufferSize(uint16_t size) [member function]
    cls.add_method('SetBufferSize', 
                   'void', 
                   [param('uint16_t', 'size')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaResponseHeader::SetDelayedBlockAck() [member function]
    cls.add_method('SetDelayedBlockAck', 
                   'void', 
                   [])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaResponseHeader::SetImmediateBlockAck() [member function]
    cls.add_method('SetImmediateBlockAck', 
                   'void', 
                   [])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaResponseHeader::SetStatusCode(ns3::StatusCode code) [member function]
    cls.add_method('SetStatusCode', 
                   'void', 
                   [param('ns3::StatusCode', 'code')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaResponseHeader::SetTid(uint8_t tid) [member function]
    cls.add_method('SetTid', 
                   'void', 
                   [param('uint8_t', 'tid')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAddBaResponseHeader::SetTimeout(uint16_t timeout) [member function]
    cls.add_method('SetTimeout', 
                   'void', 
                   [param('uint16_t', 'timeout')])
    return

def register_Ns3MgtAssocRequestHeader_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::MgtAssocRequestHeader::MgtAssocRequestHeader(ns3::MgtAssocRequestHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::MgtAssocRequestHeader const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): ns3::MgtAssocRequestHeader::MgtAssocRequestHeader() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtAssocRequestHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::CapabilityInformation ns3::MgtAssocRequestHeader::GetCapabilities() const [member function]
    cls.add_method('GetCapabilities', 
                   'ns3::CapabilityInformation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::ExtendedCapabilities ns3::MgtAssocRequestHeader::GetExtendedCapabilities() const [member function]
    cls.add_method('GetExtendedCapabilities', 
                   'ns3::ExtendedCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HeCapabilities ns3::MgtAssocRequestHeader::GetHeCapabilities() const [member function]
    cls.add_method('GetHeCapabilities', 
                   'ns3::HeCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HtCapabilities ns3::MgtAssocRequestHeader::GetHtCapabilities() const [member function]
    cls.add_method('GetHtCapabilities', 
                   'ns3::HtCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::TypeId ns3::MgtAssocRequestHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint16_t ns3::MgtAssocRequestHeader::GetListenInterval() const [member function]
    cls.add_method('GetListenInterval', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtAssocRequestHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::Ssid ns3::MgtAssocRequestHeader::GetSsid() const [member function]
    cls.add_method('GetSsid', 
                   'ns3::Ssid', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::SupportedRates ns3::MgtAssocRequestHeader::GetSupportedRates() const [member function]
    cls.add_method('GetSupportedRates', 
                   'ns3::SupportedRates', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): static ns3::TypeId ns3::MgtAssocRequestHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mgt-headers.h (module 'wifi'): ns3::VhtCapabilities ns3::MgtAssocRequestHeader::GetVhtCapabilities() const [member function]
    cls.add_method('GetVhtCapabilities', 
                   'ns3::VhtCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocRequestHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocRequestHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocRequestHeader::SetCapabilities(ns3::CapabilityInformation capabilities) [member function]
    cls.add_method('SetCapabilities', 
                   'void', 
                   [param('ns3::CapabilityInformation', 'capabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocRequestHeader::SetExtendedCapabilities(ns3::ExtendedCapabilities extendedCapabilities) [member function]
    cls.add_method('SetExtendedCapabilities', 
                   'void', 
                   [param('ns3::ExtendedCapabilities', 'extendedCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocRequestHeader::SetHeCapabilities(ns3::HeCapabilities heCapabilities) [member function]
    cls.add_method('SetHeCapabilities', 
                   'void', 
                   [param('ns3::HeCapabilities', 'heCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocRequestHeader::SetHtCapabilities(ns3::HtCapabilities htCapabilities) [member function]
    cls.add_method('SetHtCapabilities', 
                   'void', 
                   [param('ns3::HtCapabilities', 'htCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocRequestHeader::SetListenInterval(uint16_t interval) [member function]
    cls.add_method('SetListenInterval', 
                   'void', 
                   [param('uint16_t', 'interval')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocRequestHeader::SetSsid(ns3::Ssid ssid) [member function]
    cls.add_method('SetSsid', 
                   'void', 
                   [param('ns3::Ssid', 'ssid')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocRequestHeader::SetSupportedRates(ns3::SupportedRates rates) [member function]
    cls.add_method('SetSupportedRates', 
                   'void', 
                   [param('ns3::SupportedRates', 'rates')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocRequestHeader::SetVhtCapabilities(ns3::VhtCapabilities vhtCapabilities) [member function]
    cls.add_method('SetVhtCapabilities', 
                   'void', 
                   [param('ns3::VhtCapabilities', 'vhtCapabilities')])
    return

def register_Ns3MgtAssocResponseHeader_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::MgtAssocResponseHeader::MgtAssocResponseHeader(ns3::MgtAssocResponseHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::MgtAssocResponseHeader const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): ns3::MgtAssocResponseHeader::MgtAssocResponseHeader() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtAssocResponseHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint16_t ns3::MgtAssocResponseHeader::GetAssociationId() const [member function]
    cls.add_method('GetAssociationId', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::CapabilityInformation ns3::MgtAssocResponseHeader::GetCapabilities() const [member function]
    cls.add_method('GetCapabilities', 
                   'ns3::CapabilityInformation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::EdcaParameterSet ns3::MgtAssocResponseHeader::GetEdcaParameterSet() const [member function]
    cls.add_method('GetEdcaParameterSet', 
                   'ns3::EdcaParameterSet', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::ErpInformation ns3::MgtAssocResponseHeader::GetErpInformation() const [member function]
    cls.add_method('GetErpInformation', 
                   'ns3::ErpInformation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::ExtendedCapabilities ns3::MgtAssocResponseHeader::GetExtendedCapabilities() const [member function]
    cls.add_method('GetExtendedCapabilities', 
                   'ns3::ExtendedCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HeCapabilities ns3::MgtAssocResponseHeader::GetHeCapabilities() const [member function]
    cls.add_method('GetHeCapabilities', 
                   'ns3::HeCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HeOperation ns3::MgtAssocResponseHeader::GetHeOperation() const [member function]
    cls.add_method('GetHeOperation', 
                   'ns3::HeOperation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HtCapabilities ns3::MgtAssocResponseHeader::GetHtCapabilities() const [member function]
    cls.add_method('GetHtCapabilities', 
                   'ns3::HtCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HtOperation ns3::MgtAssocResponseHeader::GetHtOperation() const [member function]
    cls.add_method('GetHtOperation', 
                   'ns3::HtOperation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::TypeId ns3::MgtAssocResponseHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::MuEdcaParameterSet ns3::MgtAssocResponseHeader::GetMuEdcaParameterSet() const [member function]
    cls.add_method('GetMuEdcaParameterSet', 
                   'ns3::MuEdcaParameterSet', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtAssocResponseHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::StatusCode ns3::MgtAssocResponseHeader::GetStatusCode() [member function]
    cls.add_method('GetStatusCode', 
                   'ns3::StatusCode', 
                   [])
    ## mgt-headers.h (module 'wifi'): ns3::SupportedRates ns3::MgtAssocResponseHeader::GetSupportedRates() [member function]
    cls.add_method('GetSupportedRates', 
                   'ns3::SupportedRates', 
                   [])
    ## mgt-headers.h (module 'wifi'): static ns3::TypeId ns3::MgtAssocResponseHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mgt-headers.h (module 'wifi'): ns3::VhtCapabilities ns3::MgtAssocResponseHeader::GetVhtCapabilities() const [member function]
    cls.add_method('GetVhtCapabilities', 
                   'ns3::VhtCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::VhtOperation ns3::MgtAssocResponseHeader::GetVhtOperation() const [member function]
    cls.add_method('GetVhtOperation', 
                   'ns3::VhtOperation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetAssociationId(uint16_t aid) [member function]
    cls.add_method('SetAssociationId', 
                   'void', 
                   [param('uint16_t', 'aid')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetCapabilities(ns3::CapabilityInformation capabilities) [member function]
    cls.add_method('SetCapabilities', 
                   'void', 
                   [param('ns3::CapabilityInformation', 'capabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetEdcaParameterSet(ns3::EdcaParameterSet edcaParameterSet) [member function]
    cls.add_method('SetEdcaParameterSet', 
                   'void', 
                   [param('ns3::EdcaParameterSet', 'edcaParameterSet')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetErpInformation(ns3::ErpInformation erpInformation) [member function]
    cls.add_method('SetErpInformation', 
                   'void', 
                   [param('ns3::ErpInformation', 'erpInformation')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetExtendedCapabilities(ns3::ExtendedCapabilities extendedCapabilities) [member function]
    cls.add_method('SetExtendedCapabilities', 
                   'void', 
                   [param('ns3::ExtendedCapabilities', 'extendedCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetHeCapabilities(ns3::HeCapabilities heCapabilities) [member function]
    cls.add_method('SetHeCapabilities', 
                   'void', 
                   [param('ns3::HeCapabilities', 'heCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetHeOperation(ns3::HeOperation heOperation) [member function]
    cls.add_method('SetHeOperation', 
                   'void', 
                   [param('ns3::HeOperation', 'heOperation')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetHtCapabilities(ns3::HtCapabilities htCapabilities) [member function]
    cls.add_method('SetHtCapabilities', 
                   'void', 
                   [param('ns3::HtCapabilities', 'htCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetHtOperation(ns3::HtOperation htOperation) [member function]
    cls.add_method('SetHtOperation', 
                   'void', 
                   [param('ns3::HtOperation', 'htOperation')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetMuEdcaParameterSet(ns3::MuEdcaParameterSet muEdcaParameterSet) [member function]
    cls.add_method('SetMuEdcaParameterSet', 
                   'void', 
                   [param('ns3::MuEdcaParameterSet', 'muEdcaParameterSet')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetStatusCode(ns3::StatusCode code) [member function]
    cls.add_method('SetStatusCode', 
                   'void', 
                   [param('ns3::StatusCode', 'code')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetSupportedRates(ns3::SupportedRates rates) [member function]
    cls.add_method('SetSupportedRates', 
                   'void', 
                   [param('ns3::SupportedRates', 'rates')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetVhtCapabilities(ns3::VhtCapabilities vhtCapabilities) [member function]
    cls.add_method('SetVhtCapabilities', 
                   'void', 
                   [param('ns3::VhtCapabilities', 'vhtCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtAssocResponseHeader::SetVhtOperation(ns3::VhtOperation vhtOperation) [member function]
    cls.add_method('SetVhtOperation', 
                   'void', 
                   [param('ns3::VhtOperation', 'vhtOperation')])
    return

def register_Ns3MgtDelBaHeader_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::MgtDelBaHeader::MgtDelBaHeader(ns3::MgtDelBaHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::MgtDelBaHeader const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): ns3::MgtDelBaHeader::MgtDelBaHeader() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtDelBaHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::TypeId ns3::MgtDelBaHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtDelBaHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint8_t ns3::MgtDelBaHeader::GetTid() const [member function]
    cls.add_method('GetTid', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): static ns3::TypeId ns3::MgtDelBaHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mgt-headers.h (module 'wifi'): bool ns3::MgtDelBaHeader::IsByOriginator() const [member function]
    cls.add_method('IsByOriginator', 
                   'bool', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtDelBaHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtDelBaHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtDelBaHeader::SetByOriginator() [member function]
    cls.add_method('SetByOriginator', 
                   'void', 
                   [])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtDelBaHeader::SetByRecipient() [member function]
    cls.add_method('SetByRecipient', 
                   'void', 
                   [])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtDelBaHeader::SetTid(uint8_t tid) [member function]
    cls.add_method('SetTid', 
                   'void', 
                   [param('uint8_t', 'tid')])
    return

def register_Ns3MgtProbeRequestHeader_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::MgtProbeRequestHeader::MgtProbeRequestHeader() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): ns3::MgtProbeRequestHeader::MgtProbeRequestHeader(ns3::MgtProbeRequestHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::MgtProbeRequestHeader const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtProbeRequestHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::ExtendedCapabilities ns3::MgtProbeRequestHeader::GetExtendedCapabilities() const [member function]
    cls.add_method('GetExtendedCapabilities', 
                   'ns3::ExtendedCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HeCapabilities ns3::MgtProbeRequestHeader::GetHeCapabilities() const [member function]
    cls.add_method('GetHeCapabilities', 
                   'ns3::HeCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HtCapabilities ns3::MgtProbeRequestHeader::GetHtCapabilities() const [member function]
    cls.add_method('GetHtCapabilities', 
                   'ns3::HtCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::TypeId ns3::MgtProbeRequestHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtProbeRequestHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::Ssid ns3::MgtProbeRequestHeader::GetSsid() const [member function]
    cls.add_method('GetSsid', 
                   'ns3::Ssid', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::SupportedRates ns3::MgtProbeRequestHeader::GetSupportedRates() const [member function]
    cls.add_method('GetSupportedRates', 
                   'ns3::SupportedRates', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): static ns3::TypeId ns3::MgtProbeRequestHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mgt-headers.h (module 'wifi'): ns3::VhtCapabilities ns3::MgtProbeRequestHeader::GetVhtCapabilities() const [member function]
    cls.add_method('GetVhtCapabilities', 
                   'ns3::VhtCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeRequestHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeRequestHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeRequestHeader::SetExtendedCapabilities(ns3::ExtendedCapabilities extendedCapabilities) [member function]
    cls.add_method('SetExtendedCapabilities', 
                   'void', 
                   [param('ns3::ExtendedCapabilities', 'extendedCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeRequestHeader::SetHeCapabilities(ns3::HeCapabilities heCapabilities) [member function]
    cls.add_method('SetHeCapabilities', 
                   'void', 
                   [param('ns3::HeCapabilities', 'heCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeRequestHeader::SetHtCapabilities(ns3::HtCapabilities htCapabilities) [member function]
    cls.add_method('SetHtCapabilities', 
                   'void', 
                   [param('ns3::HtCapabilities', 'htCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeRequestHeader::SetSsid(ns3::Ssid ssid) [member function]
    cls.add_method('SetSsid', 
                   'void', 
                   [param('ns3::Ssid', 'ssid')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeRequestHeader::SetSupportedRates(ns3::SupportedRates rates) [member function]
    cls.add_method('SetSupportedRates', 
                   'void', 
                   [param('ns3::SupportedRates', 'rates')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeRequestHeader::SetVhtCapabilities(ns3::VhtCapabilities vhtCapabilities) [member function]
    cls.add_method('SetVhtCapabilities', 
                   'void', 
                   [param('ns3::VhtCapabilities', 'vhtCapabilities')])
    return

def register_Ns3MgtProbeResponseHeader_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::MgtProbeResponseHeader::MgtProbeResponseHeader(ns3::MgtProbeResponseHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::MgtProbeResponseHeader const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): ns3::MgtProbeResponseHeader::MgtProbeResponseHeader() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtProbeResponseHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint64_t ns3::MgtProbeResponseHeader::GetBeaconIntervalUs() const [member function]
    cls.add_method('GetBeaconIntervalUs', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::CapabilityInformation ns3::MgtProbeResponseHeader::GetCapabilities() const [member function]
    cls.add_method('GetCapabilities', 
                   'ns3::CapabilityInformation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::DsssParameterSet ns3::MgtProbeResponseHeader::GetDsssParameterSet() const [member function]
    cls.add_method('GetDsssParameterSet', 
                   'ns3::DsssParameterSet', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::EdcaParameterSet ns3::MgtProbeResponseHeader::GetEdcaParameterSet() const [member function]
    cls.add_method('GetEdcaParameterSet', 
                   'ns3::EdcaParameterSet', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::ErpInformation ns3::MgtProbeResponseHeader::GetErpInformation() const [member function]
    cls.add_method('GetErpInformation', 
                   'ns3::ErpInformation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::ExtendedCapabilities ns3::MgtProbeResponseHeader::GetExtendedCapabilities() const [member function]
    cls.add_method('GetExtendedCapabilities', 
                   'ns3::ExtendedCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HeCapabilities ns3::MgtProbeResponseHeader::GetHeCapabilities() const [member function]
    cls.add_method('GetHeCapabilities', 
                   'ns3::HeCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HeOperation ns3::MgtProbeResponseHeader::GetHeOperation() const [member function]
    cls.add_method('GetHeOperation', 
                   'ns3::HeOperation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HtCapabilities ns3::MgtProbeResponseHeader::GetHtCapabilities() const [member function]
    cls.add_method('GetHtCapabilities', 
                   'ns3::HtCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HtOperation ns3::MgtProbeResponseHeader::GetHtOperation() const [member function]
    cls.add_method('GetHtOperation', 
                   'ns3::HtOperation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::TypeId ns3::MgtProbeResponseHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::MuEdcaParameterSet ns3::MgtProbeResponseHeader::GetMuEdcaParameterSet() const [member function]
    cls.add_method('GetMuEdcaParameterSet', 
                   'ns3::MuEdcaParameterSet', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtProbeResponseHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::Ssid ns3::MgtProbeResponseHeader::GetSsid() const [member function]
    cls.add_method('GetSsid', 
                   'ns3::Ssid', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::SupportedRates ns3::MgtProbeResponseHeader::GetSupportedRates() const [member function]
    cls.add_method('GetSupportedRates', 
                   'ns3::SupportedRates', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): uint64_t ns3::MgtProbeResponseHeader::GetTimestamp() [member function]
    cls.add_method('GetTimestamp', 
                   'uint64_t', 
                   [])
    ## mgt-headers.h (module 'wifi'): static ns3::TypeId ns3::MgtProbeResponseHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mgt-headers.h (module 'wifi'): ns3::VhtCapabilities ns3::MgtProbeResponseHeader::GetVhtCapabilities() const [member function]
    cls.add_method('GetVhtCapabilities', 
                   'ns3::VhtCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::VhtOperation ns3::MgtProbeResponseHeader::GetVhtOperation() const [member function]
    cls.add_method('GetVhtOperation', 
                   'ns3::VhtOperation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetBeaconIntervalUs(uint64_t us) [member function]
    cls.add_method('SetBeaconIntervalUs', 
                   'void', 
                   [param('uint64_t', 'us')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetCapabilities(ns3::CapabilityInformation capabilities) [member function]
    cls.add_method('SetCapabilities', 
                   'void', 
                   [param('ns3::CapabilityInformation', 'capabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetDsssParameterSet(ns3::DsssParameterSet dsssParameterSet) [member function]
    cls.add_method('SetDsssParameterSet', 
                   'void', 
                   [param('ns3::DsssParameterSet', 'dsssParameterSet')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetEdcaParameterSet(ns3::EdcaParameterSet edcaParameterSet) [member function]
    cls.add_method('SetEdcaParameterSet', 
                   'void', 
                   [param('ns3::EdcaParameterSet', 'edcaParameterSet')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetErpInformation(ns3::ErpInformation erpInformation) [member function]
    cls.add_method('SetErpInformation', 
                   'void', 
                   [param('ns3::ErpInformation', 'erpInformation')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetExtendedCapabilities(ns3::ExtendedCapabilities extendedCapabilities) [member function]
    cls.add_method('SetExtendedCapabilities', 
                   'void', 
                   [param('ns3::ExtendedCapabilities', 'extendedCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetHeCapabilities(ns3::HeCapabilities heCapabilities) [member function]
    cls.add_method('SetHeCapabilities', 
                   'void', 
                   [param('ns3::HeCapabilities', 'heCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetHeOperation(ns3::HeOperation heOperation) [member function]
    cls.add_method('SetHeOperation', 
                   'void', 
                   [param('ns3::HeOperation', 'heOperation')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetHtCapabilities(ns3::HtCapabilities htCapabilities) [member function]
    cls.add_method('SetHtCapabilities', 
                   'void', 
                   [param('ns3::HtCapabilities', 'htCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetHtOperation(ns3::HtOperation htOperation) [member function]
    cls.add_method('SetHtOperation', 
                   'void', 
                   [param('ns3::HtOperation', 'htOperation')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetMuEdcaParameterSet(ns3::MuEdcaParameterSet muEdcaParameterSet) [member function]
    cls.add_method('SetMuEdcaParameterSet', 
                   'void', 
                   [param('ns3::MuEdcaParameterSet', 'muEdcaParameterSet')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetSsid(ns3::Ssid ssid) [member function]
    cls.add_method('SetSsid', 
                   'void', 
                   [param('ns3::Ssid', 'ssid')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetSupportedRates(ns3::SupportedRates rates) [member function]
    cls.add_method('SetSupportedRates', 
                   'void', 
                   [param('ns3::SupportedRates', 'rates')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetVhtCapabilities(ns3::VhtCapabilities vhtCapabilities) [member function]
    cls.add_method('SetVhtCapabilities', 
                   'void', 
                   [param('ns3::VhtCapabilities', 'vhtCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtProbeResponseHeader::SetVhtOperation(ns3::VhtOperation vhtOperation) [member function]
    cls.add_method('SetVhtOperation', 
                   'void', 
                   [param('ns3::VhtOperation', 'vhtOperation')])
    return

def register_Ns3MgtReassocRequestHeader_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::MgtReassocRequestHeader::MgtReassocRequestHeader(ns3::MgtReassocRequestHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::MgtReassocRequestHeader const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): ns3::MgtReassocRequestHeader::MgtReassocRequestHeader() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtReassocRequestHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::CapabilityInformation ns3::MgtReassocRequestHeader::GetCapabilities() const [member function]
    cls.add_method('GetCapabilities', 
                   'ns3::CapabilityInformation', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::ExtendedCapabilities ns3::MgtReassocRequestHeader::GetExtendedCapabilities() const [member function]
    cls.add_method('GetExtendedCapabilities', 
                   'ns3::ExtendedCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HeCapabilities ns3::MgtReassocRequestHeader::GetHeCapabilities() const [member function]
    cls.add_method('GetHeCapabilities', 
                   'ns3::HeCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::HtCapabilities ns3::MgtReassocRequestHeader::GetHtCapabilities() const [member function]
    cls.add_method('GetHtCapabilities', 
                   'ns3::HtCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::TypeId ns3::MgtReassocRequestHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint16_t ns3::MgtReassocRequestHeader::GetListenInterval() const [member function]
    cls.add_method('GetListenInterval', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::MgtReassocRequestHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::Ssid ns3::MgtReassocRequestHeader::GetSsid() const [member function]
    cls.add_method('GetSsid', 
                   'ns3::Ssid', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): ns3::SupportedRates ns3::MgtReassocRequestHeader::GetSupportedRates() const [member function]
    cls.add_method('GetSupportedRates', 
                   'ns3::SupportedRates', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): static ns3::TypeId ns3::MgtReassocRequestHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mgt-headers.h (module 'wifi'): ns3::VhtCapabilities ns3::MgtReassocRequestHeader::GetVhtCapabilities() const [member function]
    cls.add_method('GetVhtCapabilities', 
                   'ns3::VhtCapabilities', 
                   [], 
                   is_const=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::SetCapabilities(ns3::CapabilityInformation capabilities) [member function]
    cls.add_method('SetCapabilities', 
                   'void', 
                   [param('ns3::CapabilityInformation', 'capabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::SetCurrentApAddress(ns3::Mac48Address currentApAddr) [member function]
    cls.add_method('SetCurrentApAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'currentApAddr')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::SetExtendedCapabilities(ns3::ExtendedCapabilities extendedCapabilities) [member function]
    cls.add_method('SetExtendedCapabilities', 
                   'void', 
                   [param('ns3::ExtendedCapabilities', 'extendedCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::SetHeCapabilities(ns3::HeCapabilities heCapabilities) [member function]
    cls.add_method('SetHeCapabilities', 
                   'void', 
                   [param('ns3::HeCapabilities', 'heCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::SetHtCapabilities(ns3::HtCapabilities htCapabilities) [member function]
    cls.add_method('SetHtCapabilities', 
                   'void', 
                   [param('ns3::HtCapabilities', 'htCapabilities')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::SetListenInterval(uint16_t interval) [member function]
    cls.add_method('SetListenInterval', 
                   'void', 
                   [param('uint16_t', 'interval')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::SetSsid(ns3::Ssid ssid) [member function]
    cls.add_method('SetSsid', 
                   'void', 
                   [param('ns3::Ssid', 'ssid')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::SetSupportedRates(ns3::SupportedRates rates) [member function]
    cls.add_method('SetSupportedRates', 
                   'void', 
                   [param('ns3::SupportedRates', 'rates')])
    ## mgt-headers.h (module 'wifi'): void ns3::MgtReassocRequestHeader::SetVhtCapabilities(ns3::VhtCapabilities vhtCapabilities) [member function]
    cls.add_method('SetVhtCapabilities', 
                   'void', 
                   [param('ns3::VhtCapabilities', 'vhtCapabilities')])
    return

def register_Ns3Object_methods(root_module, cls):
    ## object.h (module 'core'): ns3::Object::Object() [constructor]
    cls.add_constructor([])
    ## object.h (module 'core'): void ns3::Object::AggregateObject(ns3::Ptr<ns3::Object> other) [member function]
    cls.add_method('AggregateObject', 
                   'void', 
                   [param('ns3::Ptr< ns3::Object >', 'other')])
    ## object.h (module 'core'): void ns3::Object::Dispose() [member function]
    cls.add_method('Dispose', 
                   'void', 
                   [])
    ## object.h (module 'core'): ns3::Object::AggregateIterator ns3::Object::GetAggregateIterator() const [member function]
    cls.add_method('GetAggregateIterator', 
                   'ns3::Object::AggregateIterator', 
                   [], 
                   is_const=True)
    ## object.h (module 'core'): ns3::TypeId ns3::Object::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## object.h (module 'core'): ns3::Ptr<ns3::Object> ns3::Object::GetObject() const [member function]
    cls.add_method('GetObject', 
                   'ns3::Ptr< ns3::Object >', 
                   [], 
                   custom_template_method_name='GetObject', is_const=True, template_parameters=['ns3::Object'])
    ## object.h (module 'core'): ns3::Ptr<ns3::Object> ns3::Object::GetObject(ns3::TypeId tid) const [member function]
    cls.add_method('GetObject', 
                   'ns3::Ptr< ns3::Object >', 
                   [param('ns3::TypeId', 'tid')], 
                   custom_template_method_name='GetObject', is_const=True, template_parameters=['ns3::Object'])
    ## object.h (module 'core'): static ns3::TypeId ns3::Object::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## object.h (module 'core'): void ns3::Object::Initialize() [member function]
    cls.add_method('Initialize', 
                   'void', 
                   [])
    ## object.h (module 'core'): bool ns3::Object::IsInitialized() const [member function]
    cls.add_method('IsInitialized', 
                   'bool', 
                   [], 
                   is_const=True)
    ## object.h (module 'core'): ns3::Object::Object(ns3::Object const & o) [constructor]
    cls.add_constructor([param('ns3::Object const &', 'o')], 
                        visibility='protected')
    ## object.h (module 'core'): void ns3::Object::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## object.h (module 'core'): void ns3::Object::DoInitialize() [member function]
    cls.add_method('DoInitialize', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## object.h (module 'core'): void ns3::Object::NotifyNewAggregate() [member function]
    cls.add_method('NotifyNewAggregate', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3ObjectAggregateIterator_methods(root_module, cls):
    ## object.h (module 'core'): ns3::Object::AggregateIterator::AggregateIterator(ns3::Object::AggregateIterator const & arg0) [constructor]
    cls.add_constructor([param('ns3::Object::AggregateIterator const &', 'arg0')])
    ## object.h (module 'core'): ns3::Object::AggregateIterator::AggregateIterator() [constructor]
    cls.add_constructor([])
    ## object.h (module 'core'): bool ns3::Object::AggregateIterator::HasNext() const [member function]
    cls.add_method('HasNext', 
                   'bool', 
                   [], 
                   is_const=True)
    ## object.h (module 'core'): ns3::Ptr<const ns3::Object> ns3::Object::AggregateIterator::Next() [member function]
    cls.add_method('Next', 
                   'ns3::Ptr< ns3::Object const >', 
                   [])
    return

def register_Ns3RandomVariableStream_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::RandomVariableStream::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::RandomVariableStream::RandomVariableStream() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): void ns3::RandomVariableStream::SetStream(int64_t stream) [member function]
    cls.add_method('SetStream', 
                   'void', 
                   [param('int64_t', 'stream')])
    ## random-variable-stream.h (module 'core'): int64_t ns3::RandomVariableStream::GetStream() const [member function]
    cls.add_method('GetStream', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): void ns3::RandomVariableStream::SetAntithetic(bool isAntithetic) [member function]
    cls.add_method('SetAntithetic', 
                   'void', 
                   [param('bool', 'isAntithetic')])
    ## random-variable-stream.h (module 'core'): bool ns3::RandomVariableStream::IsAntithetic() const [member function]
    cls.add_method('IsAntithetic', 
                   'bool', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::RandomVariableStream::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::RandomVariableStream::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## random-variable-stream.h (module 'core'): ns3::RngStream * ns3::RandomVariableStream::Peek() const [member function]
    cls.add_method('Peek', 
                   'ns3::RngStream *', 
                   [], 
                   is_const=True, visibility='protected')
    return

def register_Ns3SequentialRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::SequentialRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::SequentialRandomVariable::SequentialRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::SequentialRandomVariable::GetMin() const [member function]
    cls.add_method('GetMin', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::SequentialRandomVariable::GetMax() const [member function]
    cls.add_method('GetMax', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): ns3::Ptr<ns3::RandomVariableStream> ns3::SequentialRandomVariable::GetIncrement() const [member function]
    cls.add_method('GetIncrement', 
                   'ns3::Ptr< ns3::RandomVariableStream >', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::SequentialRandomVariable::GetConsecutive() const [member function]
    cls.add_method('GetConsecutive', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::SequentialRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::SequentialRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3SimpleRefCount__Ns3AttributeAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeAccessor__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::AttributeAccessor, ns3::empty, ns3::DefaultDeleter<ns3::AttributeAccessor>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::AttributeAccessor, ns3::empty, ns3::DefaultDeleter<ns3::AttributeAccessor>>::SimpleRefCount(ns3::SimpleRefCount<ns3::AttributeAccessor, ns3::empty, ns3::DefaultDeleter<ns3::AttributeAccessor>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::AttributeAccessor, ns3::empty, ns3::DefaultDeleter< ns3::AttributeAccessor > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3AttributeChecker_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeChecker__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::AttributeChecker, ns3::empty, ns3::DefaultDeleter<ns3::AttributeChecker>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::AttributeChecker, ns3::empty, ns3::DefaultDeleter<ns3::AttributeChecker>>::SimpleRefCount(ns3::SimpleRefCount<ns3::AttributeChecker, ns3::empty, ns3::DefaultDeleter<ns3::AttributeChecker>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::AttributeChecker, ns3::empty, ns3::DefaultDeleter< ns3::AttributeChecker > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3AttributeValue_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeValue__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::AttributeValue, ns3::empty, ns3::DefaultDeleter<ns3::AttributeValue>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::AttributeValue, ns3::empty, ns3::DefaultDeleter<ns3::AttributeValue>>::SimpleRefCount(ns3::SimpleRefCount<ns3::AttributeValue, ns3::empty, ns3::DefaultDeleter<ns3::AttributeValue>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::AttributeValue, ns3::empty, ns3::DefaultDeleter< ns3::AttributeValue > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3CallbackImplBase_Ns3Empty_Ns3DefaultDeleter__lt__ns3CallbackImplBase__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter<ns3::CallbackImplBase>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter<ns3::CallbackImplBase>>::SimpleRefCount(ns3::SimpleRefCount<ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter<ns3::CallbackImplBase>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter< ns3::CallbackImplBase > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3EventImpl_Ns3Empty_Ns3DefaultDeleter__lt__ns3EventImpl__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl>>::SimpleRefCount(ns3::SimpleRefCount<ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::EventImpl, ns3::empty, ns3::DefaultDeleter< ns3::EventImpl > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3HashImplementation_Ns3Empty_Ns3DefaultDeleter__lt__ns3HashImplementation__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Hash::Implementation, ns3::empty, ns3::DefaultDeleter<ns3::Hash::Implementation>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Hash::Implementation, ns3::empty, ns3::DefaultDeleter<ns3::Hash::Implementation>>::SimpleRefCount(ns3::SimpleRefCount<ns3::Hash::Implementation, ns3::empty, ns3::DefaultDeleter<ns3::Hash::Implementation>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::Hash::Implementation, ns3::empty, ns3::DefaultDeleter< ns3::Hash::Implementation > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3MeshWifiInterfaceMacPlugin_Ns3Empty_Ns3DefaultDeleter__lt__ns3MeshWifiInterfaceMacPlugin__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::MeshWifiInterfaceMacPlugin, ns3::empty, ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::MeshWifiInterfaceMacPlugin, ns3::empty, ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin>>::SimpleRefCount(ns3::SimpleRefCount<ns3::MeshWifiInterfaceMacPlugin, ns3::empty, ns3::DefaultDeleter<ns3::MeshWifiInterfaceMacPlugin>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::MeshWifiInterfaceMacPlugin, ns3::empty, ns3::DefaultDeleter< ns3::MeshWifiInterfaceMacPlugin > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3NixVector_Ns3Empty_Ns3DefaultDeleter__lt__ns3NixVector__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector>>::SimpleRefCount(ns3::SimpleRefCount<ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::NixVector, ns3::empty, ns3::DefaultDeleter< ns3::NixVector > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3Packet_Ns3Empty_Ns3DefaultDeleter__lt__ns3Packet__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet>>::SimpleRefCount(ns3::SimpleRefCount<ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::Packet, ns3::empty, ns3::DefaultDeleter< ns3::Packet > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3TraceSourceAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3TraceSourceAccessor__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>>::SimpleRefCount(ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter< ns3::TraceSourceAccessor > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3WifiInformationElement_Ns3Empty_Ns3DefaultDeleter__lt__ns3WifiInformationElement__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::WifiInformationElement, ns3::empty, ns3::DefaultDeleter<ns3::WifiInformationElement>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::WifiInformationElement, ns3::empty, ns3::DefaultDeleter<ns3::WifiInformationElement>>::SimpleRefCount(ns3::SimpleRefCount<ns3::WifiInformationElement, ns3::empty, ns3::DefaultDeleter<ns3::WifiInformationElement>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::WifiInformationElement, ns3::empty, ns3::DefaultDeleter< ns3::WifiInformationElement > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3Dot11sDestinationAddressUnit_Ns3Empty_Ns3DefaultDeleter__lt__ns3Dot11sDestinationAddressUnit__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::dot11s::DestinationAddressUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::dot11s::DestinationAddressUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit>>::SimpleRefCount(ns3::SimpleRefCount<ns3::dot11s::DestinationAddressUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::DestinationAddressUnit>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::dot11s::DestinationAddressUnit, ns3::empty, ns3::DefaultDeleter< ns3::dot11s::DestinationAddressUnit > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3Dot11sIeBeaconTimingUnit_Ns3Empty_Ns3DefaultDeleter__lt__ns3Dot11sIeBeaconTimingUnit__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::dot11s::IeBeaconTimingUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::dot11s::IeBeaconTimingUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit>>::SimpleRefCount(ns3::SimpleRefCount<ns3::dot11s::IeBeaconTimingUnit, ns3::empty, ns3::DefaultDeleter<ns3::dot11s::IeBeaconTimingUnit>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::dot11s::IeBeaconTimingUnit, ns3::empty, ns3::DefaultDeleter< ns3::dot11s::IeBeaconTimingUnit > > const &', 'o')])
    return

def register_Ns3TraceSourceAccessor_methods(root_module, cls):
    ## trace-source-accessor.h (module 'core'): ns3::TraceSourceAccessor::TraceSourceAccessor(ns3::TraceSourceAccessor const & arg0) [constructor]
    cls.add_constructor([param('ns3::TraceSourceAccessor const &', 'arg0')])
    ## trace-source-accessor.h (module 'core'): ns3::TraceSourceAccessor::TraceSourceAccessor() [constructor]
    cls.add_constructor([])
    ## trace-source-accessor.h (module 'core'): bool ns3::TraceSourceAccessor::Connect(ns3::ObjectBase * obj, std::string context, ns3::CallbackBase const & cb) const [member function]
    cls.add_method('Connect', 
                   'bool', 
                   [param('ns3::ObjectBase *', 'obj', transfer_ownership=False), param('std::string', 'context'), param('ns3::CallbackBase const &', 'cb')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## trace-source-accessor.h (module 'core'): bool ns3::TraceSourceAccessor::ConnectWithoutContext(ns3::ObjectBase * obj, ns3::CallbackBase const & cb) const [member function]
    cls.add_method('ConnectWithoutContext', 
                   'bool', 
                   [param('ns3::ObjectBase *', 'obj', transfer_ownership=False), param('ns3::CallbackBase const &', 'cb')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## trace-source-accessor.h (module 'core'): bool ns3::TraceSourceAccessor::Disconnect(ns3::ObjectBase * obj, std::string context, ns3::CallbackBase const & cb) const [member function]
    cls.add_method('Disconnect', 
                   'bool', 
                   [param('ns3::ObjectBase *', 'obj', transfer_ownership=False), param('std::string', 'context'), param('ns3::CallbackBase const &', 'cb')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## trace-source-accessor.h (module 'core'): bool ns3::TraceSourceAccessor::DisconnectWithoutContext(ns3::ObjectBase * obj, ns3::CallbackBase const & cb) const [member function]
    cls.add_method('DisconnectWithoutContext', 
                   'bool', 
                   [param('ns3::ObjectBase *', 'obj', transfer_ownership=False), param('ns3::CallbackBase const &', 'cb')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3Trailer_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## trailer.h (module 'network'): ns3::Trailer::Trailer() [constructor]
    cls.add_constructor([])
    ## trailer.h (module 'network'): ns3::Trailer::Trailer(ns3::Trailer const & arg0) [constructor]
    cls.add_constructor([param('ns3::Trailer const &', 'arg0')])
    ## trailer.h (module 'network'): uint32_t ns3::Trailer::Deserialize(ns3::Buffer::Iterator end) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'end')], 
                   is_pure_virtual=True, is_virtual=True)
    ## trailer.h (module 'network'): uint32_t ns3::Trailer::Deserialize(ns3::Buffer::Iterator start, ns3::Buffer::Iterator end) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('ns3::Buffer::Iterator', 'end')], 
                   is_virtual=True)
    ## trailer.h (module 'network'): uint32_t ns3::Trailer::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## trailer.h (module 'network'): static ns3::TypeId ns3::Trailer::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## trailer.h (module 'network'): void ns3::Trailer::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## trailer.h (module 'network'): void ns3::Trailer::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3TriangularRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::TriangularRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::TriangularRandomVariable::TriangularRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::TriangularRandomVariable::GetMean() const [member function]
    cls.add_method('GetMean', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::TriangularRandomVariable::GetMin() const [member function]
    cls.add_method('GetMin', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::TriangularRandomVariable::GetMax() const [member function]
    cls.add_method('GetMax', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::TriangularRandomVariable::GetValue(double mean, double min, double max) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('double', 'mean'), param('double', 'min'), param('double', 'max')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::TriangularRandomVariable::GetInteger(uint32_t mean, uint32_t min, uint32_t max) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'mean'), param('uint32_t', 'min'), param('uint32_t', 'max')])
    ## random-variable-stream.h (module 'core'): double ns3::TriangularRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::TriangularRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3UniformRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::UniformRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::UniformRandomVariable::UniformRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::UniformRandomVariable::GetMin() const [member function]
    cls.add_method('GetMin', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::UniformRandomVariable::GetMax() const [member function]
    cls.add_method('GetMax', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::UniformRandomVariable::GetValue(double min, double max) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('double', 'min'), param('double', 'max')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::UniformRandomVariable::GetInteger(uint32_t min, uint32_t max) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'min'), param('uint32_t', 'max')])
    ## random-variable-stream.h (module 'core'): double ns3::UniformRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::UniformRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3WeibullRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::WeibullRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::WeibullRandomVariable::WeibullRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::WeibullRandomVariable::GetScale() const [member function]
    cls.add_method('GetScale', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::WeibullRandomVariable::GetShape() const [member function]
    cls.add_method('GetShape', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::WeibullRandomVariable::GetBound() const [member function]
    cls.add_method('GetBound', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::WeibullRandomVariable::GetValue(double scale, double shape, double bound) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('double', 'scale'), param('double', 'shape'), param('double', 'bound')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::WeibullRandomVariable::GetInteger(uint32_t scale, uint32_t shape, uint32_t bound) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'scale'), param('uint32_t', 'shape'), param('uint32_t', 'bound')])
    ## random-variable-stream.h (module 'core'): double ns3::WeibullRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::WeibullRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3WifiActionHeader_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::WifiActionHeader(ns3::WifiActionHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiActionHeader const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::WifiActionHeader() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::WifiActionHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::ActionValue ns3::WifiActionHeader::GetAction() [member function]
    cls.add_method('GetAction', 
                   'ns3::WifiActionHeader::ActionValue', 
                   [])
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::CategoryValue ns3::WifiActionHeader::GetCategory() [member function]
    cls.add_method('GetCategory', 
                   'ns3::WifiActionHeader::CategoryValue', 
                   [])
    ## mgt-headers.h (module 'wifi'): ns3::TypeId ns3::WifiActionHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): uint32_t ns3::WifiActionHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): static ns3::TypeId ns3::WifiActionHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mgt-headers.h (module 'wifi'): void ns3::WifiActionHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::WifiActionHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mgt-headers.h (module 'wifi'): void ns3::WifiActionHeader::SetAction(ns3::WifiActionHeader::CategoryValue type, ns3::WifiActionHeader::ActionValue action) [member function]
    cls.add_method('SetAction', 
                   'void', 
                   [param('ns3::WifiActionHeader::CategoryValue', 'type'), param('ns3::WifiActionHeader::ActionValue', 'action')])
    return

def register_Ns3WifiActionHeaderActionValue_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::ActionValue() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::ActionValue(ns3::WifiActionHeader::ActionValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiActionHeader::ActionValue const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::ActionValue::blockAck [variable]
    cls.add_instance_attribute('blockAck', 'ns3::WifiActionHeader::BlockAckActionValue', is_const=False)
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::ActionValue::meshAction [variable]
    cls.add_instance_attribute('meshAction', 'ns3::WifiActionHeader::MeshActionValue', is_const=False)
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::ActionValue::multihopAction [variable]
    cls.add_instance_attribute('multihopAction', 'ns3::WifiActionHeader::MultihopActionValue', is_const=False)
    ## mgt-headers.h (module 'wifi'): ns3::WifiActionHeader::ActionValue::selfProtectedAction [variable]
    cls.add_instance_attribute('selfProtectedAction', 'ns3::WifiActionHeader::SelfProtectedActionValue', is_const=False)
    return

def register_Ns3WifiInformationElement_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## wifi-information-element.h (module 'wifi'): ns3::WifiInformationElement::WifiInformationElement() [constructor]
    cls.add_constructor([])
    ## wifi-information-element.h (module 'wifi'): ns3::WifiInformationElement::WifiInformationElement(ns3::WifiInformationElement const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiInformationElement const &', 'arg0')])
    ## wifi-information-element.h (module 'wifi'): ns3::Buffer::Iterator ns3::WifiInformationElement::Deserialize(ns3::Buffer::Iterator i) [member function]
    cls.add_method('Deserialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'i')])
    ## wifi-information-element.h (module 'wifi'): ns3::Buffer::Iterator ns3::WifiInformationElement::DeserializeIfPresent(ns3::Buffer::Iterator i) [member function]
    cls.add_method('DeserializeIfPresent', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'i')])
    ## wifi-information-element.h (module 'wifi'): uint8_t ns3::WifiInformationElement::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_pure_virtual=True, is_virtual=True)
    ## wifi-information-element.h (module 'wifi'): ns3::WifiInformationElementId ns3::WifiInformationElement::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## wifi-information-element.h (module 'wifi'): ns3::WifiInformationElementId ns3::WifiInformationElement::ElementIdExt() const [member function]
    cls.add_method('ElementIdExt', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wifi-information-element.h (module 'wifi'): uint8_t ns3::WifiInformationElement::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## wifi-information-element.h (module 'wifi'): uint16_t ns3::WifiInformationElement::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wifi-information-element.h (module 'wifi'): void ns3::WifiInformationElement::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## wifi-information-element.h (module 'wifi'): ns3::Buffer::Iterator ns3::WifiInformationElement::Serialize(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    ## wifi-information-element.h (module 'wifi'): void ns3::WifiInformationElement::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3WifiInformationElementVector_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## wifi-information-element-vector.h (module 'wifi'): ns3::WifiInformationElementVector::WifiInformationElementVector(ns3::WifiInformationElementVector const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiInformationElementVector const &', 'arg0')])
    ## wifi-information-element-vector.h (module 'wifi'): ns3::WifiInformationElementVector::WifiInformationElementVector() [constructor]
    cls.add_constructor([])
    ## wifi-information-element-vector.h (module 'wifi'): bool ns3::WifiInformationElementVector::AddInformationElement(ns3::Ptr<ns3::WifiInformationElement> element) [member function]
    cls.add_method('AddInformationElement', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WifiInformationElement >', 'element')])
    ## wifi-information-element-vector.h (module 'wifi'): ns3::WifiInformationElementVector::Iterator ns3::WifiInformationElementVector::Begin() [member function]
    cls.add_method('Begin', 
                   'ns3::WifiInformationElementVector::Iterator', 
                   [])
    ## wifi-information-element-vector.h (module 'wifi'): uint32_t ns3::WifiInformationElementVector::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## wifi-information-element-vector.h (module 'wifi'): uint32_t ns3::WifiInformationElementVector::Deserialize(ns3::Buffer::Iterator start, ns3::Buffer::Iterator end) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('ns3::Buffer::Iterator', 'end')], 
                   is_virtual=True)
    ## wifi-information-element-vector.h (module 'wifi'): uint32_t ns3::WifiInformationElementVector::DeserializeSingleIe(ns3::Buffer::Iterator start) [member function]
    cls.add_method('DeserializeSingleIe', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## wifi-information-element-vector.h (module 'wifi'): ns3::WifiInformationElementVector::Iterator ns3::WifiInformationElementVector::End() [member function]
    cls.add_method('End', 
                   'ns3::WifiInformationElementVector::Iterator', 
                   [])
    ## wifi-information-element-vector.h (module 'wifi'): ns3::Ptr<ns3::WifiInformationElement> ns3::WifiInformationElementVector::FindFirst(ns3::WifiInformationElementId id) const [member function]
    cls.add_method('FindFirst', 
                   'ns3::Ptr< ns3::WifiInformationElement >', 
                   [param('uint8_t', 'id')], 
                   is_const=True)
    ## wifi-information-element-vector.h (module 'wifi'): ns3::TypeId ns3::WifiInformationElementVector::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wifi-information-element-vector.h (module 'wifi'): uint32_t ns3::WifiInformationElementVector::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wifi-information-element-vector.h (module 'wifi'): static ns3::TypeId ns3::WifiInformationElementVector::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wifi-information-element-vector.h (module 'wifi'): void ns3::WifiInformationElementVector::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## wifi-information-element-vector.h (module 'wifi'): void ns3::WifiInformationElementVector::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## wifi-information-element-vector.h (module 'wifi'): uint32_t ns3::WifiInformationElementVector::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, visibility='protected')
    return

def register_Ns3WifiMac_methods(root_module, cls):
    ## wifi-mac.h (module 'wifi'): static ns3::TypeId ns3::WifiMac::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wifi-mac.h (module 'wifi'): ns3::WifiMac::WifiMac() [constructor]
    cls.add_constructor([])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetDevice(ns3::Ptr<ns3::WifiNetDevice> const device) [member function]
    cls.add_method('SetDevice', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiNetDevice > const', 'device')])
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::WifiNetDevice> ns3::WifiMac::GetDevice() const [member function]
    cls.add_method('GetDevice', 
                   'ns3::Ptr< ns3::WifiNetDevice >', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::FrameExchangeManager> ns3::WifiMac::GetFrameExchangeManager() const [member function]
    cls.add_method('GetFrameExchangeManager', 
                   'ns3::Ptr< ns3::FrameExchangeManager >', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::Txop> ns3::WifiMac::GetTxop() const [member function]
    cls.add_method('GetTxop', 
                   'ns3::Ptr< ns3::Txop >', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::QosTxop> ns3::WifiMac::GetQosTxop(ns3::AcIndex ac) const [member function]
    cls.add_method('GetQosTxop', 
                   'ns3::Ptr< ns3::QosTxop >', 
                   [param('ns3::AcIndex', 'ac')], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::QosTxop> ns3::WifiMac::GetQosTxop(uint8_t tid) const [member function]
    cls.add_method('GetQosTxop', 
                   'ns3::Ptr< ns3::QosTxop >', 
                   [param('uint8_t', 'tid')], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::WifiMacQueue> ns3::WifiMac::GetTxopQueue(ns3::AcIndex ac) const [member function]
    cls.add_method('GetTxopQueue', 
                   'ns3::Ptr< ns3::WifiMacQueue >', 
                   [param('ns3::AcIndex', 'ac')], 
                   is_const=True, is_virtual=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetTypeOfStation(ns3::TypeOfStation type) [member function]
    cls.add_method('SetTypeOfStation', 
                   'void', 
                   [param('ns3::TypeOfStation', 'type')])
    ## wifi-mac.h (module 'wifi'): ns3::TypeOfStation ns3::WifiMac::GetTypeOfStation() const [member function]
    cls.add_method('GetTypeOfStation', 
                   'ns3::TypeOfStation', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetSsid(ns3::Ssid ssid) [member function]
    cls.add_method('SetSsid', 
                   'void', 
                   [param('ns3::Ssid', 'ssid')])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetPromisc() [member function]
    cls.add_method('SetPromisc', 
                   'void', 
                   [])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetCtsToSelfSupported(bool enable) [member function]
    cls.add_method('SetCtsToSelfSupported', 
                   'void', 
                   [param('bool', 'enable')])
    ## wifi-mac.h (module 'wifi'): ns3::Mac48Address ns3::WifiMac::GetAddress() const [member function]
    cls.add_method('GetAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::Ssid ns3::WifiMac::GetSsid() const [member function]
    cls.add_method('GetSsid', 
                   'ns3::Ssid', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetAddress(ns3::Mac48Address address) [member function]
    cls.add_method('SetAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_virtual=True)
    ## wifi-mac.h (module 'wifi'): ns3::Mac48Address ns3::WifiMac::GetBssid() const [member function]
    cls.add_method('GetBssid', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetBssid(ns3::Mac48Address bssid) [member function]
    cls.add_method('SetBssid', 
                   'void', 
                   [param('ns3::Mac48Address', 'bssid')])
    ## wifi-mac.h (module 'wifi'): bool ns3::WifiMac::CanForwardPacketsTo(ns3::Mac48Address to) const [member function]
    cls.add_method('CanForwardPacketsTo', 
                   'bool', 
                   [param('ns3::Mac48Address', 'to')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::Enqueue(ns3::Ptr<ns3::Packet> packet, ns3::Mac48Address to, ns3::Mac48Address from) [member function]
    cls.add_method('Enqueue', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Mac48Address', 'to'), param('ns3::Mac48Address', 'from')], 
                   is_virtual=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::Enqueue(ns3::Ptr<ns3::Packet> packet, ns3::Mac48Address to) [member function]
    cls.add_method('Enqueue', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Mac48Address', 'to')], 
                   is_pure_virtual=True, is_virtual=True)
    ## wifi-mac.h (module 'wifi'): bool ns3::WifiMac::SupportsSendFrom() const [member function]
    cls.add_method('SupportsSendFrom', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetWifiPhy(ns3::Ptr<ns3::WifiPhy> phy) [member function]
    cls.add_method('SetWifiPhy', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiPhy >', 'phy')], 
                   is_virtual=True)
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::WifiPhy> ns3::WifiMac::GetWifiPhy() const [member function]
    cls.add_method('GetWifiPhy', 
                   'ns3::Ptr< ns3::WifiPhy >', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::ResetWifiPhy() [member function]
    cls.add_method('ResetWifiPhy', 
                   'void', 
                   [])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetWifiRemoteStationManager(ns3::Ptr<ns3::WifiRemoteStationManager> stationManager) [member function]
    cls.add_method('SetWifiRemoteStationManager', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiRemoteStationManager >', 'stationManager')])
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::WifiRemoteStationManager> ns3::WifiMac::GetWifiRemoteStationManager() const [member function]
    cls.add_method('GetWifiRemoteStationManager', 
                   'ns3::Ptr< ns3::WifiRemoteStationManager >', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetForwardUpCallback(ns3::WifiMac::ForwardUpCallback upCallback) [member function]
    cls.add_method('SetForwardUpCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::Packet const >, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'upCallback')])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetLinkUpCallback(ns3::Callback<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> linkUp) [member function]
    cls.add_method('SetLinkUpCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'linkUp')], 
                   is_virtual=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetLinkDownCallback(ns3::Callback<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> linkDown) [member function]
    cls.add_method('SetLinkDownCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'linkDown')])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::NotifyChannelSwitching() [member function]
    cls.add_method('NotifyChannelSwitching', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::NotifyTx(ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('NotifyTx', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet')])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::NotifyTxDrop(ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('NotifyTxDrop', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet')])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::NotifyRx(ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('NotifyRx', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet')])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::NotifyPromiscRx(ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('NotifyPromiscRx', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet')])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::NotifyRxDrop(ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('NotifyRxDrop', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet')])
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::ConfigureStandard(ns3::WifiStandard standard) [member function]
    cls.add_method('ConfigureStandard', 
                   'void', 
                   [param('ns3::WifiStandard', 'standard')], 
                   is_virtual=True)
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::HtConfiguration> ns3::WifiMac::GetHtConfiguration() const [member function]
    cls.add_method('GetHtConfiguration', 
                   'ns3::Ptr< ns3::HtConfiguration >', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::VhtConfiguration> ns3::WifiMac::GetVhtConfiguration() const [member function]
    cls.add_method('GetVhtConfiguration', 
                   'ns3::Ptr< ns3::VhtConfiguration >', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::HeConfiguration> ns3::WifiMac::GetHeConfiguration() const [member function]
    cls.add_method('GetHeConfiguration', 
                   'ns3::Ptr< ns3::HeConfiguration >', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::ExtendedCapabilities ns3::WifiMac::GetExtendedCapabilities() const [member function]
    cls.add_method('GetExtendedCapabilities', 
                   'ns3::ExtendedCapabilities', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::HtCapabilities ns3::WifiMac::GetHtCapabilities() const [member function]
    cls.add_method('GetHtCapabilities', 
                   'ns3::HtCapabilities', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::VhtCapabilities ns3::WifiMac::GetVhtCapabilities() const [member function]
    cls.add_method('GetVhtCapabilities', 
                   'ns3::VhtCapabilities', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): ns3::HeCapabilities ns3::WifiMac::GetHeCapabilities() const [member function]
    cls.add_method('GetHeCapabilities', 
                   'ns3::HeCapabilities', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): bool ns3::WifiMac::GetQosSupported() const [member function]
    cls.add_method('GetQosSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): bool ns3::WifiMac::GetErpSupported() const [member function]
    cls.add_method('GetErpSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): bool ns3::WifiMac::GetDsssSupported() const [member function]
    cls.add_method('GetDsssSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): bool ns3::WifiMac::GetHtSupported() const [member function]
    cls.add_method('GetHtSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): bool ns3::WifiMac::GetVhtSupported() const [member function]
    cls.add_method('GetVhtSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): bool ns3::WifiMac::GetHeSupported() const [member function]
    cls.add_method('GetHeSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): uint32_t ns3::WifiMac::GetMaxAmpduSize(ns3::AcIndex ac) const [member function]
    cls.add_method('GetMaxAmpduSize', 
                   'uint32_t', 
                   [param('ns3::AcIndex', 'ac')], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): uint16_t ns3::WifiMac::GetMaxAmsduSize(ns3::AcIndex ac) const [member function]
    cls.add_method('GetMaxAmsduSize', 
                   'uint16_t', 
                   [param('ns3::AcIndex', 'ac')], 
                   is_const=True)
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::DoInitialize() [member function]
    cls.add_method('DoInitialize', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::ConfigureContentionWindow(uint32_t cwMin, uint32_t cwMax) [member function]
    cls.add_method('ConfigureContentionWindow', 
                   'void', 
                   [param('uint32_t', 'cwMin'), param('uint32_t', 'cwMax')], 
                   is_virtual=True, visibility='protected')
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetQosSupported(bool enable) [member function]
    cls.add_method('SetQosSupported', 
                   'void', 
                   [param('bool', 'enable')], 
                   visibility='protected')
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::SetShortSlotTimeSupported(bool enable) [member function]
    cls.add_method('SetShortSlotTimeSupported', 
                   'void', 
                   [param('bool', 'enable')], 
                   visibility='protected')
    ## wifi-mac.h (module 'wifi'): bool ns3::WifiMac::GetShortSlotTimeSupported() const [member function]
    cls.add_method('GetShortSlotTimeSupported', 
                   'bool', 
                   [], 
                   is_const=True, visibility='protected')
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::QosTxop> ns3::WifiMac::GetVOQueue() const [member function]
    cls.add_method('GetVOQueue', 
                   'ns3::Ptr< ns3::QosTxop >', 
                   [], 
                   is_const=True, visibility='protected')
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::QosTxop> ns3::WifiMac::GetVIQueue() const [member function]
    cls.add_method('GetVIQueue', 
                   'ns3::Ptr< ns3::QosTxop >', 
                   [], 
                   is_const=True, visibility='protected')
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::QosTxop> ns3::WifiMac::GetBEQueue() const [member function]
    cls.add_method('GetBEQueue', 
                   'ns3::Ptr< ns3::QosTxop >', 
                   [], 
                   is_const=True, visibility='protected')
    ## wifi-mac.h (module 'wifi'): ns3::Ptr<ns3::QosTxop> ns3::WifiMac::GetBKQueue() const [member function]
    cls.add_method('GetBKQueue', 
                   'ns3::Ptr< ns3::QosTxop >', 
                   [], 
                   is_const=True, visibility='protected')
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::Receive(ns3::Ptr<ns3::WifiMacQueueItem> mpdu) [member function]
    cls.add_method('Receive', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem >', 'mpdu')], 
                   is_virtual=True, visibility='protected')
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::ForwardUp(ns3::Ptr<const ns3::Packet> packet, ns3::Mac48Address from, ns3::Mac48Address to) [member function]
    cls.add_method('ForwardUp', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet'), param('ns3::Mac48Address', 'from'), param('ns3::Mac48Address', 'to')], 
                   visibility='protected')
    ## wifi-mac.h (module 'wifi'): void ns3::WifiMac::DeaggregateAmsduAndForward(ns3::Ptr<ns3::WifiMacQueueItem> mpdu) [member function]
    cls.add_method('DeaggregateAmsduAndForward', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem >', 'mpdu')], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3WifiRemoteStationManager_methods(root_module, cls):
    ## wifi-remote-station-manager.h (module 'wifi'): static ns3::TypeId ns3::WifiRemoteStationManager::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationManager::WifiRemoteStationManager() [constructor]
    cls.add_constructor([])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetupPhy(ns3::Ptr<ns3::WifiPhy> const phy) [member function]
    cls.add_method('SetupPhy', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiPhy > const', 'phy')], 
                   is_virtual=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetupMac(ns3::Ptr<ns3::WifiMac> const mac) [member function]
    cls.add_method('SetupMac', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiMac > const', 'mac')], 
                   is_virtual=True)
    ## wifi-remote-station-manager.h (module 'wifi'): int64_t ns3::WifiRemoteStationManager::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetMaxSsrc(uint32_t maxSsrc) [member function]
    cls.add_method('SetMaxSsrc', 
                   'void', 
                   [param('uint32_t', 'maxSsrc')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetMaxSlrc(uint32_t maxSlrc) [member function]
    cls.add_method('SetMaxSlrc', 
                   'void', 
                   [param('uint32_t', 'maxSlrc')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetRtsCtsThreshold(uint32_t threshold) [member function]
    cls.add_method('SetRtsCtsThreshold', 
                   'void', 
                   [param('uint32_t', 'threshold')])
    ## wifi-remote-station-manager.h (module 'wifi'): uint32_t ns3::WifiRemoteStationManager::GetFragmentationThreshold() const [member function]
    cls.add_method('GetFragmentationThreshold', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetFragmentationThreshold(uint32_t threshold) [member function]
    cls.add_method('SetFragmentationThreshold', 
                   'void', 
                   [param('uint32_t', 'threshold')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetAssociationId(ns3::Mac48Address remoteAddress, uint16_t aid) [member function]
    cls.add_method('SetAssociationId', 
                   'void', 
                   [param('ns3::Mac48Address', 'remoteAddress'), param('uint16_t', 'aid')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetQosSupport(ns3::Mac48Address from, bool qosSupported) [member function]
    cls.add_method('SetQosSupport', 
                   'void', 
                   [param('ns3::Mac48Address', 'from'), param('bool', 'qosSupported')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddStationHtCapabilities(ns3::Mac48Address from, ns3::HtCapabilities htCapabilities) [member function]
    cls.add_method('AddStationHtCapabilities', 
                   'void', 
                   [param('ns3::Mac48Address', 'from'), param('ns3::HtCapabilities', 'htCapabilities')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddStationVhtCapabilities(ns3::Mac48Address from, ns3::VhtCapabilities vhtCapabilities) [member function]
    cls.add_method('AddStationVhtCapabilities', 
                   'void', 
                   [param('ns3::Mac48Address', 'from'), param('ns3::VhtCapabilities', 'vhtCapabilities')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddStationHeCapabilities(ns3::Mac48Address from, ns3::HeCapabilities heCapabilities) [member function]
    cls.add_method('AddStationHeCapabilities', 
                   'void', 
                   [param('ns3::Mac48Address', 'from'), param('ns3::HeCapabilities', 'heCapabilities')])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::Ptr<const ns3::HtCapabilities> ns3::WifiRemoteStationManager::GetStationHtCapabilities(ns3::Mac48Address from) [member function]
    cls.add_method('GetStationHtCapabilities', 
                   'ns3::Ptr< ns3::HtCapabilities const >', 
                   [param('ns3::Mac48Address', 'from')])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::Ptr<const ns3::VhtCapabilities> ns3::WifiRemoteStationManager::GetStationVhtCapabilities(ns3::Mac48Address from) [member function]
    cls.add_method('GetStationVhtCapabilities', 
                   'ns3::Ptr< ns3::VhtCapabilities const >', 
                   [param('ns3::Mac48Address', 'from')])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::Ptr<const ns3::HeCapabilities> ns3::WifiRemoteStationManager::GetStationHeCapabilities(ns3::Mac48Address from) [member function]
    cls.add_method('GetStationHeCapabilities', 
                   'ns3::Ptr< ns3::HeCapabilities const >', 
                   [param('ns3::Mac48Address', 'from')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetHtSupported() const [member function]
    cls.add_method('GetHtSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetVhtSupported() const [member function]
    cls.add_method('GetVhtSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetHeSupported() const [member function]
    cls.add_method('GetHeSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetLdpcSupported() const [member function]
    cls.add_method('GetLdpcSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetShortGuardIntervalSupported() const [member function]
    cls.add_method('GetShortGuardIntervalSupported', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): uint16_t ns3::WifiRemoteStationManager::GetGuardInterval() const [member function]
    cls.add_method('GetGuardInterval', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetUseNonErpProtection(bool enable) [member function]
    cls.add_method('SetUseNonErpProtection', 
                   'void', 
                   [param('bool', 'enable')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetUseNonErpProtection() const [member function]
    cls.add_method('GetUseNonErpProtection', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetUseNonHtProtection(bool enable) [member function]
    cls.add_method('SetUseNonHtProtection', 
                   'void', 
                   [param('bool', 'enable')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetUseNonHtProtection() const [member function]
    cls.add_method('GetUseNonHtProtection', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetShortPreambleEnabled(bool enable) [member function]
    cls.add_method('SetShortPreambleEnabled', 
                   'void', 
                   [param('bool', 'enable')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetShortPreambleEnabled() const [member function]
    cls.add_method('GetShortPreambleEnabled', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetShortSlotTimeEnabled(bool enable) [member function]
    cls.add_method('SetShortSlotTimeEnabled', 
                   'void', 
                   [param('bool', 'enable')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetShortSlotTimeEnabled() const [member function]
    cls.add_method('GetShortSlotTimeEnabled', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::Reset() [member function]
    cls.add_method('Reset', 
                   'void', 
                   [])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddBasicMode(ns3::WifiMode mode) [member function]
    cls.add_method('AddBasicMode', 
                   'void', 
                   [param('ns3::WifiMode', 'mode')])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiMode ns3::WifiRemoteStationManager::GetDefaultMode() const [member function]
    cls.add_method('GetDefaultMode', 
                   'ns3::WifiMode', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetNBasicModes() const [member function]
    cls.add_method('GetNBasicModes', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiMode ns3::WifiRemoteStationManager::GetBasicMode(uint8_t i) const [member function]
    cls.add_method('GetBasicMode', 
                   'ns3::WifiMode', 
                   [param('uint8_t', 'i')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): uint32_t ns3::WifiRemoteStationManager::GetNNonErpBasicModes() const [member function]
    cls.add_method('GetNNonErpBasicModes', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiMode ns3::WifiRemoteStationManager::GetNonErpBasicMode(uint8_t i) const [member function]
    cls.add_method('GetNonErpBasicMode', 
                   'ns3::WifiMode', 
                   [param('uint8_t', 'i')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetLdpcSupported(ns3::Mac48Address address) const [member function]
    cls.add_method('GetLdpcSupported', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetShortPreambleSupported(ns3::Mac48Address address) const [member function]
    cls.add_method('GetShortPreambleSupported', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetShortSlotTimeSupported(ns3::Mac48Address address) const [member function]
    cls.add_method('GetShortSlotTimeSupported', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetQosSupported(ns3::Mac48Address address) const [member function]
    cls.add_method('GetQosSupported', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): uint16_t ns3::WifiRemoteStationManager::GetAssociationId(ns3::Mac48Address remoteAddress) const [member function]
    cls.add_method('GetAssociationId', 
                   'uint16_t', 
                   [param('ns3::Mac48Address', 'remoteAddress')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddBasicMcs(ns3::WifiMode mcs) [member function]
    cls.add_method('AddBasicMcs', 
                   'void', 
                   [param('ns3::WifiMode', 'mcs')])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiMode ns3::WifiRemoteStationManager::GetDefaultMcs() const [member function]
    cls.add_method('GetDefaultMcs', 
                   'ns3::WifiMode', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetNBasicMcs() const [member function]
    cls.add_method('GetNBasicMcs', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiMode ns3::WifiRemoteStationManager::GetBasicMcs(uint8_t i) const [member function]
    cls.add_method('GetBasicMcs', 
                   'ns3::WifiMode', 
                   [param('uint8_t', 'i')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddSupportedMcs(ns3::Mac48Address address, ns3::WifiMode mcs) [member function]
    cls.add_method('AddSupportedMcs', 
                   'void', 
                   [param('ns3::Mac48Address', 'address'), param('ns3::WifiMode', 'mcs')])
    ## wifi-remote-station-manager.h (module 'wifi'): uint16_t ns3::WifiRemoteStationManager::GetChannelWidthSupported(ns3::Mac48Address address) const [member function]
    cls.add_method('GetChannelWidthSupported', 
                   'uint16_t', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetShortGuardIntervalSupported(ns3::Mac48Address address) const [member function]
    cls.add_method('GetShortGuardIntervalSupported', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetNumberOfSupportedStreams(ns3::Mac48Address address) const [member function]
    cls.add_method('GetNumberOfSupportedStreams', 
                   'uint8_t', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetNMcsSupported(ns3::Mac48Address address) const [member function]
    cls.add_method('GetNMcsSupported', 
                   'uint8_t', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetDsssSupported(ns3::Mac48Address const & address) const [member function]
    cls.add_method('GetDsssSupported', 
                   'bool', 
                   [param('ns3::Mac48Address const &', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetErpOfdmSupported(ns3::Mac48Address const & address) const [member function]
    cls.add_method('GetErpOfdmSupported', 
                   'bool', 
                   [param('ns3::Mac48Address const &', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetOfdmSupported(ns3::Mac48Address const & address) const [member function]
    cls.add_method('GetOfdmSupported', 
                   'bool', 
                   [param('ns3::Mac48Address const &', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetHtSupported(ns3::Mac48Address address) const [member function]
    cls.add_method('GetHtSupported', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetVhtSupported(ns3::Mac48Address address) const [member function]
    cls.add_method('GetVhtSupported', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetHeSupported(ns3::Mac48Address address) const [member function]
    cls.add_method('GetHeSupported', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiMode ns3::WifiRemoteStationManager::GetNonUnicastMode() const [member function]
    cls.add_method('GetNonUnicastMode', 
                   'ns3::WifiMode', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddSupportedMode(ns3::Mac48Address address, ns3::WifiMode mode) [member function]
    cls.add_method('AddSupportedMode', 
                   'void', 
                   [param('ns3::Mac48Address', 'address'), param('ns3::WifiMode', 'mode')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddAllSupportedModes(ns3::Mac48Address address) [member function]
    cls.add_method('AddAllSupportedModes', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddAllSupportedMcs(ns3::Mac48Address address) [member function]
    cls.add_method('AddAllSupportedMcs', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::RemoveAllSupportedMcs(ns3::Mac48Address address) [member function]
    cls.add_method('RemoveAllSupportedMcs', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddSupportedPhyPreamble(ns3::Mac48Address address, bool isShortPreambleSupported) [member function]
    cls.add_method('AddSupportedPhyPreamble', 
                   'void', 
                   [param('ns3::Mac48Address', 'address'), param('bool', 'isShortPreambleSupported')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::AddSupportedErpSlotTime(ns3::Mac48Address address, bool isShortSlotTimeSupported) [member function]
    cls.add_method('AddSupportedErpSlotTime', 
                   'void', 
                   [param('ns3::Mac48Address', 'address'), param('bool', 'isShortSlotTimeSupported')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::IsBrandNew(ns3::Mac48Address address) const [member function]
    cls.add_method('IsBrandNew', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::IsAssociated(ns3::Mac48Address address) const [member function]
    cls.add_method('IsAssociated', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::IsWaitAssocTxOk(ns3::Mac48Address address) const [member function]
    cls.add_method('IsWaitAssocTxOk', 
                   'bool', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::RecordWaitAssocTxOk(ns3::Mac48Address address) [member function]
    cls.add_method('RecordWaitAssocTxOk', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::RecordGotAssocTxOk(ns3::Mac48Address address) [member function]
    cls.add_method('RecordGotAssocTxOk', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::RecordGotAssocTxFailed(ns3::Mac48Address address) [member function]
    cls.add_method('RecordGotAssocTxFailed', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::RecordDisassociated(ns3::Mac48Address address) [member function]
    cls.add_method('RecordDisassociated', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiTxVector ns3::WifiRemoteStationManager::GetDataTxVector(ns3::WifiMacHeader const & header) [member function]
    cls.add_method('GetDataTxVector', 
                   'ns3::WifiTxVector', 
                   [param('ns3::WifiMacHeader const &', 'header')])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiTxVector ns3::WifiRemoteStationManager::GetRtsTxVector(ns3::Mac48Address address) [member function]
    cls.add_method('GetRtsTxVector', 
                   'ns3::WifiTxVector', 
                   [param('ns3::Mac48Address', 'address')])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiTxVector ns3::WifiRemoteStationManager::GetCtsTxVector(ns3::Mac48Address to, ns3::WifiMode rtsTxMode) const [member function]
    cls.add_method('GetCtsTxVector', 
                   'ns3::WifiTxVector', 
                   [param('ns3::Mac48Address', 'to'), param('ns3::WifiMode', 'rtsTxMode')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiTxVector ns3::WifiRemoteStationManager::GetCtsToSelfTxVector() [member function]
    cls.add_method('GetCtsToSelfTxVector', 
                   'ns3::WifiTxVector', 
                   [])
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiTxVector ns3::WifiRemoteStationManager::GetAckTxVector(ns3::Mac48Address to, ns3::WifiTxVector const & dataTxVector) const [member function]
    cls.add_method('GetAckTxVector', 
                   'ns3::WifiTxVector', 
                   [param('ns3::Mac48Address', 'to'), param('ns3::WifiTxVector const &', 'dataTxVector')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiTxVector ns3::WifiRemoteStationManager::GetBlockAckTxVector(ns3::Mac48Address to, ns3::WifiTxVector const & dataTxVector) const [member function]
    cls.add_method('GetBlockAckTxVector', 
                   'ns3::WifiTxVector', 
                   [param('ns3::Mac48Address', 'to'), param('ns3::WifiTxVector const &', 'dataTxVector')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiMode ns3::WifiRemoteStationManager::GetControlAnswerMode(ns3::WifiMode reqMode) const [member function]
    cls.add_method('GetControlAnswerMode', 
                   'ns3::WifiMode', 
                   [param('ns3::WifiMode', 'reqMode')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::ReportRtsFailed(ns3::WifiMacHeader const & header) [member function]
    cls.add_method('ReportRtsFailed', 
                   'void', 
                   [param('ns3::WifiMacHeader const &', 'header')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::ReportDataFailed(ns3::Ptr<const ns3::WifiMacQueueItem> mpdu) [member function]
    cls.add_method('ReportDataFailed', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'mpdu')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::ReportRtsOk(ns3::WifiMacHeader const & header, double ctsSnr, ns3::WifiMode ctsMode, double rtsSnr) [member function]
    cls.add_method('ReportRtsOk', 
                   'void', 
                   [param('ns3::WifiMacHeader const &', 'header'), param('double', 'ctsSnr'), param('ns3::WifiMode', 'ctsMode'), param('double', 'rtsSnr')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::ReportDataOk(ns3::Ptr<const ns3::WifiMacQueueItem> mpdu, double ackSnr, ns3::WifiMode ackMode, double dataSnr, ns3::WifiTxVector dataTxVector) [member function]
    cls.add_method('ReportDataOk', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'mpdu'), param('double', 'ackSnr'), param('ns3::WifiMode', 'ackMode'), param('double', 'dataSnr'), param('ns3::WifiTxVector', 'dataTxVector')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::ReportFinalRtsFailed(ns3::WifiMacHeader const & header) [member function]
    cls.add_method('ReportFinalRtsFailed', 
                   'void', 
                   [param('ns3::WifiMacHeader const &', 'header')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::ReportFinalDataFailed(ns3::Ptr<const ns3::WifiMacQueueItem> mpdu) [member function]
    cls.add_method('ReportFinalDataFailed', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'mpdu')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::ReportAmpduTxStatus(ns3::Mac48Address address, uint16_t nSuccessfulMpdus, uint16_t nFailedMpdus, double rxSnr, double dataSnr, ns3::WifiTxVector dataTxVector) [member function]
    cls.add_method('ReportAmpduTxStatus', 
                   'void', 
                   [param('ns3::Mac48Address', 'address'), param('uint16_t', 'nSuccessfulMpdus'), param('uint16_t', 'nFailedMpdus'), param('double', 'rxSnr'), param('double', 'dataSnr'), param('ns3::WifiTxVector', 'dataTxVector')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::ReportRxOk(ns3::Mac48Address address, ns3::RxSignalInfo rxSignalInfo, ns3::WifiTxVector txVector) [member function]
    cls.add_method('ReportRxOk', 
                   'void', 
                   [param('ns3::Mac48Address', 'address'), param('ns3::RxSignalInfo', 'rxSignalInfo'), param('ns3::WifiTxVector', 'txVector')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::NeedRts(ns3::WifiMacHeader const & header, uint32_t size) [member function]
    cls.add_method('NeedRts', 
                   'bool', 
                   [param('ns3::WifiMacHeader const &', 'header'), param('uint32_t', 'size')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::NeedCtsToSelf(ns3::WifiTxVector txVector) [member function]
    cls.add_method('NeedCtsToSelf', 
                   'bool', 
                   [param('ns3::WifiTxVector', 'txVector')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::NeedRetransmission(ns3::Ptr<const ns3::WifiMacQueueItem> mpdu) [member function]
    cls.add_method('NeedRetransmission', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'mpdu')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::NeedFragmentation(ns3::Ptr<const ns3::WifiMacQueueItem> mpdu) [member function]
    cls.add_method('NeedFragmentation', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'mpdu')])
    ## wifi-remote-station-manager.h (module 'wifi'): uint32_t ns3::WifiRemoteStationManager::GetFragmentSize(ns3::Ptr<const ns3::WifiMacQueueItem> mpdu, uint32_t fragmentNumber) [member function]
    cls.add_method('GetFragmentSize', 
                   'uint32_t', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'mpdu'), param('uint32_t', 'fragmentNumber')])
    ## wifi-remote-station-manager.h (module 'wifi'): uint32_t ns3::WifiRemoteStationManager::GetFragmentOffset(ns3::Ptr<const ns3::WifiMacQueueItem> mpdu, uint32_t fragmentNumber) [member function]
    cls.add_method('GetFragmentOffset', 
                   'uint32_t', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'mpdu'), param('uint32_t', 'fragmentNumber')])
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::IsLastFragment(ns3::Ptr<const ns3::WifiMacQueueItem> mpdu, uint32_t fragmentNumber) [member function]
    cls.add_method('IsLastFragment', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'mpdu'), param('uint32_t', 'fragmentNumber')])
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetDefaultTxPowerLevel() const [member function]
    cls.add_method('GetDefaultTxPowerLevel', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationInfo ns3::WifiRemoteStationManager::GetInfo(ns3::Mac48Address address) [member function]
    cls.add_method('GetInfo', 
                   'ns3::WifiRemoteStationInfo', 
                   [param('ns3::Mac48Address', 'address')])
    ## wifi-remote-station-manager.h (module 'wifi'): double ns3::WifiRemoteStationManager::GetMostRecentRssi(ns3::Mac48Address address) const [member function]
    cls.add_method('GetMostRecentRssi', 
                   'double', 
                   [param('ns3::Mac48Address', 'address')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::SetDefaultTxPowerLevel(uint8_t txPower) [member function]
    cls.add_method('SetDefaultTxPowerLevel', 
                   'void', 
                   [param('uint8_t', 'txPower')])
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetNumberOfAntennas() const [member function]
    cls.add_method('GetNumberOfAntennas', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetMaxNumberOfTransmitStreams() const [member function]
    cls.add_method('GetMaxNumberOfTransmitStreams', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::UseLdpcForDestination(ns3::Mac48Address dest) const [member function]
    cls.add_method('UseLdpcForDestination', 
                   'bool', 
                   [param('ns3::Mac48Address', 'dest')], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::Ptr<ns3::WifiPhy> ns3::WifiRemoteStationManager::GetPhy() const [member function]
    cls.add_method('GetPhy', 
                   'ns3::Ptr< ns3::WifiPhy >', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::Ptr<ns3::WifiMac> ns3::WifiRemoteStationManager::GetMac() const [member function]
    cls.add_method('GetMac', 
                   'ns3::Ptr< ns3::WifiMac >', 
                   [], 
                   is_const=True)
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStationManager::WifiRemoteStationManager(ns3::WifiRemoteStationManager const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiRemoteStationManager const &', 'arg0')])
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiMode ns3::WifiRemoteStationManager::GetSupported(ns3::WifiRemoteStation const * station, uint8_t i) const [member function]
    cls.add_method('GetSupported', 
                   'ns3::WifiMode', 
                   [param('ns3::WifiRemoteStation const *', 'station'), param('uint8_t', 'i')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetNSupported(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetNSupported', 
                   'uint8_t', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetQosSupported(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetQosSupported', 
                   'bool', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetHtSupported(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetHtSupported', 
                   'bool', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetVhtSupported(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetVhtSupported', 
                   'bool', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetHeSupported(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetHeSupported', 
                   'bool', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiMode ns3::WifiRemoteStationManager::GetMcsSupported(ns3::WifiRemoteStation const * station, uint8_t i) const [member function]
    cls.add_method('GetMcsSupported', 
                   'ns3::WifiMode', 
                   [param('ns3::WifiRemoteStation const *', 'station'), param('uint8_t', 'i')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetNMcsSupported(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetNMcsSupported', 
                   'uint8_t', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiMode ns3::WifiRemoteStationManager::GetNonErpSupported(ns3::WifiRemoteStation const * station, uint8_t i) const [member function]
    cls.add_method('GetNonErpSupported', 
                   'ns3::WifiMode', 
                   [param('ns3::WifiRemoteStation const *', 'station'), param('uint8_t', 'i')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): uint32_t ns3::WifiRemoteStationManager::GetNNonErpSupported(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetNNonErpSupported', 
                   'uint32_t', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::Mac48Address ns3::WifiRemoteStationManager::GetAddress(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetAddress', 
                   'ns3::Mac48Address', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): uint16_t ns3::WifiRemoteStationManager::GetChannelWidth(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetChannelWidth', 
                   'uint16_t', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetShortGuardIntervalSupported(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetShortGuardIntervalSupported', 
                   'bool', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): uint16_t ns3::WifiRemoteStationManager::GetGuardInterval(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetGuardInterval', 
                   'uint16_t', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::GetAggregation(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetAggregation', 
                   'bool', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetNumberOfSupportedStreams(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetNumberOfSupportedStreams', 
                   'uint8_t', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): uint8_t ns3::WifiRemoteStationManager::GetNess(ns3::WifiRemoteStation const * station) const [member function]
    cls.add_method('GetNess', 
                   'uint8_t', 
                   [param('ns3::WifiRemoteStation const *', 'station')], 
                   is_const=True, visibility='protected')
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::DoNeedRts(ns3::WifiRemoteStation * station, uint32_t size, bool normally) [member function]
    cls.add_method('DoNeedRts', 
                   'bool', 
                   [param('ns3::WifiRemoteStation *', 'station'), param('uint32_t', 'size'), param('bool', 'normally')], 
                   is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::DoNeedRetransmission(ns3::WifiRemoteStation * station, ns3::Ptr<const ns3::Packet> packet, bool normally) [member function]
    cls.add_method('DoNeedRetransmission', 
                   'bool', 
                   [param('ns3::WifiRemoteStation *', 'station'), param('ns3::Ptr< ns3::Packet const >', 'packet'), param('bool', 'normally')], 
                   is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): bool ns3::WifiRemoteStationManager::DoNeedFragmentation(ns3::WifiRemoteStation * station, ns3::Ptr<const ns3::Packet> packet, bool normally) [member function]
    cls.add_method('DoNeedFragmentation', 
                   'bool', 
                   [param('ns3::WifiRemoteStation *', 'station'), param('ns3::Ptr< ns3::Packet const >', 'packet'), param('bool', 'normally')], 
                   is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiRemoteStation * ns3::WifiRemoteStationManager::DoCreateStation() const [member function]
    cls.add_method('DoCreateStation', 
                   'ns3::WifiRemoteStation *', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiTxVector ns3::WifiRemoteStationManager::DoGetDataTxVector(ns3::WifiRemoteStation * station) [member function]
    cls.add_method('DoGetDataTxVector', 
                   'ns3::WifiTxVector', 
                   [param('ns3::WifiRemoteStation *', 'station')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): ns3::WifiTxVector ns3::WifiRemoteStationManager::DoGetRtsTxVector(ns3::WifiRemoteStation * station) [member function]
    cls.add_method('DoGetRtsTxVector', 
                   'ns3::WifiTxVector', 
                   [param('ns3::WifiRemoteStation *', 'station')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::DoReportRtsFailed(ns3::WifiRemoteStation * station) [member function]
    cls.add_method('DoReportRtsFailed', 
                   'void', 
                   [param('ns3::WifiRemoteStation *', 'station')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::DoReportDataFailed(ns3::WifiRemoteStation * station) [member function]
    cls.add_method('DoReportDataFailed', 
                   'void', 
                   [param('ns3::WifiRemoteStation *', 'station')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::DoReportRtsOk(ns3::WifiRemoteStation * station, double ctsSnr, ns3::WifiMode ctsMode, double rtsSnr) [member function]
    cls.add_method('DoReportRtsOk', 
                   'void', 
                   [param('ns3::WifiRemoteStation *', 'station'), param('double', 'ctsSnr'), param('ns3::WifiMode', 'ctsMode'), param('double', 'rtsSnr')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::DoReportDataOk(ns3::WifiRemoteStation * station, double ackSnr, ns3::WifiMode ackMode, double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss) [member function]
    cls.add_method('DoReportDataOk', 
                   'void', 
                   [param('ns3::WifiRemoteStation *', 'station'), param('double', 'ackSnr'), param('ns3::WifiMode', 'ackMode'), param('double', 'dataSnr'), param('uint16_t', 'dataChannelWidth'), param('uint8_t', 'dataNss')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::DoReportFinalRtsFailed(ns3::WifiRemoteStation * station) [member function]
    cls.add_method('DoReportFinalRtsFailed', 
                   'void', 
                   [param('ns3::WifiRemoteStation *', 'station')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::DoReportFinalDataFailed(ns3::WifiRemoteStation * station) [member function]
    cls.add_method('DoReportFinalDataFailed', 
                   'void', 
                   [param('ns3::WifiRemoteStation *', 'station')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::DoReportRxOk(ns3::WifiRemoteStation * station, double rxSnr, ns3::WifiMode txMode) [member function]
    cls.add_method('DoReportRxOk', 
                   'void', 
                   [param('ns3::WifiRemoteStation *', 'station'), param('double', 'rxSnr'), param('ns3::WifiMode', 'txMode')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wifi-remote-station-manager.h (module 'wifi'): void ns3::WifiRemoteStationManager::DoReportAmpduTxStatus(ns3::WifiRemoteStation * station, uint16_t nSuccessfulMpdus, uint16_t nFailedMpdus, double rxSnr, double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss) [member function]
    cls.add_method('DoReportAmpduTxStatus', 
                   'void', 
                   [param('ns3::WifiRemoteStation *', 'station'), param('uint16_t', 'nSuccessfulMpdus'), param('uint16_t', 'nFailedMpdus'), param('double', 'rxSnr'), param('double', 'dataSnr'), param('uint16_t', 'dataChannelWidth'), param('uint8_t', 'dataNss')], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3ZetaRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::ZetaRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::ZetaRandomVariable::ZetaRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::ZetaRandomVariable::GetAlpha() const [member function]
    cls.add_method('GetAlpha', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ZetaRandomVariable::GetValue(double alpha) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('double', 'alpha')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ZetaRandomVariable::GetInteger(uint32_t alpha) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'alpha')])
    ## random-variable-stream.h (module 'core'): double ns3::ZetaRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ZetaRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3ZipfRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::ZipfRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::ZipfRandomVariable::ZipfRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ZipfRandomVariable::GetN() const [member function]
    cls.add_method('GetN', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ZipfRandomVariable::GetAlpha() const [member function]
    cls.add_method('GetAlpha', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ZipfRandomVariable::GetValue(uint32_t n, double alpha) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('uint32_t', 'n'), param('double', 'alpha')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ZipfRandomVariable::GetInteger(uint32_t n, uint32_t alpha) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'n'), param('uint32_t', 'alpha')])
    ## random-variable-stream.h (module 'core'): double ns3::ZipfRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ZipfRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3AttributeAccessor_methods(root_module, cls):
    ## attribute.h (module 'core'): ns3::AttributeAccessor::AttributeAccessor(ns3::AttributeAccessor const & arg0) [constructor]
    cls.add_constructor([param('ns3::AttributeAccessor const &', 'arg0')])
    ## attribute.h (module 'core'): ns3::AttributeAccessor::AttributeAccessor() [constructor]
    cls.add_constructor([])
    ## attribute.h (module 'core'): bool ns3::AttributeAccessor::Get(ns3::ObjectBase const * object, ns3::AttributeValue & attribute) const [member function]
    cls.add_method('Get', 
                   'bool', 
                   [param('ns3::ObjectBase const *', 'object'), param('ns3::AttributeValue &', 'attribute')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::AttributeAccessor::HasGetter() const [member function]
    cls.add_method('HasGetter', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::AttributeAccessor::HasSetter() const [member function]
    cls.add_method('HasSetter', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::AttributeAccessor::Set(ns3::ObjectBase * object, ns3::AttributeValue const & value) const [member function]
    cls.add_method('Set', 
                   'bool', 
                   [param('ns3::ObjectBase *', 'object', transfer_ownership=False), param('ns3::AttributeValue const &', 'value')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3AttributeChecker_methods(root_module, cls):
    ## attribute.h (module 'core'): ns3::AttributeChecker::AttributeChecker(ns3::AttributeChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::AttributeChecker const &', 'arg0')])
    ## attribute.h (module 'core'): ns3::AttributeChecker::AttributeChecker() [constructor]
    cls.add_constructor([])
    ## attribute.h (module 'core'): bool ns3::AttributeChecker::Check(ns3::AttributeValue const & value) const [member function]
    cls.add_method('Check', 
                   'bool', 
                   [param('ns3::AttributeValue const &', 'value')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::AttributeChecker::Copy(ns3::AttributeValue const & source, ns3::AttributeValue & destination) const [member function]
    cls.add_method('Copy', 
                   'bool', 
                   [param('ns3::AttributeValue const &', 'source'), param('ns3::AttributeValue &', 'destination')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## attribute.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::AttributeChecker::Create() const [member function]
    cls.add_method('Create', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## attribute.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::AttributeChecker::CreateValidValue(ns3::AttributeValue const & value) const [member function]
    cls.add_method('CreateValidValue', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [param('ns3::AttributeValue const &', 'value')], 
                   is_const=True)
    ## attribute.h (module 'core'): std::string ns3::AttributeChecker::GetUnderlyingTypeInformation() const [member function]
    cls.add_method('GetUnderlyingTypeInformation', 
                   'std::string', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## attribute.h (module 'core'): std::string ns3::AttributeChecker::GetValueTypeName() const [member function]
    cls.add_method('GetValueTypeName', 
                   'std::string', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::AttributeChecker::HasUnderlyingTypeInformation() const [member function]
    cls.add_method('HasUnderlyingTypeInformation', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3AttributeValue_methods(root_module, cls):
    ## attribute.h (module 'core'): ns3::AttributeValue::AttributeValue(ns3::AttributeValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::AttributeValue const &', 'arg0')])
    ## attribute.h (module 'core'): ns3::AttributeValue::AttributeValue() [constructor]
    cls.add_constructor([])
    ## attribute.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::AttributeValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::AttributeValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_pure_virtual=True, is_virtual=True)
    ## attribute.h (module 'core'): std::string ns3::AttributeValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3BooleanChecker_methods(root_module, cls):
    ## boolean.h (module 'core'): ns3::BooleanChecker::BooleanChecker() [constructor]
    cls.add_constructor([])
    ## boolean.h (module 'core'): ns3::BooleanChecker::BooleanChecker(ns3::BooleanChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::BooleanChecker const &', 'arg0')])
    return

def register_Ns3BooleanValue_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## boolean.h (module 'core'): ns3::BooleanValue::BooleanValue(ns3::BooleanValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::BooleanValue const &', 'arg0')])
    ## boolean.h (module 'core'): ns3::BooleanValue::BooleanValue() [constructor]
    cls.add_constructor([])
    ## boolean.h (module 'core'): ns3::BooleanValue::BooleanValue(bool value) [constructor]
    cls.add_constructor([param('bool', 'value')])
    ## boolean.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::BooleanValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## boolean.h (module 'core'): bool ns3::BooleanValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## boolean.h (module 'core'): bool ns3::BooleanValue::Get() const [member function]
    cls.add_method('Get', 
                   'bool', 
                   [], 
                   is_const=True)
    ## boolean.h (module 'core'): std::string ns3::BooleanValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## boolean.h (module 'core'): void ns3::BooleanValue::Set(bool value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('bool', 'value')])
    return

def register_Ns3CallbackChecker_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackChecker::CallbackChecker() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackChecker::CallbackChecker(ns3::CallbackChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackChecker const &', 'arg0')])
    return

def register_Ns3CallbackImplBase_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImplBase::CallbackImplBase() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImplBase::CallbackImplBase(ns3::CallbackImplBase const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImplBase const &', 'arg0')])
    ## callback.h (module 'core'): std::string ns3::CallbackImplBase::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## callback.h (module 'core'): bool ns3::CallbackImplBase::IsEqual(ns3::Ptr<const ns3::CallbackImplBase> other) const [member function]
    cls.add_method('IsEqual', 
                   'bool', 
                   [param('ns3::Ptr< ns3::CallbackImplBase const >', 'other')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::Demangle(std::string const & mangled) [member function]
    cls.add_method('Demangle', 
                   'std::string', 
                   [param('std::string const &', 'mangled')], 
                   is_static=True, visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::ObjectBase*'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['void'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Ptr<ns3::NetDevice> '], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Ptr<ns3::Packet const> '], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['unsigned short'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Address const&'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::NetDevice::PacketType'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['bool'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Ptr<ns3::Packet> '], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Mac48Address'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['unsigned int'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Time'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::dot11s::RouteChange'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['unsigned long long'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::WifiMacDropReason'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Ptr<ns3::WifiMacQueueItem const> '], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['unsigned char'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Ptr<ns3::WifiPsdu const> '], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['std::unordered_map<unsigned short', ' ns3::Ptr<ns3::WifiPsdu>', ' std::hash<unsigned short>', ' std::equal_to<unsigned short>', ' std::allocator<std::pair<unsigned short const', ' ns3::Ptr<ns3::WifiPsdu> > > >*'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['std::set<ns3::Mac48Address', ' std::less<ns3::Mac48Address>', ' std::allocator<ns3::Mac48Address> > const*'], visibility='protected')
    return

def register_Ns3CallbackValue_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackValue::CallbackValue(ns3::CallbackValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackValue const &', 'arg0')])
    ## callback.h (module 'core'): ns3::CallbackValue::CallbackValue() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackValue::CallbackValue(ns3::CallbackBase const & base) [constructor]
    cls.add_constructor([param('ns3::CallbackBase const &', 'base')])
    ## callback.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::CallbackValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): bool ns3::CallbackValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## callback.h (module 'core'): std::string ns3::CallbackValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackValue::Set(ns3::CallbackBase base) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::CallbackBase', 'base')])
    return

def register_Ns3Channel_methods(root_module, cls):
    ## channel.h (module 'network'): ns3::Channel::Channel(ns3::Channel const & arg0) [constructor]
    cls.add_constructor([param('ns3::Channel const &', 'arg0')])
    ## channel.h (module 'network'): ns3::Channel::Channel() [constructor]
    cls.add_constructor([])
    ## channel.h (module 'network'): ns3::Ptr<ns3::NetDevice> ns3::Channel::GetDevice(std::size_t i) const [member function]
    cls.add_method('GetDevice', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [param('std::size_t', 'i')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## channel.h (module 'network'): uint32_t ns3::Channel::GetId() const [member function]
    cls.add_method('GetId', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## channel.h (module 'network'): std::size_t ns3::Channel::GetNDevices() const [member function]
    cls.add_method('GetNDevices', 
                   'std::size_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## channel.h (module 'network'): static ns3::TypeId ns3::Channel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    return

def register_Ns3ConstantRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::ConstantRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::ConstantRandomVariable::ConstantRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::ConstantRandomVariable::GetConstant() const [member function]
    cls.add_method('GetConstant', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ConstantRandomVariable::GetValue(double constant) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('double', 'constant')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ConstantRandomVariable::GetInteger(uint32_t constant) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'constant')])
    ## random-variable-stream.h (module 'core'): double ns3::ConstantRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ConstantRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3DataRateChecker_methods(root_module, cls):
    ## data-rate.h (module 'network'): ns3::DataRateChecker::DataRateChecker() [constructor]
    cls.add_constructor([])
    ## data-rate.h (module 'network'): ns3::DataRateChecker::DataRateChecker(ns3::DataRateChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::DataRateChecker const &', 'arg0')])
    return

def register_Ns3DataRateValue_methods(root_module, cls):
    ## data-rate.h (module 'network'): ns3::DataRateValue::DataRateValue() [constructor]
    cls.add_constructor([])
    ## data-rate.h (module 'network'): ns3::DataRateValue::DataRateValue(ns3::DataRate const & value) [constructor]
    cls.add_constructor([param('ns3::DataRate const &', 'value')])
    ## data-rate.h (module 'network'): ns3::DataRateValue::DataRateValue(ns3::DataRateValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::DataRateValue const &', 'arg0')])
    ## data-rate.h (module 'network'): ns3::Ptr<ns3::AttributeValue> ns3::DataRateValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## data-rate.h (module 'network'): bool ns3::DataRateValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## data-rate.h (module 'network'): ns3::DataRate ns3::DataRateValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::DataRate', 
                   [], 
                   is_const=True)
    ## data-rate.h (module 'network'): std::string ns3::DataRateValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## data-rate.h (module 'network'): void ns3::DataRateValue::Set(ns3::DataRate const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::DataRate const &', 'value')])
    return

def register_Ns3DeterministicRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::DeterministicRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::DeterministicRandomVariable::DeterministicRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): void ns3::DeterministicRandomVariable::SetValueArray(double * values, std::size_t length) [member function]
    cls.add_method('SetValueArray', 
                   'void', 
                   [param('double *', 'values'), param('std::size_t', 'length')])
    ## random-variable-stream.h (module 'core'): double ns3::DeterministicRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::DeterministicRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3DoubleValue_methods(root_module, cls):
    ## double.h (module 'core'): ns3::DoubleValue::DoubleValue() [constructor]
    cls.add_constructor([])
    ## double.h (module 'core'): ns3::DoubleValue::DoubleValue(double const & value) [constructor]
    cls.add_constructor([param('double const &', 'value')])
    ## double.h (module 'core'): ns3::DoubleValue::DoubleValue(ns3::DoubleValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::DoubleValue const &', 'arg0')])
    ## double.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::DoubleValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## double.h (module 'core'): bool ns3::DoubleValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## double.h (module 'core'): double ns3::DoubleValue::Get() const [member function]
    cls.add_method('Get', 
                   'double', 
                   [], 
                   is_const=True)
    ## double.h (module 'core'): std::string ns3::DoubleValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## double.h (module 'core'): void ns3::DoubleValue::Set(double const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('double const &', 'value')])
    return

def register_Ns3DsssParameterSet_methods(root_module, cls):
    ## dsss-parameter-set.h (module 'wifi'): ns3::DsssParameterSet::DsssParameterSet(ns3::DsssParameterSet const & arg0) [constructor]
    cls.add_constructor([param('ns3::DsssParameterSet const &', 'arg0')])
    ## dsss-parameter-set.h (module 'wifi'): ns3::DsssParameterSet::DsssParameterSet() [constructor]
    cls.add_constructor([])
    ## dsss-parameter-set.h (module 'wifi'): uint8_t ns3::DsssParameterSet::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## dsss-parameter-set.h (module 'wifi'): ns3::WifiInformationElementId ns3::DsssParameterSet::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## dsss-parameter-set.h (module 'wifi'): uint8_t ns3::DsssParameterSet::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## dsss-parameter-set.h (module 'wifi'): uint16_t ns3::DsssParameterSet::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## dsss-parameter-set.h (module 'wifi'): ns3::Buffer::Iterator ns3::DsssParameterSet::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## dsss-parameter-set.h (module 'wifi'): void ns3::DsssParameterSet::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## dsss-parameter-set.h (module 'wifi'): void ns3::DsssParameterSet::SetCurrentChannel(uint8_t currentChannel) [member function]
    cls.add_method('SetCurrentChannel', 
                   'void', 
                   [param('uint8_t', 'currentChannel')])
    ## dsss-parameter-set.h (module 'wifi'): void ns3::DsssParameterSet::SetDsssSupported(uint8_t dsssSupported) [member function]
    cls.add_method('SetDsssSupported', 
                   'void', 
                   [param('uint8_t', 'dsssSupported')])
    return

def register_Ns3EdcaParameterSet_methods(root_module, cls):
    ## edca-parameter-set.h (module 'wifi'): ns3::EdcaParameterSet::EdcaParameterSet(ns3::EdcaParameterSet const & arg0) [constructor]
    cls.add_constructor([param('ns3::EdcaParameterSet const &', 'arg0')])
    ## edca-parameter-set.h (module 'wifi'): ns3::EdcaParameterSet::EdcaParameterSet() [constructor]
    cls.add_constructor([])
    ## edca-parameter-set.h (module 'wifi'): uint8_t ns3::EdcaParameterSet::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## edca-parameter-set.h (module 'wifi'): ns3::WifiInformationElementId ns3::EdcaParameterSet::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## edca-parameter-set.h (module 'wifi'): uint8_t ns3::EdcaParameterSet::GetBeAifsn() const [member function]
    cls.add_method('GetBeAifsn', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint32_t ns3::EdcaParameterSet::GetBeCWmax() const [member function]
    cls.add_method('GetBeCWmax', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint32_t ns3::EdcaParameterSet::GetBeCWmin() const [member function]
    cls.add_method('GetBeCWmin', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint16_t ns3::EdcaParameterSet::GetBeTxopLimit() const [member function]
    cls.add_method('GetBeTxopLimit', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint8_t ns3::EdcaParameterSet::GetBkAifsn() const [member function]
    cls.add_method('GetBkAifsn', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint32_t ns3::EdcaParameterSet::GetBkCWmax() const [member function]
    cls.add_method('GetBkCWmax', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint32_t ns3::EdcaParameterSet::GetBkCWmin() const [member function]
    cls.add_method('GetBkCWmin', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint16_t ns3::EdcaParameterSet::GetBkTxopLimit() const [member function]
    cls.add_method('GetBkTxopLimit', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint8_t ns3::EdcaParameterSet::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## edca-parameter-set.h (module 'wifi'): uint8_t ns3::EdcaParameterSet::GetQosInfo() const [member function]
    cls.add_method('GetQosInfo', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint16_t ns3::EdcaParameterSet::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## edca-parameter-set.h (module 'wifi'): uint8_t ns3::EdcaParameterSet::GetViAifsn() const [member function]
    cls.add_method('GetViAifsn', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint32_t ns3::EdcaParameterSet::GetViCWmax() const [member function]
    cls.add_method('GetViCWmax', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint32_t ns3::EdcaParameterSet::GetViCWmin() const [member function]
    cls.add_method('GetViCWmin', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint16_t ns3::EdcaParameterSet::GetViTxopLimit() const [member function]
    cls.add_method('GetViTxopLimit', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint8_t ns3::EdcaParameterSet::GetVoAifsn() const [member function]
    cls.add_method('GetVoAifsn', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint32_t ns3::EdcaParameterSet::GetVoCWmax() const [member function]
    cls.add_method('GetVoCWmax', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint32_t ns3::EdcaParameterSet::GetVoCWmin() const [member function]
    cls.add_method('GetVoCWmin', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint16_t ns3::EdcaParameterSet::GetVoTxopLimit() const [member function]
    cls.add_method('GetVoTxopLimit', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): uint8_t ns3::EdcaParameterSet::IsQosSupported() const [member function]
    cls.add_method('IsQosSupported', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## edca-parameter-set.h (module 'wifi'): ns3::Buffer::Iterator ns3::EdcaParameterSet::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetBeAci(uint8_t aci) [member function]
    cls.add_method('SetBeAci', 
                   'void', 
                   [param('uint8_t', 'aci')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetBeAifsn(uint8_t aifsn) [member function]
    cls.add_method('SetBeAifsn', 
                   'void', 
                   [param('uint8_t', 'aifsn')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetBeCWmax(uint32_t cwMax) [member function]
    cls.add_method('SetBeCWmax', 
                   'void', 
                   [param('uint32_t', 'cwMax')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetBeCWmin(uint32_t cwMin) [member function]
    cls.add_method('SetBeCWmin', 
                   'void', 
                   [param('uint32_t', 'cwMin')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetBeTxopLimit(uint16_t txop) [member function]
    cls.add_method('SetBeTxopLimit', 
                   'void', 
                   [param('uint16_t', 'txop')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetBkAci(uint8_t aci) [member function]
    cls.add_method('SetBkAci', 
                   'void', 
                   [param('uint8_t', 'aci')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetBkAifsn(uint8_t aifsn) [member function]
    cls.add_method('SetBkAifsn', 
                   'void', 
                   [param('uint8_t', 'aifsn')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetBkCWmax(uint32_t cwMax) [member function]
    cls.add_method('SetBkCWmax', 
                   'void', 
                   [param('uint32_t', 'cwMax')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetBkCWmin(uint32_t cwMin) [member function]
    cls.add_method('SetBkCWmin', 
                   'void', 
                   [param('uint32_t', 'cwMin')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetBkTxopLimit(uint16_t txop) [member function]
    cls.add_method('SetBkTxopLimit', 
                   'void', 
                   [param('uint16_t', 'txop')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetQosInfo(uint8_t qosInfo) [member function]
    cls.add_method('SetQosInfo', 
                   'void', 
                   [param('uint8_t', 'qosInfo')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetQosSupported(uint8_t qosSupported) [member function]
    cls.add_method('SetQosSupported', 
                   'void', 
                   [param('uint8_t', 'qosSupported')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetViAci(uint8_t aci) [member function]
    cls.add_method('SetViAci', 
                   'void', 
                   [param('uint8_t', 'aci')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetViAifsn(uint8_t aifsn) [member function]
    cls.add_method('SetViAifsn', 
                   'void', 
                   [param('uint8_t', 'aifsn')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetViCWmax(uint32_t cwMax) [member function]
    cls.add_method('SetViCWmax', 
                   'void', 
                   [param('uint32_t', 'cwMax')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetViCWmin(uint32_t cwMin) [member function]
    cls.add_method('SetViCWmin', 
                   'void', 
                   [param('uint32_t', 'cwMin')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetViTxopLimit(uint16_t txop) [member function]
    cls.add_method('SetViTxopLimit', 
                   'void', 
                   [param('uint16_t', 'txop')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetVoAci(uint8_t aci) [member function]
    cls.add_method('SetVoAci', 
                   'void', 
                   [param('uint8_t', 'aci')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetVoAifsn(uint8_t aifsn) [member function]
    cls.add_method('SetVoAifsn', 
                   'void', 
                   [param('uint8_t', 'aifsn')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetVoCWmax(uint32_t cwMax) [member function]
    cls.add_method('SetVoCWmax', 
                   'void', 
                   [param('uint32_t', 'cwMax')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetVoCWmin(uint32_t cwMin) [member function]
    cls.add_method('SetVoCWmin', 
                   'void', 
                   [param('uint32_t', 'cwMin')])
    ## edca-parameter-set.h (module 'wifi'): void ns3::EdcaParameterSet::SetVoTxopLimit(uint16_t txop) [member function]
    cls.add_method('SetVoTxopLimit', 
                   'void', 
                   [param('uint16_t', 'txop')])
    return

def register_Ns3EmpiricalRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): ns3::EmpiricalRandomVariable::EmpiricalRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): void ns3::EmpiricalRandomVariable::CDF(double v, double c) [member function]
    cls.add_method('CDF', 
                   'void', 
                   [param('double', 'v'), param('double', 'c')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::EmpiricalRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::EmpiricalRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): double ns3::EmpiricalRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): double ns3::EmpiricalRandomVariable::Interpolate() [member function]
    cls.add_method('Interpolate', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): bool ns3::EmpiricalRandomVariable::SetInterpolate(bool interpolate) [member function]
    cls.add_method('SetInterpolate', 
                   'bool', 
                   [param('bool', 'interpolate')])
    return

def register_Ns3EmptyAttributeAccessor_methods(root_module, cls):
    ## attribute.h (module 'core'): ns3::EmptyAttributeAccessor::EmptyAttributeAccessor(ns3::EmptyAttributeAccessor const & arg0) [constructor]
    cls.add_constructor([param('ns3::EmptyAttributeAccessor const &', 'arg0')])
    ## attribute.h (module 'core'): ns3::EmptyAttributeAccessor::EmptyAttributeAccessor() [constructor]
    cls.add_constructor([])
    ## attribute.h (module 'core'): bool ns3::EmptyAttributeAccessor::Get(ns3::ObjectBase const * object, ns3::AttributeValue & attribute) const [member function]
    cls.add_method('Get', 
                   'bool', 
                   [param('ns3::ObjectBase const *', 'object'), param('ns3::AttributeValue &', 'attribute')], 
                   is_const=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::EmptyAttributeAccessor::HasGetter() const [member function]
    cls.add_method('HasGetter', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::EmptyAttributeAccessor::HasSetter() const [member function]
    cls.add_method('HasSetter', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::EmptyAttributeAccessor::Set(ns3::ObjectBase * object, ns3::AttributeValue const & value) const [member function]
    cls.add_method('Set', 
                   'bool', 
                   [param('ns3::ObjectBase *', 'object'), param('ns3::AttributeValue const &', 'value')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3EmptyAttributeChecker_methods(root_module, cls):
    ## attribute.h (module 'core'): ns3::EmptyAttributeChecker::EmptyAttributeChecker(ns3::EmptyAttributeChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::EmptyAttributeChecker const &', 'arg0')])
    ## attribute.h (module 'core'): ns3::EmptyAttributeChecker::EmptyAttributeChecker() [constructor]
    cls.add_constructor([])
    ## attribute.h (module 'core'): bool ns3::EmptyAttributeChecker::Check(ns3::AttributeValue const & value) const [member function]
    cls.add_method('Check', 
                   'bool', 
                   [param('ns3::AttributeValue const &', 'value')], 
                   is_const=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::EmptyAttributeChecker::Copy(ns3::AttributeValue const & source, ns3::AttributeValue & destination) const [member function]
    cls.add_method('Copy', 
                   'bool', 
                   [param('ns3::AttributeValue const &', 'source'), param('ns3::AttributeValue &', 'destination')], 
                   is_const=True, is_virtual=True)
    ## attribute.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::EmptyAttributeChecker::Create() const [member function]
    cls.add_method('Create', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## attribute.h (module 'core'): std::string ns3::EmptyAttributeChecker::GetUnderlyingTypeInformation() const [member function]
    cls.add_method('GetUnderlyingTypeInformation', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## attribute.h (module 'core'): std::string ns3::EmptyAttributeChecker::GetValueTypeName() const [member function]
    cls.add_method('GetValueTypeName', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## attribute.h (module 'core'): bool ns3::EmptyAttributeChecker::HasUnderlyingTypeInformation() const [member function]
    cls.add_method('HasUnderlyingTypeInformation', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3EmptyAttributeValue_methods(root_module, cls):
    ## attribute.h (module 'core'): ns3::EmptyAttributeValue::EmptyAttributeValue(ns3::EmptyAttributeValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::EmptyAttributeValue const &', 'arg0')])
    ## attribute.h (module 'core'): ns3::EmptyAttributeValue::EmptyAttributeValue() [constructor]
    cls.add_constructor([])
    ## attribute.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::EmptyAttributeValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True, visibility='private')
    ## attribute.h (module 'core'): bool ns3::EmptyAttributeValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True, visibility='private')
    ## attribute.h (module 'core'): std::string ns3::EmptyAttributeValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True, visibility='private')
    return

def register_Ns3EnumChecker_methods(root_module, cls):
    ## enum.h (module 'core'): ns3::EnumChecker::EnumChecker(ns3::EnumChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::EnumChecker const &', 'arg0')])
    ## enum.h (module 'core'): ns3::EnumChecker::EnumChecker() [constructor]
    cls.add_constructor([])
    ## enum.h (module 'core'): void ns3::EnumChecker::Add(int value, std::string name) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('int', 'value'), param('std::string', 'name')])
    ## enum.h (module 'core'): void ns3::EnumChecker::AddDefault(int value, std::string name) [member function]
    cls.add_method('AddDefault', 
                   'void', 
                   [param('int', 'value'), param('std::string', 'name')])
    ## enum.h (module 'core'): bool ns3::EnumChecker::Check(ns3::AttributeValue const & value) const [member function]
    cls.add_method('Check', 
                   'bool', 
                   [param('ns3::AttributeValue const &', 'value')], 
                   is_const=True, is_virtual=True)
    ## enum.h (module 'core'): bool ns3::EnumChecker::Copy(ns3::AttributeValue const & src, ns3::AttributeValue & dst) const [member function]
    cls.add_method('Copy', 
                   'bool', 
                   [param('ns3::AttributeValue const &', 'src'), param('ns3::AttributeValue &', 'dst')], 
                   is_const=True, is_virtual=True)
    ## enum.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::EnumChecker::Create() const [member function]
    cls.add_method('Create', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## enum.h (module 'core'): std::string ns3::EnumChecker::GetName(int value) const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [param('int', 'value')], 
                   is_const=True)
    ## enum.h (module 'core'): std::string ns3::EnumChecker::GetUnderlyingTypeInformation() const [member function]
    cls.add_method('GetUnderlyingTypeInformation', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## enum.h (module 'core'): int ns3::EnumChecker::GetValue(std::string const name) const [member function]
    cls.add_method('GetValue', 
                   'int', 
                   [param('std::string const', 'name')], 
                   is_const=True)
    ## enum.h (module 'core'): std::string ns3::EnumChecker::GetValueTypeName() const [member function]
    cls.add_method('GetValueTypeName', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## enum.h (module 'core'): bool ns3::EnumChecker::HasUnderlyingTypeInformation() const [member function]
    cls.add_method('HasUnderlyingTypeInformation', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3EnumValue_methods(root_module, cls):
    ## enum.h (module 'core'): ns3::EnumValue::EnumValue(ns3::EnumValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::EnumValue const &', 'arg0')])
    ## enum.h (module 'core'): ns3::EnumValue::EnumValue() [constructor]
    cls.add_constructor([])
    ## enum.h (module 'core'): ns3::EnumValue::EnumValue(int value) [constructor]
    cls.add_constructor([param('int', 'value')])
    ## enum.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::EnumValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## enum.h (module 'core'): bool ns3::EnumValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## enum.h (module 'core'): int ns3::EnumValue::Get() const [member function]
    cls.add_method('Get', 
                   'int', 
                   [], 
                   is_const=True)
    ## enum.h (module 'core'): std::string ns3::EnumValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## enum.h (module 'core'): void ns3::EnumValue::Set(int value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('int', 'value')])
    return

def register_Ns3ErlangRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::ErlangRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::ErlangRandomVariable::ErlangRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ErlangRandomVariable::GetK() const [member function]
    cls.add_method('GetK', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ErlangRandomVariable::GetLambda() const [member function]
    cls.add_method('GetLambda', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ErlangRandomVariable::GetValue(uint32_t k, double lambda) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('uint32_t', 'k'), param('double', 'lambda')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ErlangRandomVariable::GetInteger(uint32_t k, uint32_t lambda) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'k'), param('uint32_t', 'lambda')])
    ## random-variable-stream.h (module 'core'): double ns3::ErlangRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ErlangRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3ErpInformation_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## erp-information.h (module 'wifi'): ns3::ErpInformation::ErpInformation(ns3::ErpInformation const & arg0) [constructor]
    cls.add_constructor([param('ns3::ErpInformation const &', 'arg0')])
    ## erp-information.h (module 'wifi'): ns3::ErpInformation::ErpInformation() [constructor]
    cls.add_constructor([])
    ## erp-information.h (module 'wifi'): uint8_t ns3::ErpInformation::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## erp-information.h (module 'wifi'): ns3::WifiInformationElementId ns3::ErpInformation::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## erp-information.h (module 'wifi'): uint8_t ns3::ErpInformation::GetBarkerPreambleMode() const [member function]
    cls.add_method('GetBarkerPreambleMode', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## erp-information.h (module 'wifi'): uint8_t ns3::ErpInformation::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## erp-information.h (module 'wifi'): uint8_t ns3::ErpInformation::GetNonErpPresent() const [member function]
    cls.add_method('GetNonErpPresent', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## erp-information.h (module 'wifi'): uint16_t ns3::ErpInformation::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## erp-information.h (module 'wifi'): uint8_t ns3::ErpInformation::GetUseProtection() const [member function]
    cls.add_method('GetUseProtection', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## erp-information.h (module 'wifi'): ns3::Buffer::Iterator ns3::ErpInformation::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## erp-information.h (module 'wifi'): void ns3::ErpInformation::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## erp-information.h (module 'wifi'): void ns3::ErpInformation::SetBarkerPreambleMode(uint8_t barkerPreambleMode) [member function]
    cls.add_method('SetBarkerPreambleMode', 
                   'void', 
                   [param('uint8_t', 'barkerPreambleMode')])
    ## erp-information.h (module 'wifi'): void ns3::ErpInformation::SetErpSupported(uint8_t erpSupported) [member function]
    cls.add_method('SetErpSupported', 
                   'void', 
                   [param('uint8_t', 'erpSupported')])
    ## erp-information.h (module 'wifi'): void ns3::ErpInformation::SetNonErpPresent(uint8_t nonErpPresent) [member function]
    cls.add_method('SetNonErpPresent', 
                   'void', 
                   [param('uint8_t', 'nonErpPresent')])
    ## erp-information.h (module 'wifi'): void ns3::ErpInformation::SetUseProtection(uint8_t useProtection) [member function]
    cls.add_method('SetUseProtection', 
                   'void', 
                   [param('uint8_t', 'useProtection')])
    return

def register_Ns3EventImpl_methods(root_module, cls):
    ## event-impl.h (module 'core'): ns3::EventImpl::EventImpl(ns3::EventImpl const & arg0) [constructor]
    cls.add_constructor([param('ns3::EventImpl const &', 'arg0')])
    ## event-impl.h (module 'core'): ns3::EventImpl::EventImpl() [constructor]
    cls.add_constructor([])
    ## event-impl.h (module 'core'): void ns3::EventImpl::Cancel() [member function]
    cls.add_method('Cancel', 
                   'void', 
                   [])
    ## event-impl.h (module 'core'): void ns3::EventImpl::Invoke() [member function]
    cls.add_method('Invoke', 
                   'void', 
                   [])
    ## event-impl.h (module 'core'): bool ns3::EventImpl::IsCancelled() [member function]
    cls.add_method('IsCancelled', 
                   'bool', 
                   [])
    ## event-impl.h (module 'core'): void ns3::EventImpl::Notify() [member function]
    cls.add_method('Notify', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True, visibility='protected')
    return

def register_Ns3ExponentialRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::ExponentialRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::ExponentialRandomVariable::ExponentialRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::ExponentialRandomVariable::GetMean() const [member function]
    cls.add_method('GetMean', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ExponentialRandomVariable::GetBound() const [member function]
    cls.add_method('GetBound', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ExponentialRandomVariable::GetValue(double mean, double bound) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('double', 'mean'), param('double', 'bound')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ExponentialRandomVariable::GetInteger(uint32_t mean, uint32_t bound) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'mean'), param('uint32_t', 'bound')])
    ## random-variable-stream.h (module 'core'): double ns3::ExponentialRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ExponentialRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3ExtendedCapabilities_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## extended-capabilities.h (module 'wifi'): ns3::ExtendedCapabilities::ExtendedCapabilities(ns3::ExtendedCapabilities const & arg0) [constructor]
    cls.add_constructor([param('ns3::ExtendedCapabilities const &', 'arg0')])
    ## extended-capabilities.h (module 'wifi'): ns3::ExtendedCapabilities::ExtendedCapabilities() [constructor]
    cls.add_constructor([])
    ## extended-capabilities.h (module 'wifi'): uint8_t ns3::ExtendedCapabilities::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## extended-capabilities.h (module 'wifi'): ns3::WifiInformationElementId ns3::ExtendedCapabilities::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## extended-capabilities.h (module 'wifi'): uint8_t ns3::ExtendedCapabilities::GetExtendedCapabilitiesByte1() const [member function]
    cls.add_method('GetExtendedCapabilitiesByte1', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## extended-capabilities.h (module 'wifi'): uint8_t ns3::ExtendedCapabilities::GetExtendedCapabilitiesByte2() const [member function]
    cls.add_method('GetExtendedCapabilitiesByte2', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## extended-capabilities.h (module 'wifi'): uint8_t ns3::ExtendedCapabilities::GetExtendedCapabilitiesByte3() const [member function]
    cls.add_method('GetExtendedCapabilitiesByte3', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## extended-capabilities.h (module 'wifi'): uint8_t ns3::ExtendedCapabilities::GetExtendedCapabilitiesByte4() const [member function]
    cls.add_method('GetExtendedCapabilitiesByte4', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## extended-capabilities.h (module 'wifi'): uint8_t ns3::ExtendedCapabilities::GetExtendedCapabilitiesByte5() const [member function]
    cls.add_method('GetExtendedCapabilitiesByte5', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## extended-capabilities.h (module 'wifi'): uint8_t ns3::ExtendedCapabilities::GetExtendedCapabilitiesByte6() const [member function]
    cls.add_method('GetExtendedCapabilitiesByte6', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## extended-capabilities.h (module 'wifi'): uint8_t ns3::ExtendedCapabilities::GetExtendedCapabilitiesByte7() const [member function]
    cls.add_method('GetExtendedCapabilitiesByte7', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## extended-capabilities.h (module 'wifi'): uint8_t ns3::ExtendedCapabilities::GetExtendedCapabilitiesByte8() const [member function]
    cls.add_method('GetExtendedCapabilitiesByte8', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## extended-capabilities.h (module 'wifi'): uint8_t ns3::ExtendedCapabilities::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## extended-capabilities.h (module 'wifi'): uint16_t ns3::ExtendedCapabilities::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## extended-capabilities.h (module 'wifi'): ns3::Buffer::Iterator ns3::ExtendedCapabilities::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SetExtendedCapabilitiesByte1(uint8_t ctrl) [member function]
    cls.add_method('SetExtendedCapabilitiesByte1', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SetExtendedCapabilitiesByte2(uint8_t ctrl) [member function]
    cls.add_method('SetExtendedCapabilitiesByte2', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SetExtendedCapabilitiesByte3(uint8_t ctrl) [member function]
    cls.add_method('SetExtendedCapabilitiesByte3', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SetExtendedCapabilitiesByte4(uint8_t ctrl) [member function]
    cls.add_method('SetExtendedCapabilitiesByte4', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SetExtendedCapabilitiesByte5(uint8_t ctrl) [member function]
    cls.add_method('SetExtendedCapabilitiesByte5', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SetExtendedCapabilitiesByte6(uint8_t ctrl) [member function]
    cls.add_method('SetExtendedCapabilitiesByte6', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SetExtendedCapabilitiesByte7(uint8_t ctrl) [member function]
    cls.add_method('SetExtendedCapabilitiesByte7', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SetExtendedCapabilitiesByte8(uint8_t ctrl) [member function]
    cls.add_method('SetExtendedCapabilitiesByte8', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SetHtSupported(uint8_t htSupported) [member function]
    cls.add_method('SetHtSupported', 
                   'void', 
                   [param('uint8_t', 'htSupported')])
    ## extended-capabilities.h (module 'wifi'): void ns3::ExtendedCapabilities::SetVhtSupported(uint8_t vhtSupported) [member function]
    cls.add_method('SetVhtSupported', 
                   'void', 
                   [param('uint8_t', 'vhtSupported')])
    return

def register_Ns3ExtendedSupportedRatesIE_methods(root_module, cls):
    ## supported-rates.h (module 'wifi'): ns3::ExtendedSupportedRatesIE::ExtendedSupportedRatesIE(ns3::ExtendedSupportedRatesIE const & arg0) [constructor]
    cls.add_constructor([param('ns3::ExtendedSupportedRatesIE const &', 'arg0')])
    ## supported-rates.h (module 'wifi'): ns3::ExtendedSupportedRatesIE::ExtendedSupportedRatesIE() [constructor]
    cls.add_constructor([])
    ## supported-rates.h (module 'wifi'): ns3::ExtendedSupportedRatesIE::ExtendedSupportedRatesIE(ns3::SupportedRates * rates) [constructor]
    cls.add_constructor([param('ns3::SupportedRates *', 'rates')])
    ## supported-rates.h (module 'wifi'): uint8_t ns3::ExtendedSupportedRatesIE::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## supported-rates.h (module 'wifi'): ns3::WifiInformationElementId ns3::ExtendedSupportedRatesIE::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## supported-rates.h (module 'wifi'): uint8_t ns3::ExtendedSupportedRatesIE::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## supported-rates.h (module 'wifi'): uint16_t ns3::ExtendedSupportedRatesIE::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## supported-rates.h (module 'wifi'): ns3::Buffer::Iterator ns3::ExtendedSupportedRatesIE::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## supported-rates.h (module 'wifi'): void ns3::ExtendedSupportedRatesIE::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## supported-rates.h (module 'wifi'): void ns3::ExtendedSupportedRatesIE::SetSupportedRates(ns3::SupportedRates * rates) [member function]
    cls.add_method('SetSupportedRates', 
                   'void', 
                   [param('ns3::SupportedRates *', 'rates')])
    return

def register_Ns3GammaRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::GammaRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::GammaRandomVariable::GammaRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::GammaRandomVariable::GetAlpha() const [member function]
    cls.add_method('GetAlpha', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::GammaRandomVariable::GetBeta() const [member function]
    cls.add_method('GetBeta', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::GammaRandomVariable::GetValue(double alpha, double beta) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('double', 'alpha'), param('double', 'beta')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::GammaRandomVariable::GetInteger(uint32_t alpha, uint32_t beta) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'alpha'), param('uint32_t', 'beta')])
    ## random-variable-stream.h (module 'core'): double ns3::GammaRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::GammaRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3HeCapabilities_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## he-capabilities.h (module 'wifi'): ns3::HeCapabilities::HeCapabilities(ns3::HeCapabilities const & arg0) [constructor]
    cls.add_constructor([param('ns3::HeCapabilities const &', 'arg0')])
    ## he-capabilities.h (module 'wifi'): ns3::HeCapabilities::HeCapabilities() [constructor]
    cls.add_constructor([])
    ## he-capabilities.h (module 'wifi'): uint8_t ns3::HeCapabilities::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## he-capabilities.h (module 'wifi'): ns3::WifiInformationElementId ns3::HeCapabilities::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## he-capabilities.h (module 'wifi'): ns3::WifiInformationElementId ns3::HeCapabilities::ElementIdExt() const [member function]
    cls.add_method('ElementIdExt', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## he-capabilities.h (module 'wifi'): uint8_t ns3::HeCapabilities::GetChannelWidthSet() const [member function]
    cls.add_method('GetChannelWidthSet', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): uint32_t ns3::HeCapabilities::GetHeMacCapabilitiesInfo1() const [member function]
    cls.add_method('GetHeMacCapabilitiesInfo1', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): uint16_t ns3::HeCapabilities::GetHeMacCapabilitiesInfo2() const [member function]
    cls.add_method('GetHeMacCapabilitiesInfo2', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): uint64_t ns3::HeCapabilities::GetHePhyCapabilitiesInfo1() const [member function]
    cls.add_method('GetHePhyCapabilitiesInfo1', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): uint16_t ns3::HeCapabilities::GetHePhyCapabilitiesInfo2() const [member function]
    cls.add_method('GetHePhyCapabilitiesInfo2', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): uint8_t ns3::HeCapabilities::GetHePhyCapabilitiesInfo3() const [member function]
    cls.add_method('GetHePhyCapabilitiesInfo3', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): bool ns3::HeCapabilities::GetHePpdu4xHeLtf800nsGi() const [member function]
    cls.add_method('GetHePpdu4xHeLtf800nsGi', 
                   'bool', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): bool ns3::HeCapabilities::GetHeSuPpdu1xHeLtf800nsGi() const [member function]
    cls.add_method('GetHeSuPpdu1xHeLtf800nsGi', 
                   'bool', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): uint8_t ns3::HeCapabilities::GetHighestMcsSupported() const [member function]
    cls.add_method('GetHighestMcsSupported', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): uint8_t ns3::HeCapabilities::GetHighestNssSupported() const [member function]
    cls.add_method('GetHighestNssSupported', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): uint8_t ns3::HeCapabilities::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## he-capabilities.h (module 'wifi'): uint8_t ns3::HeCapabilities::GetLdpcCodingInPayload() const [member function]
    cls.add_method('GetLdpcCodingInPayload', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): uint32_t ns3::HeCapabilities::GetMaxAmpduLength() const [member function]
    cls.add_method('GetMaxAmpduLength', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): uint16_t ns3::HeCapabilities::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## he-capabilities.h (module 'wifi'): uint16_t ns3::HeCapabilities::GetSupportedMcsAndNss() const [member function]
    cls.add_method('GetSupportedMcsAndNss', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): bool ns3::HeCapabilities::IsSupportedRxMcs(uint8_t mcs) const [member function]
    cls.add_method('IsSupportedRxMcs', 
                   'bool', 
                   [param('uint8_t', 'mcs')], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): bool ns3::HeCapabilities::IsSupportedTxMcs(uint8_t mcs) const [member function]
    cls.add_method('IsSupportedTxMcs', 
                   'bool', 
                   [param('uint8_t', 'mcs')], 
                   is_const=True)
    ## he-capabilities.h (module 'wifi'): ns3::Buffer::Iterator ns3::HeCapabilities::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetChannelWidthSet(uint8_t channelWidthSet) [member function]
    cls.add_method('SetChannelWidthSet', 
                   'void', 
                   [param('uint8_t', 'channelWidthSet')])
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetHeMacCapabilitiesInfo(uint32_t ctrl1, uint16_t ctrl2) [member function]
    cls.add_method('SetHeMacCapabilitiesInfo', 
                   'void', 
                   [param('uint32_t', 'ctrl1'), param('uint16_t', 'ctrl2')])
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetHePhyCapabilitiesInfo(uint64_t ctrl1, uint16_t ctrl2, uint8_t ctrl3) [member function]
    cls.add_method('SetHePhyCapabilitiesInfo', 
                   'void', 
                   [param('uint64_t', 'ctrl1'), param('uint16_t', 'ctrl2'), param('uint8_t', 'ctrl3')])
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetHePpdu4xHeLtf800nsGi(bool heSuPpdu4xHeLtf800nsGi) [member function]
    cls.add_method('SetHePpdu4xHeLtf800nsGi', 
                   'void', 
                   [param('bool', 'heSuPpdu4xHeLtf800nsGi')])
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetHeSuPpdu1xHeLtf800nsGi(bool heSuPpdu1xHeLtf800nsGi) [member function]
    cls.add_method('SetHeSuPpdu1xHeLtf800nsGi', 
                   'void', 
                   [param('bool', 'heSuPpdu1xHeLtf800nsGi')])
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetHeSupported(uint8_t heSupported) [member function]
    cls.add_method('SetHeSupported', 
                   'void', 
                   [param('uint8_t', 'heSupported')])
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetHighestMcsSupported(uint8_t mcs) [member function]
    cls.add_method('SetHighestMcsSupported', 
                   'void', 
                   [param('uint8_t', 'mcs')])
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetHighestNssSupported(uint8_t nss) [member function]
    cls.add_method('SetHighestNssSupported', 
                   'void', 
                   [param('uint8_t', 'nss')])
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetLdpcCodingInPayload(uint8_t ldpcCodingInPayload) [member function]
    cls.add_method('SetLdpcCodingInPayload', 
                   'void', 
                   [param('uint8_t', 'ldpcCodingInPayload')])
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetMaxAmpduLength(uint32_t maxAmpduLength) [member function]
    cls.add_method('SetMaxAmpduLength', 
                   'void', 
                   [param('uint32_t', 'maxAmpduLength')])
    ## he-capabilities.h (module 'wifi'): void ns3::HeCapabilities::SetSupportedMcsAndNss(uint16_t ctrl) [member function]
    cls.add_method('SetSupportedMcsAndNss', 
                   'void', 
                   [param('uint16_t', 'ctrl')])
    return

def register_Ns3HeOperation_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## he-operation.h (module 'wifi'): ns3::HeOperation::HeOperation(ns3::HeOperation const & arg0) [constructor]
    cls.add_constructor([param('ns3::HeOperation const &', 'arg0')])
    ## he-operation.h (module 'wifi'): ns3::HeOperation::HeOperation() [constructor]
    cls.add_constructor([])
    ## he-operation.h (module 'wifi'): uint8_t ns3::HeOperation::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## he-operation.h (module 'wifi'): ns3::WifiInformationElementId ns3::HeOperation::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## he-operation.h (module 'wifi'): ns3::WifiInformationElementId ns3::HeOperation::ElementIdExt() const [member function]
    cls.add_method('ElementIdExt', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## he-operation.h (module 'wifi'): uint16_t ns3::HeOperation::GetBasicHeMcsAndNssSet() const [member function]
    cls.add_method('GetBasicHeMcsAndNssSet', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## he-operation.h (module 'wifi'): uint8_t ns3::HeOperation::GetBssColor() const [member function]
    cls.add_method('GetBssColor', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## he-operation.h (module 'wifi'): uint32_t ns3::HeOperation::GetHeOperationParameters() const [member function]
    cls.add_method('GetHeOperationParameters', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## he-operation.h (module 'wifi'): uint8_t ns3::HeOperation::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## he-operation.h (module 'wifi'): uint16_t ns3::HeOperation::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## he-operation.h (module 'wifi'): ns3::Buffer::Iterator ns3::HeOperation::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## he-operation.h (module 'wifi'): void ns3::HeOperation::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## he-operation.h (module 'wifi'): void ns3::HeOperation::SetBssColor(uint8_t bssColor) [member function]
    cls.add_method('SetBssColor', 
                   'void', 
                   [param('uint8_t', 'bssColor')])
    ## he-operation.h (module 'wifi'): void ns3::HeOperation::SetHeOperationParameters(uint32_t ctrl) [member function]
    cls.add_method('SetHeOperationParameters', 
                   'void', 
                   [param('uint32_t', 'ctrl')])
    ## he-operation.h (module 'wifi'): void ns3::HeOperation::SetHeSupported(uint8_t heSupported) [member function]
    cls.add_method('SetHeSupported', 
                   'void', 
                   [param('uint8_t', 'heSupported')])
    ## he-operation.h (module 'wifi'): void ns3::HeOperation::SetMaxHeMcsPerNss(uint8_t nss, uint8_t maxHeMcs) [member function]
    cls.add_method('SetMaxHeMcsPerNss', 
                   'void', 
                   [param('uint8_t', 'nss'), param('uint8_t', 'maxHeMcs')])
    return

def register_Ns3HtCapabilities_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## ht-capabilities.h (module 'wifi'): ns3::HtCapabilities::HtCapabilities(ns3::HtCapabilities const & arg0) [constructor]
    cls.add_constructor([param('ns3::HtCapabilities const &', 'arg0')])
    ## ht-capabilities.h (module 'wifi'): ns3::HtCapabilities::HtCapabilities() [constructor]
    cls.add_constructor([])
    ## ht-capabilities.h (module 'wifi'): uint8_t ns3::HtCapabilities::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ht-capabilities.h (module 'wifi'): ns3::WifiInformationElementId ns3::HtCapabilities::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ht-capabilities.h (module 'wifi'): uint8_t ns3::HtCapabilities::GetAmpduParameters() const [member function]
    cls.add_method('GetAmpduParameters', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint8_t ns3::HtCapabilities::GetAntennaSelectionCapabilities() const [member function]
    cls.add_method('GetAntennaSelectionCapabilities', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint16_t ns3::HtCapabilities::GetExtendedHtCapabilities() const [member function]
    cls.add_method('GetExtendedHtCapabilities', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint16_t ns3::HtCapabilities::GetHtCapabilitiesInfo() const [member function]
    cls.add_method('GetHtCapabilitiesInfo', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint8_t ns3::HtCapabilities::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ht-capabilities.h (module 'wifi'): uint8_t ns3::HtCapabilities::GetLdpc() const [member function]
    cls.add_method('GetLdpc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint32_t ns3::HtCapabilities::GetMaxAmpduLength() const [member function]
    cls.add_method('GetMaxAmpduLength', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint16_t ns3::HtCapabilities::GetMaxAmsduLength() const [member function]
    cls.add_method('GetMaxAmsduLength', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint8_t ns3::HtCapabilities::GetRxHighestSupportedAntennas() const [member function]
    cls.add_method('GetRxHighestSupportedAntennas', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint16_t ns3::HtCapabilities::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ht-capabilities.h (module 'wifi'): uint8_t ns3::HtCapabilities::GetShortGuardInterval20() const [member function]
    cls.add_method('GetShortGuardInterval20', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint8_t ns3::HtCapabilities::GetSupportedChannelWidth() const [member function]
    cls.add_method('GetSupportedChannelWidth', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint64_t ns3::HtCapabilities::GetSupportedMcsSet1() const [member function]
    cls.add_method('GetSupportedMcsSet1', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint64_t ns3::HtCapabilities::GetSupportedMcsSet2() const [member function]
    cls.add_method('GetSupportedMcsSet2', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): uint32_t ns3::HtCapabilities::GetTxBfCapabilities() const [member function]
    cls.add_method('GetTxBfCapabilities', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): bool ns3::HtCapabilities::IsSupportedMcs(uint8_t mcs) const [member function]
    cls.add_method('IsSupportedMcs', 
                   'bool', 
                   [param('uint8_t', 'mcs')], 
                   is_const=True)
    ## ht-capabilities.h (module 'wifi'): ns3::Buffer::Iterator ns3::HtCapabilities::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetAmpduParameters(uint8_t ctrl) [member function]
    cls.add_method('SetAmpduParameters', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetAntennaSelectionCapabilities(uint8_t ctrl) [member function]
    cls.add_method('SetAntennaSelectionCapabilities', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetExtendedHtCapabilities(uint16_t ctrl) [member function]
    cls.add_method('SetExtendedHtCapabilities', 
                   'void', 
                   [param('uint16_t', 'ctrl')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetHtCapabilitiesInfo(uint16_t ctrl) [member function]
    cls.add_method('SetHtCapabilitiesInfo', 
                   'void', 
                   [param('uint16_t', 'ctrl')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetHtSupported(uint8_t htSupported) [member function]
    cls.add_method('SetHtSupported', 
                   'void', 
                   [param('uint8_t', 'htSupported')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetLSigProtectionSupport(uint8_t lSigProtection) [member function]
    cls.add_method('SetLSigProtectionSupport', 
                   'void', 
                   [param('uint8_t', 'lSigProtection')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetLdpc(uint8_t ldpc) [member function]
    cls.add_method('SetLdpc', 
                   'void', 
                   [param('uint8_t', 'ldpc')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetMaxAmpduLength(uint32_t maxAmpduLength) [member function]
    cls.add_method('SetMaxAmpduLength', 
                   'void', 
                   [param('uint32_t', 'maxAmpduLength')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetMaxAmsduLength(uint16_t maxAmsduLength) [member function]
    cls.add_method('SetMaxAmsduLength', 
                   'void', 
                   [param('uint16_t', 'maxAmsduLength')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetRxHighestSupportedDataRate(uint16_t maxSupportedRate) [member function]
    cls.add_method('SetRxHighestSupportedDataRate', 
                   'void', 
                   [param('uint16_t', 'maxSupportedRate')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetRxMcsBitmask(uint8_t index) [member function]
    cls.add_method('SetRxMcsBitmask', 
                   'void', 
                   [param('uint8_t', 'index')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetShortGuardInterval20(uint8_t shortGuardInterval) [member function]
    cls.add_method('SetShortGuardInterval20', 
                   'void', 
                   [param('uint8_t', 'shortGuardInterval')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetShortGuardInterval40(uint8_t shortGuardInterval) [member function]
    cls.add_method('SetShortGuardInterval40', 
                   'void', 
                   [param('uint8_t', 'shortGuardInterval')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetSupportedChannelWidth(uint8_t supportedChannelWidth) [member function]
    cls.add_method('SetSupportedChannelWidth', 
                   'void', 
                   [param('uint8_t', 'supportedChannelWidth')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetSupportedMcsSet(uint64_t ctrl1, uint64_t ctrl2) [member function]
    cls.add_method('SetSupportedMcsSet', 
                   'void', 
                   [param('uint64_t', 'ctrl1'), param('uint64_t', 'ctrl2')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetTxBfCapabilities(uint32_t ctrl) [member function]
    cls.add_method('SetTxBfCapabilities', 
                   'void', 
                   [param('uint32_t', 'ctrl')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetTxMaxNSpatialStreams(uint8_t maxTxSpatialStreams) [member function]
    cls.add_method('SetTxMaxNSpatialStreams', 
                   'void', 
                   [param('uint8_t', 'maxTxSpatialStreams')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetTxMcsSetDefined(uint8_t txMcsSetDefined) [member function]
    cls.add_method('SetTxMcsSetDefined', 
                   'void', 
                   [param('uint8_t', 'txMcsSetDefined')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetTxRxMcsSetUnequal(uint8_t txRxMcsSetUnequal) [member function]
    cls.add_method('SetTxRxMcsSetUnequal', 
                   'void', 
                   [param('uint8_t', 'txRxMcsSetUnequal')])
    ## ht-capabilities.h (module 'wifi'): void ns3::HtCapabilities::SetTxUnequalModulation(uint8_t txUnequalModulation) [member function]
    cls.add_method('SetTxUnequalModulation', 
                   'void', 
                   [param('uint8_t', 'txUnequalModulation')])
    return

def register_Ns3HtOperation_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## ht-operation.h (module 'wifi'): ns3::HtOperation::HtOperation(ns3::HtOperation const & arg0) [constructor]
    cls.add_constructor([param('ns3::HtOperation const &', 'arg0')])
    ## ht-operation.h (module 'wifi'): ns3::HtOperation::HtOperation() [constructor]
    cls.add_constructor([])
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ht-operation.h (module 'wifi'): ns3::WifiInformationElementId ns3::HtOperation::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ht-operation.h (module 'wifi'): uint64_t ns3::HtOperation::GetBasicMcsSet1() const [member function]
    cls.add_method('GetBasicMcsSet1', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint64_t ns3::HtOperation::GetBasicMcsSet2() const [member function]
    cls.add_method('GetBasicMcsSet2', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetDualBeacon() const [member function]
    cls.add_method('GetDualBeacon', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetDualCtsProtection() const [member function]
    cls.add_method('GetDualCtsProtection', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetHtProtection() const [member function]
    cls.add_method('GetHtProtection', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetInformationSubset1() const [member function]
    cls.add_method('GetInformationSubset1', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint16_t ns3::HtOperation::GetInformationSubset2() const [member function]
    cls.add_method('GetInformationSubset2', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint16_t ns3::HtOperation::GetInformationSubset3() const [member function]
    cls.add_method('GetInformationSubset3', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetLSigTxopProtectionFullSupport() const [member function]
    cls.add_method('GetLSigTxopProtectionFullSupport', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetNonGfHtStasPresent() const [member function]
    cls.add_method('GetNonGfHtStasPresent', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetObssNonHtStasPresent() const [member function]
    cls.add_method('GetObssNonHtStasPresent', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetPcoActive() const [member function]
    cls.add_method('GetPcoActive', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetPhase() const [member function]
    cls.add_method('GetPhase', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetPrimaryChannel() const [member function]
    cls.add_method('GetPrimaryChannel', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetRifsMode() const [member function]
    cls.add_method('GetRifsMode', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint16_t ns3::HtOperation::GetRxHighestSupportedDataRate() const [member function]
    cls.add_method('GetRxHighestSupportedDataRate', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetSecondaryChannelOffset() const [member function]
    cls.add_method('GetSecondaryChannelOffset', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint16_t ns3::HtOperation::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetStaChannelWidth() const [member function]
    cls.add_method('GetStaChannelWidth', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetStbcBeacon() const [member function]
    cls.add_method('GetStbcBeacon', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetTxMaxNSpatialStreams() const [member function]
    cls.add_method('GetTxMaxNSpatialStreams', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetTxMcsSetDefined() const [member function]
    cls.add_method('GetTxMcsSetDefined', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetTxRxMcsSetUnequal() const [member function]
    cls.add_method('GetTxRxMcsSetUnequal', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): uint8_t ns3::HtOperation::GetTxUnequalModulation() const [member function]
    cls.add_method('GetTxUnequalModulation', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): bool ns3::HtOperation::IsSupportedMcs(uint8_t mcs) const [member function]
    cls.add_method('IsSupportedMcs', 
                   'bool', 
                   [param('uint8_t', 'mcs')], 
                   is_const=True)
    ## ht-operation.h (module 'wifi'): ns3::Buffer::Iterator ns3::HtOperation::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetBasicMcsSet(uint64_t ctrl1, uint64_t ctrl2) [member function]
    cls.add_method('SetBasicMcsSet', 
                   'void', 
                   [param('uint64_t', 'ctrl1'), param('uint64_t', 'ctrl2')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetDualBeacon(uint8_t dualBeacon) [member function]
    cls.add_method('SetDualBeacon', 
                   'void', 
                   [param('uint8_t', 'dualBeacon')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetDualCtsProtection(uint8_t dualCtsProtection) [member function]
    cls.add_method('SetDualCtsProtection', 
                   'void', 
                   [param('uint8_t', 'dualCtsProtection')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetHtProtection(uint8_t htProtection) [member function]
    cls.add_method('SetHtProtection', 
                   'void', 
                   [param('uint8_t', 'htProtection')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetHtSupported(uint8_t htSupported) [member function]
    cls.add_method('SetHtSupported', 
                   'void', 
                   [param('uint8_t', 'htSupported')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetInformationSubset1(uint8_t ctrl) [member function]
    cls.add_method('SetInformationSubset1', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetInformationSubset2(uint16_t ctrl) [member function]
    cls.add_method('SetInformationSubset2', 
                   'void', 
                   [param('uint16_t', 'ctrl')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetInformationSubset3(uint16_t ctrl) [member function]
    cls.add_method('SetInformationSubset3', 
                   'void', 
                   [param('uint16_t', 'ctrl')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetLSigTxopProtectionFullSupport(uint8_t lSigTxopProtectionFullSupport) [member function]
    cls.add_method('SetLSigTxopProtectionFullSupport', 
                   'void', 
                   [param('uint8_t', 'lSigTxopProtectionFullSupport')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetNonGfHtStasPresent(uint8_t nonGfHtStasPresent) [member function]
    cls.add_method('SetNonGfHtStasPresent', 
                   'void', 
                   [param('uint8_t', 'nonGfHtStasPresent')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetObssNonHtStasPresent(uint8_t obssNonHtStasPresent) [member function]
    cls.add_method('SetObssNonHtStasPresent', 
                   'void', 
                   [param('uint8_t', 'obssNonHtStasPresent')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetPcoActive(uint8_t pcoActive) [member function]
    cls.add_method('SetPcoActive', 
                   'void', 
                   [param('uint8_t', 'pcoActive')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetPhase(uint8_t pcoPhase) [member function]
    cls.add_method('SetPhase', 
                   'void', 
                   [param('uint8_t', 'pcoPhase')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetPrimaryChannel(uint8_t ctrl) [member function]
    cls.add_method('SetPrimaryChannel', 
                   'void', 
                   [param('uint8_t', 'ctrl')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetRifsMode(uint8_t rifsMode) [member function]
    cls.add_method('SetRifsMode', 
                   'void', 
                   [param('uint8_t', 'rifsMode')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetRxHighestSupportedDataRate(uint16_t maxSupportedRate) [member function]
    cls.add_method('SetRxHighestSupportedDataRate', 
                   'void', 
                   [param('uint16_t', 'maxSupportedRate')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetRxMcsBitmask(uint8_t index) [member function]
    cls.add_method('SetRxMcsBitmask', 
                   'void', 
                   [param('uint8_t', 'index')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetSecondaryChannelOffset(uint8_t secondaryChannelOffset) [member function]
    cls.add_method('SetSecondaryChannelOffset', 
                   'void', 
                   [param('uint8_t', 'secondaryChannelOffset')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetStaChannelWidth(uint8_t staChannelWidth) [member function]
    cls.add_method('SetStaChannelWidth', 
                   'void', 
                   [param('uint8_t', 'staChannelWidth')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetStbcBeacon(uint8_t stbcBeacon) [member function]
    cls.add_method('SetStbcBeacon', 
                   'void', 
                   [param('uint8_t', 'stbcBeacon')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetTxMaxNSpatialStreams(uint8_t maxTxSpatialStreams) [member function]
    cls.add_method('SetTxMaxNSpatialStreams', 
                   'void', 
                   [param('uint8_t', 'maxTxSpatialStreams')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetTxMcsSetDefined(uint8_t txMcsSetDefined) [member function]
    cls.add_method('SetTxMcsSetDefined', 
                   'void', 
                   [param('uint8_t', 'txMcsSetDefined')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetTxRxMcsSetUnequal(uint8_t txRxMcsSetUnequal) [member function]
    cls.add_method('SetTxRxMcsSetUnequal', 
                   'void', 
                   [param('uint8_t', 'txRxMcsSetUnequal')])
    ## ht-operation.h (module 'wifi'): void ns3::HtOperation::SetTxUnequalModulation(uint8_t txUnequalModulation) [member function]
    cls.add_method('SetTxUnequalModulation', 
                   'void', 
                   [param('uint8_t', 'txUnequalModulation')])
    return

def register_Ns3IntegerValue_methods(root_module, cls):
    ## integer.h (module 'core'): ns3::IntegerValue::IntegerValue() [constructor]
    cls.add_constructor([])
    ## integer.h (module 'core'): ns3::IntegerValue::IntegerValue(int64_t const & value) [constructor]
    cls.add_constructor([param('int64_t const &', 'value')])
    ## integer.h (module 'core'): ns3::IntegerValue::IntegerValue(ns3::IntegerValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::IntegerValue const &', 'arg0')])
    ## integer.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::IntegerValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## integer.h (module 'core'): bool ns3::IntegerValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## integer.h (module 'core'): int64_t ns3::IntegerValue::Get() const [member function]
    cls.add_method('Get', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## integer.h (module 'core'): std::string ns3::IntegerValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## integer.h (module 'core'): void ns3::IntegerValue::Set(int64_t const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('int64_t const &', 'value')])
    return

def register_Ns3Ipv4AddressChecker_methods(root_module, cls):
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressChecker::Ipv4AddressChecker() [constructor]
    cls.add_constructor([])
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressChecker::Ipv4AddressChecker(ns3::Ipv4AddressChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv4AddressChecker const &', 'arg0')])
    return

def register_Ns3Ipv4AddressValue_methods(root_module, cls):
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressValue::Ipv4AddressValue() [constructor]
    cls.add_constructor([])
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressValue::Ipv4AddressValue(ns3::Ipv4Address const & value) [constructor]
    cls.add_constructor([param('ns3::Ipv4Address const &', 'value')])
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressValue::Ipv4AddressValue(ns3::Ipv4AddressValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv4AddressValue const &', 'arg0')])
    ## ipv4-address.h (module 'network'): ns3::Ptr<ns3::AttributeValue> ns3::Ipv4AddressValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ipv4-address.h (module 'network'): bool ns3::Ipv4AddressValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address ns3::Ipv4AddressValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::Ipv4Address', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): std::string ns3::Ipv4AddressValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## ipv4-address.h (module 'network'): void ns3::Ipv4AddressValue::Set(ns3::Ipv4Address const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::Ipv4Address const &', 'value')])
    return

def register_Ns3Ipv4MaskChecker_methods(root_module, cls):
    ## ipv4-address.h (module 'network'): ns3::Ipv4MaskChecker::Ipv4MaskChecker() [constructor]
    cls.add_constructor([])
    ## ipv4-address.h (module 'network'): ns3::Ipv4MaskChecker::Ipv4MaskChecker(ns3::Ipv4MaskChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv4MaskChecker const &', 'arg0')])
    return

def register_Ns3Ipv4MaskValue_methods(root_module, cls):
    ## ipv4-address.h (module 'network'): ns3::Ipv4MaskValue::Ipv4MaskValue() [constructor]
    cls.add_constructor([])
    ## ipv4-address.h (module 'network'): ns3::Ipv4MaskValue::Ipv4MaskValue(ns3::Ipv4Mask const & value) [constructor]
    cls.add_constructor([param('ns3::Ipv4Mask const &', 'value')])
    ## ipv4-address.h (module 'network'): ns3::Ipv4MaskValue::Ipv4MaskValue(ns3::Ipv4MaskValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv4MaskValue const &', 'arg0')])
    ## ipv4-address.h (module 'network'): ns3::Ptr<ns3::AttributeValue> ns3::Ipv4MaskValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ipv4-address.h (module 'network'): bool ns3::Ipv4MaskValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## ipv4-address.h (module 'network'): ns3::Ipv4Mask ns3::Ipv4MaskValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::Ipv4Mask', 
                   [], 
                   is_const=True)
    ## ipv4-address.h (module 'network'): std::string ns3::Ipv4MaskValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## ipv4-address.h (module 'network'): void ns3::Ipv4MaskValue::Set(ns3::Ipv4Mask const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::Ipv4Mask const &', 'value')])
    return

def register_Ns3Ipv6AddressChecker_methods(root_module, cls):
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressChecker::Ipv6AddressChecker() [constructor]
    cls.add_constructor([])
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressChecker::Ipv6AddressChecker(ns3::Ipv6AddressChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv6AddressChecker const &', 'arg0')])
    return

def register_Ns3Ipv6AddressValue_methods(root_module, cls):
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressValue::Ipv6AddressValue() [constructor]
    cls.add_constructor([])
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressValue::Ipv6AddressValue(ns3::Ipv6Address const & value) [constructor]
    cls.add_constructor([param('ns3::Ipv6Address const &', 'value')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressValue::Ipv6AddressValue(ns3::Ipv6AddressValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv6AddressValue const &', 'arg0')])
    ## ipv6-address.h (module 'network'): ns3::Ptr<ns3::AttributeValue> ns3::Ipv6AddressValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6AddressValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address ns3::Ipv6AddressValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::Ipv6Address', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): std::string ns3::Ipv6AddressValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## ipv6-address.h (module 'network'): void ns3::Ipv6AddressValue::Set(ns3::Ipv6Address const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::Ipv6Address const &', 'value')])
    return

def register_Ns3Ipv6PrefixChecker_methods(root_module, cls):
    ## ipv6-address.h (module 'network'): ns3::Ipv6PrefixChecker::Ipv6PrefixChecker() [constructor]
    cls.add_constructor([])
    ## ipv6-address.h (module 'network'): ns3::Ipv6PrefixChecker::Ipv6PrefixChecker(ns3::Ipv6PrefixChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv6PrefixChecker const &', 'arg0')])
    return

def register_Ns3Ipv6PrefixValue_methods(root_module, cls):
    ## ipv6-address.h (module 'network'): ns3::Ipv6PrefixValue::Ipv6PrefixValue() [constructor]
    cls.add_constructor([])
    ## ipv6-address.h (module 'network'): ns3::Ipv6PrefixValue::Ipv6PrefixValue(ns3::Ipv6Prefix const & value) [constructor]
    cls.add_constructor([param('ns3::Ipv6Prefix const &', 'value')])
    ## ipv6-address.h (module 'network'): ns3::Ipv6PrefixValue::Ipv6PrefixValue(ns3::Ipv6PrefixValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv6PrefixValue const &', 'arg0')])
    ## ipv6-address.h (module 'network'): ns3::Ptr<ns3::AttributeValue> ns3::Ipv6PrefixValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ipv6-address.h (module 'network'): bool ns3::Ipv6PrefixValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix ns3::Ipv6PrefixValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::Ipv6Prefix', 
                   [], 
                   is_const=True)
    ## ipv6-address.h (module 'network'): std::string ns3::Ipv6PrefixValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## ipv6-address.h (module 'network'): void ns3::Ipv6PrefixValue::Set(ns3::Ipv6Prefix const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::Ipv6Prefix const &', 'value')])
    return

def register_Ns3LogNormalRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::LogNormalRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::LogNormalRandomVariable::LogNormalRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::LogNormalRandomVariable::GetMu() const [member function]
    cls.add_method('GetMu', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::LogNormalRandomVariable::GetSigma() const [member function]
    cls.add_method('GetSigma', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::LogNormalRandomVariable::GetValue(double mu, double sigma) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('double', 'mu'), param('double', 'sigma')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::LogNormalRandomVariable::GetInteger(uint32_t mu, uint32_t sigma) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'mu'), param('uint32_t', 'sigma')])
    ## random-variable-stream.h (module 'core'): double ns3::LogNormalRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::LogNormalRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3Mac48AddressChecker_methods(root_module, cls):
    ## mac48-address.h (module 'network'): ns3::Mac48AddressChecker::Mac48AddressChecker() [constructor]
    cls.add_constructor([])
    ## mac48-address.h (module 'network'): ns3::Mac48AddressChecker::Mac48AddressChecker(ns3::Mac48AddressChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::Mac48AddressChecker const &', 'arg0')])
    return

def register_Ns3Mac48AddressValue_methods(root_module, cls):
    ## mac48-address.h (module 'network'): ns3::Mac48AddressValue::Mac48AddressValue() [constructor]
    cls.add_constructor([])
    ## mac48-address.h (module 'network'): ns3::Mac48AddressValue::Mac48AddressValue(ns3::Mac48Address const & value) [constructor]
    cls.add_constructor([param('ns3::Mac48Address const &', 'value')])
    ## mac48-address.h (module 'network'): ns3::Mac48AddressValue::Mac48AddressValue(ns3::Mac48AddressValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::Mac48AddressValue const &', 'arg0')])
    ## mac48-address.h (module 'network'): ns3::Ptr<ns3::AttributeValue> ns3::Mac48AddressValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac48-address.h (module 'network'): bool ns3::Mac48AddressValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## mac48-address.h (module 'network'): ns3::Mac48Address ns3::Mac48AddressValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## mac48-address.h (module 'network'): std::string ns3::Mac48AddressValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## mac48-address.h (module 'network'): void ns3::Mac48AddressValue::Set(ns3::Mac48Address const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::Mac48Address const &', 'value')])
    return

def register_Ns3MeshInformationElementVector_methods(root_module, cls):
    ## mesh-information-element-vector.h (module 'mesh'): ns3::MeshInformationElementVector::MeshInformationElementVector(ns3::MeshInformationElementVector const & arg0) [constructor]
    cls.add_constructor([param('ns3::MeshInformationElementVector const &', 'arg0')])
    ## mesh-information-element-vector.h (module 'mesh'): ns3::MeshInformationElementVector::MeshInformationElementVector() [constructor]
    cls.add_constructor([])
    ## mesh-information-element-vector.h (module 'mesh'): uint32_t ns3::MeshInformationElementVector::DeserializeSingleIe(ns3::Buffer::Iterator start) [member function]
    cls.add_method('DeserializeSingleIe', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mesh-information-element-vector.h (module 'mesh'): ns3::TypeId ns3::MeshInformationElementVector::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-information-element-vector.h (module 'mesh'): static ns3::TypeId ns3::MeshInformationElementVector::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    return

def register_Ns3MeshL2RoutingProtocol_methods(root_module, cls):
    ## mesh-l2-routing-protocol.h (module 'mesh'): ns3::MeshL2RoutingProtocol::MeshL2RoutingProtocol() [constructor]
    cls.add_constructor([])
    ## mesh-l2-routing-protocol.h (module 'mesh'): ns3::MeshL2RoutingProtocol::MeshL2RoutingProtocol(ns3::MeshL2RoutingProtocol const & arg0) [constructor]
    cls.add_constructor([param('ns3::MeshL2RoutingProtocol const &', 'arg0')])
    ## mesh-l2-routing-protocol.h (module 'mesh'): ns3::Ptr<ns3::MeshPointDevice> ns3::MeshL2RoutingProtocol::GetMeshPoint() const [member function]
    cls.add_method('GetMeshPoint', 
                   'ns3::Ptr< ns3::MeshPointDevice >', 
                   [], 
                   is_const=True)
    ## mesh-l2-routing-protocol.h (module 'mesh'): static ns3::TypeId ns3::MeshL2RoutingProtocol::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mesh-l2-routing-protocol.h (module 'mesh'): bool ns3::MeshL2RoutingProtocol::RemoveRoutingStuff(uint32_t fromIface, ns3::Mac48Address const source, ns3::Mac48Address const destination, ns3::Ptr<ns3::Packet> packet, uint16_t & protocolType) [member function]
    cls.add_method('RemoveRoutingStuff', 
                   'bool', 
                   [param('uint32_t', 'fromIface'), param('ns3::Mac48Address const', 'source'), param('ns3::Mac48Address const', 'destination'), param('ns3::Ptr< ns3::Packet >', 'packet'), param('uint16_t &', 'protocolType')], 
                   is_pure_virtual=True, is_virtual=True)
    ## mesh-l2-routing-protocol.h (module 'mesh'): bool ns3::MeshL2RoutingProtocol::RequestRoute(uint32_t sourceIface, ns3::Mac48Address const source, ns3::Mac48Address const destination, ns3::Ptr<const ns3::Packet> packet, uint16_t protocolType, ns3::MeshL2RoutingProtocol::RouteReplyCallback routeReply) [member function]
    cls.add_method('RequestRoute', 
                   'bool', 
                   [param('uint32_t', 'sourceIface'), param('ns3::Mac48Address const', 'source'), param('ns3::Mac48Address const', 'destination'), param('ns3::Ptr< ns3::Packet const >', 'packet'), param('uint16_t', 'protocolType'), param('ns3::Callback< void, bool, ns3::Ptr< ns3::Packet >, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty >', 'routeReply')], 
                   is_pure_virtual=True, is_virtual=True)
    ## mesh-l2-routing-protocol.h (module 'mesh'): void ns3::MeshL2RoutingProtocol::SetMeshPoint(ns3::Ptr<ns3::MeshPointDevice> mp) [member function]
    cls.add_method('SetMeshPoint', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshPointDevice >', 'mp')])
    return

def register_Ns3MeshStack_methods(root_module, cls):
    ## mesh-stack-installer.h (module 'mesh'): ns3::MeshStack::MeshStack() [constructor]
    cls.add_constructor([])
    ## mesh-stack-installer.h (module 'mesh'): ns3::MeshStack::MeshStack(ns3::MeshStack const & arg0) [constructor]
    cls.add_constructor([param('ns3::MeshStack const &', 'arg0')])
    ## mesh-stack-installer.h (module 'mesh'): static ns3::TypeId ns3::MeshStack::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mesh-stack-installer.h (module 'mesh'): bool ns3::MeshStack::InstallStack(ns3::Ptr<ns3::MeshPointDevice> mp) [member function]
    cls.add_method('InstallStack', 
                   'bool', 
                   [param('ns3::Ptr< ns3::MeshPointDevice >', 'mp')], 
                   is_pure_virtual=True, is_virtual=True)
    ## mesh-stack-installer.h (module 'mesh'): void ns3::MeshStack::Report(ns3::Ptr<ns3::MeshPointDevice> const mp, std::ostream & os) [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshPointDevice > const', 'mp'), param('std::ostream &', 'os')], 
                   is_pure_virtual=True, is_virtual=True)
    ## mesh-stack-installer.h (module 'mesh'): void ns3::MeshStack::ResetStats(ns3::Ptr<ns3::MeshPointDevice> const mp) [member function]
    cls.add_method('ResetStats', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshPointDevice > const', 'mp')], 
                   is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3MeshWifiInterfaceMac_methods(root_module, cls):
    ## mesh-wifi-interface-mac.h (module 'mesh'): ns3::MeshWifiInterfaceMac::MeshWifiInterfaceMac() [constructor]
    cls.add_constructor([])
    ## mesh-wifi-interface-mac.h (module 'mesh'): int64_t ns3::MeshWifiInterfaceMac::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): bool ns3::MeshWifiInterfaceMac::CanForwardPacketsTo(ns3::Mac48Address to) const [member function]
    cls.add_method('CanForwardPacketsTo', 
                   'bool', 
                   [param('ns3::Mac48Address', 'to')], 
                   is_const=True, is_virtual=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): bool ns3::MeshWifiInterfaceMac::CheckSupportedRates(ns3::SupportedRates rates) const [member function]
    cls.add_method('CheckSupportedRates', 
                   'bool', 
                   [param('ns3::SupportedRates', 'rates')], 
                   is_const=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::ConfigureContentionWindow(uint32_t cwMin, uint32_t cwMax) [member function]
    cls.add_method('ConfigureContentionWindow', 
                   'void', 
                   [param('uint32_t', 'cwMin'), param('uint32_t', 'cwMax')], 
                   is_virtual=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::ConfigureStandard(ns3::WifiStandard standard) [member function]
    cls.add_method('ConfigureStandard', 
                   'void', 
                   [param('ns3::WifiStandard', 'standard')], 
                   is_virtual=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::Enqueue(ns3::Ptr<ns3::Packet> packet, ns3::Mac48Address to, ns3::Mac48Address from) [member function]
    cls.add_method('Enqueue', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Mac48Address', 'to'), param('ns3::Mac48Address', 'from')], 
                   is_virtual=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::Enqueue(ns3::Ptr<ns3::Packet> packet, ns3::Mac48Address to) [member function]
    cls.add_method('Enqueue', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Mac48Address', 'to')], 
                   is_virtual=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): ns3::Time ns3::MeshWifiInterfaceMac::GetBeaconInterval() const [member function]
    cls.add_method('GetBeaconInterval', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): uint16_t ns3::MeshWifiInterfaceMac::GetFrequencyChannel() const [member function]
    cls.add_method('GetFrequencyChannel', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): uint32_t ns3::MeshWifiInterfaceMac::GetLinkMetric(ns3::Mac48Address peerAddress) [member function]
    cls.add_method('GetLinkMetric', 
                   'uint32_t', 
                   [param('ns3::Mac48Address', 'peerAddress')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): ns3::Mac48Address ns3::MeshWifiInterfaceMac::GetMeshPointAddress() const [member function]
    cls.add_method('GetMeshPointAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): ns3::SupportedRates ns3::MeshWifiInterfaceMac::GetSupportedRates() const [member function]
    cls.add_method('GetSupportedRates', 
                   'ns3::SupportedRates', 
                   [], 
                   is_const=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): ns3::Time ns3::MeshWifiInterfaceMac::GetTbtt() const [member function]
    cls.add_method('GetTbtt', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): static ns3::TypeId ns3::MeshWifiInterfaceMac::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::InstallPlugin(ns3::Ptr<ns3::MeshWifiInterfaceMacPlugin> plugin) [member function]
    cls.add_method('InstallPlugin', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshWifiInterfaceMacPlugin >', 'plugin')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::Report(std::ostream & os) const [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::ResetStats() [member function]
    cls.add_method('ResetStats', 
                   'void', 
                   [])
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::SendManagementFrame(ns3::Ptr<ns3::Packet> frame, ns3::WifiMacHeader const & hdr) [member function]
    cls.add_method('SendManagementFrame', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'frame'), param('ns3::WifiMacHeader const &', 'hdr')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::SetBeaconGeneration(bool enable) [member function]
    cls.add_method('SetBeaconGeneration', 
                   'void', 
                   [param('bool', 'enable')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::SetBeaconInterval(ns3::Time interval) [member function]
    cls.add_method('SetBeaconInterval', 
                   'void', 
                   [param('ns3::Time', 'interval')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::SetLinkMetricCallback(ns3::Callback<unsigned int, ns3::Mac48Address, ns3::Ptr<ns3::MeshWifiInterfaceMac>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> cb) [member function]
    cls.add_method('SetLinkMetricCallback', 
                   'void', 
                   [param('ns3::Callback< unsigned int, ns3::Mac48Address, ns3::Ptr< ns3::MeshWifiInterfaceMac >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::SetLinkUpCallback(ns3::Callback<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> linkUp) [member function]
    cls.add_method('SetLinkUpCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'linkUp')], 
                   is_virtual=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::SetMeshPointAddress(ns3::Mac48Address addr) [member function]
    cls.add_method('SetMeshPointAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'addr')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::SetRandomStartDelay(ns3::Time interval) [member function]
    cls.add_method('SetRandomStartDelay', 
                   'void', 
                   [param('ns3::Time', 'interval')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::ShiftTbtt(ns3::Time shift) [member function]
    cls.add_method('ShiftTbtt', 
                   'void', 
                   [param('ns3::Time', 'shift')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): bool ns3::MeshWifiInterfaceMac::SupportsSendFrom() const [member function]
    cls.add_method('SupportsSendFrom', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::SwitchFrequencyChannel(uint16_t new_id) [member function]
    cls.add_method('SwitchFrequencyChannel', 
                   'void', 
                   [param('uint16_t', 'new_id')])
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::DoInitialize() [member function]
    cls.add_method('DoInitialize', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    ## mesh-wifi-interface-mac.h (module 'mesh'): void ns3::MeshWifiInterfaceMac::Receive(ns3::Ptr<ns3::WifiMacQueueItem> mpdu) [member function]
    cls.add_method('Receive', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem >', 'mpdu')], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3MeshWifiInterfaceMacPlugin_methods(root_module, cls):
    ## mesh-wifi-interface-mac-plugin.h (module 'mesh'): ns3::MeshWifiInterfaceMacPlugin::MeshWifiInterfaceMacPlugin() [constructor]
    cls.add_constructor([])
    ## mesh-wifi-interface-mac-plugin.h (module 'mesh'): ns3::MeshWifiInterfaceMacPlugin::MeshWifiInterfaceMacPlugin(ns3::MeshWifiInterfaceMacPlugin const & arg0) [constructor]
    cls.add_constructor([param('ns3::MeshWifiInterfaceMacPlugin const &', 'arg0')])
    ## mesh-wifi-interface-mac-plugin.h (module 'mesh'): int64_t ns3::MeshWifiInterfaceMacPlugin::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_pure_virtual=True, is_virtual=True)
    ## mesh-wifi-interface-mac-plugin.h (module 'mesh'): bool ns3::MeshWifiInterfaceMacPlugin::Receive(ns3::Ptr<ns3::Packet> packet, ns3::WifiMacHeader const & header) [member function]
    cls.add_method('Receive', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::WifiMacHeader const &', 'header')], 
                   is_pure_virtual=True, is_virtual=True)
    ## mesh-wifi-interface-mac-plugin.h (module 'mesh'): void ns3::MeshWifiInterfaceMacPlugin::SetParent(ns3::Ptr<ns3::MeshWifiInterfaceMac> parent) [member function]
    cls.add_method('SetParent', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshWifiInterfaceMac >', 'parent')], 
                   is_pure_virtual=True, is_virtual=True)
    ## mesh-wifi-interface-mac-plugin.h (module 'mesh'): void ns3::MeshWifiInterfaceMacPlugin::UpdateBeacon(ns3::MeshWifiBeacon & beacon) const [member function]
    cls.add_method('UpdateBeacon', 
                   'void', 
                   [param('ns3::MeshWifiBeacon &', 'beacon')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## mesh-wifi-interface-mac-plugin.h (module 'mesh'): bool ns3::MeshWifiInterfaceMacPlugin::UpdateOutcomingFrame(ns3::Ptr<ns3::Packet> packet, ns3::WifiMacHeader & header, ns3::Mac48Address from, ns3::Mac48Address to) [member function]
    cls.add_method('UpdateOutcomingFrame', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::WifiMacHeader &', 'header'), param('ns3::Mac48Address', 'from'), param('ns3::Mac48Address', 'to')], 
                   is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3MgtBeaconHeader_methods(root_module, cls):
    ## mgt-headers.h (module 'wifi'): ns3::MgtBeaconHeader::MgtBeaconHeader() [constructor]
    cls.add_constructor([])
    ## mgt-headers.h (module 'wifi'): ns3::MgtBeaconHeader::MgtBeaconHeader(ns3::MgtBeaconHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::MgtBeaconHeader const &', 'arg0')])
    ## mgt-headers.h (module 'wifi'): static ns3::TypeId ns3::MgtBeaconHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    return

def register_Ns3MuEdcaParameterSet_methods(root_module, cls):
    ## mu-edca-parameter-set.h (module 'wifi'): ns3::MuEdcaParameterSet::MuEdcaParameterSet(ns3::MuEdcaParameterSet const & arg0) [constructor]
    cls.add_constructor([param('ns3::MuEdcaParameterSet const &', 'arg0')])
    ## mu-edca-parameter-set.h (module 'wifi'): ns3::MuEdcaParameterSet::MuEdcaParameterSet() [constructor]
    cls.add_constructor([])
    ## mu-edca-parameter-set.h (module 'wifi'): uint8_t ns3::MuEdcaParameterSet::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## mu-edca-parameter-set.h (module 'wifi'): ns3::WifiInformationElementId ns3::MuEdcaParameterSet::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mu-edca-parameter-set.h (module 'wifi'): ns3::WifiInformationElementId ns3::MuEdcaParameterSet::ElementIdExt() const [member function]
    cls.add_method('ElementIdExt', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mu-edca-parameter-set.h (module 'wifi'): uint8_t ns3::MuEdcaParameterSet::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mu-edca-parameter-set.h (module 'wifi'): uint8_t ns3::MuEdcaParameterSet::GetMuAifsn(uint8_t aci) const [member function]
    cls.add_method('GetMuAifsn', 
                   'uint8_t', 
                   [param('uint8_t', 'aci')], 
                   is_const=True)
    ## mu-edca-parameter-set.h (module 'wifi'): uint16_t ns3::MuEdcaParameterSet::GetMuCwMax(uint8_t aci) const [member function]
    cls.add_method('GetMuCwMax', 
                   'uint16_t', 
                   [param('uint8_t', 'aci')], 
                   is_const=True)
    ## mu-edca-parameter-set.h (module 'wifi'): uint16_t ns3::MuEdcaParameterSet::GetMuCwMin(uint8_t aci) const [member function]
    cls.add_method('GetMuCwMin', 
                   'uint16_t', 
                   [param('uint8_t', 'aci')], 
                   is_const=True)
    ## mu-edca-parameter-set.h (module 'wifi'): ns3::Time ns3::MuEdcaParameterSet::GetMuEdcaTimer(uint8_t aci) const [member function]
    cls.add_method('GetMuEdcaTimer', 
                   'ns3::Time', 
                   [param('uint8_t', 'aci')], 
                   is_const=True)
    ## mu-edca-parameter-set.h (module 'wifi'): uint8_t ns3::MuEdcaParameterSet::GetQosInfo() const [member function]
    cls.add_method('GetQosInfo', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mu-edca-parameter-set.h (module 'wifi'): uint16_t ns3::MuEdcaParameterSet::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mu-edca-parameter-set.h (module 'wifi'): bool ns3::MuEdcaParameterSet::IsPresent() const [member function]
    cls.add_method('IsPresent', 
                   'bool', 
                   [], 
                   is_const=True)
    ## mu-edca-parameter-set.h (module 'wifi'): ns3::Buffer::Iterator ns3::MuEdcaParameterSet::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mu-edca-parameter-set.h (module 'wifi'): void ns3::MuEdcaParameterSet::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mu-edca-parameter-set.h (module 'wifi'): void ns3::MuEdcaParameterSet::SetMuAifsn(uint8_t aci, uint8_t aifsn) [member function]
    cls.add_method('SetMuAifsn', 
                   'void', 
                   [param('uint8_t', 'aci'), param('uint8_t', 'aifsn')])
    ## mu-edca-parameter-set.h (module 'wifi'): void ns3::MuEdcaParameterSet::SetMuCwMax(uint8_t aci, uint16_t cwMax) [member function]
    cls.add_method('SetMuCwMax', 
                   'void', 
                   [param('uint8_t', 'aci'), param('uint16_t', 'cwMax')])
    ## mu-edca-parameter-set.h (module 'wifi'): void ns3::MuEdcaParameterSet::SetMuCwMin(uint8_t aci, uint16_t cwMin) [member function]
    cls.add_method('SetMuCwMin', 
                   'void', 
                   [param('uint8_t', 'aci'), param('uint16_t', 'cwMin')])
    ## mu-edca-parameter-set.h (module 'wifi'): void ns3::MuEdcaParameterSet::SetMuEdcaTimer(uint8_t aci, ns3::Time timer) [member function]
    cls.add_method('SetMuEdcaTimer', 
                   'void', 
                   [param('uint8_t', 'aci'), param('ns3::Time', 'timer')])
    ## mu-edca-parameter-set.h (module 'wifi'): void ns3::MuEdcaParameterSet::SetQosInfo(uint8_t qosInfo) [member function]
    cls.add_method('SetQosInfo', 
                   'void', 
                   [param('uint8_t', 'qosInfo')])
    return

def register_Ns3NetDevice_methods(root_module, cls):
    ## net-device.h (module 'network'): ns3::NetDevice::NetDevice() [constructor]
    cls.add_constructor([])
    ## net-device.h (module 'network'): ns3::NetDevice::NetDevice(ns3::NetDevice const & arg0) [constructor]
    cls.add_constructor([param('ns3::NetDevice const &', 'arg0')])
    ## net-device.h (module 'network'): void ns3::NetDevice::AddLinkChangeCallback(ns3::Callback<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> callback) [member function]
    cls.add_method('AddLinkChangeCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'callback')], 
                   is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): ns3::Address ns3::NetDevice::GetAddress() const [member function]
    cls.add_method('GetAddress', 
                   'ns3::Address', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): ns3::Address ns3::NetDevice::GetBroadcast() const [member function]
    cls.add_method('GetBroadcast', 
                   'ns3::Address', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): ns3::Ptr<ns3::Channel> ns3::NetDevice::GetChannel() const [member function]
    cls.add_method('GetChannel', 
                   'ns3::Ptr< ns3::Channel >', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): uint32_t ns3::NetDevice::GetIfIndex() const [member function]
    cls.add_method('GetIfIndex', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): uint16_t ns3::NetDevice::GetMtu() const [member function]
    cls.add_method('GetMtu', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): ns3::Address ns3::NetDevice::GetMulticast(ns3::Ipv4Address multicastGroup) const [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Address', 
                   [param('ns3::Ipv4Address', 'multicastGroup')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): ns3::Address ns3::NetDevice::GetMulticast(ns3::Ipv6Address addr) const [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Address', 
                   [param('ns3::Ipv6Address', 'addr')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): ns3::Ptr<ns3::Node> ns3::NetDevice::GetNode() const [member function]
    cls.add_method('GetNode', 
                   'ns3::Ptr< ns3::Node >', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): static ns3::TypeId ns3::NetDevice::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## net-device.h (module 'network'): bool ns3::NetDevice::IsBridge() const [member function]
    cls.add_method('IsBridge', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): bool ns3::NetDevice::IsBroadcast() const [member function]
    cls.add_method('IsBroadcast', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): bool ns3::NetDevice::IsLinkUp() const [member function]
    cls.add_method('IsLinkUp', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): bool ns3::NetDevice::IsMulticast() const [member function]
    cls.add_method('IsMulticast', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): bool ns3::NetDevice::IsPointToPoint() const [member function]
    cls.add_method('IsPointToPoint', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): bool ns3::NetDevice::NeedsArp() const [member function]
    cls.add_method('NeedsArp', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): bool ns3::NetDevice::Send(ns3::Ptr<ns3::Packet> packet, ns3::Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('Send', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): bool ns3::NetDevice::SendFrom(ns3::Ptr<ns3::Packet> packet, ns3::Address const & source, ns3::Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('SendFrom', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Address const &', 'source'), param('ns3::Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): void ns3::NetDevice::SetAddress(ns3::Address address) [member function]
    cls.add_method('SetAddress', 
                   'void', 
                   [param('ns3::Address', 'address')], 
                   is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): void ns3::NetDevice::SetIfIndex(uint32_t const index) [member function]
    cls.add_method('SetIfIndex', 
                   'void', 
                   [param('uint32_t const', 'index')], 
                   is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): bool ns3::NetDevice::SetMtu(uint16_t const mtu) [member function]
    cls.add_method('SetMtu', 
                   'bool', 
                   [param('uint16_t const', 'mtu')], 
                   is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): void ns3::NetDevice::SetNode(ns3::Ptr<ns3::Node> node) [member function]
    cls.add_method('SetNode', 
                   'void', 
                   [param('ns3::Ptr< ns3::Node >', 'node')], 
                   is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): void ns3::NetDevice::SetPromiscReceiveCallback(ns3::NetDevice::PromiscReceiveCallback cb) [member function]
    cls.add_method('SetPromiscReceiveCallback', 
                   'void', 
                   [param('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >', 'cb')], 
                   is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): void ns3::NetDevice::SetReceiveCallback(ns3::NetDevice::ReceiveCallback cb) [member function]
    cls.add_method('SetReceiveCallback', 
                   'void', 
                   [param('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')], 
                   is_pure_virtual=True, is_virtual=True)
    ## net-device.h (module 'network'): bool ns3::NetDevice::SupportsSendFrom() const [member function]
    cls.add_method('SupportsSendFrom', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3NixVector_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## nix-vector.h (module 'network'): ns3::NixVector::NixVector() [constructor]
    cls.add_constructor([])
    ## nix-vector.h (module 'network'): ns3::NixVector::NixVector(ns3::NixVector const & o) [constructor]
    cls.add_constructor([param('ns3::NixVector const &', 'o')])
    ## nix-vector.h (module 'network'): void ns3::NixVector::AddNeighborIndex(uint32_t newBits, uint32_t numberOfBits) [member function]
    cls.add_method('AddNeighborIndex', 
                   'void', 
                   [param('uint32_t', 'newBits'), param('uint32_t', 'numberOfBits')])
    ## nix-vector.h (module 'network'): uint32_t ns3::NixVector::BitCount(uint32_t numberOfNeighbors) const [member function]
    cls.add_method('BitCount', 
                   'uint32_t', 
                   [param('uint32_t', 'numberOfNeighbors')], 
                   is_const=True)
    ## nix-vector.h (module 'network'): ns3::Ptr<ns3::NixVector> ns3::NixVector::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::NixVector >', 
                   [], 
                   is_const=True)
    ## nix-vector.h (module 'network'): uint32_t ns3::NixVector::Deserialize(uint32_t const * buffer, uint32_t size) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('uint32_t const *', 'buffer'), param('uint32_t', 'size')])
    ## nix-vector.h (module 'network'): uint32_t ns3::NixVector::ExtractNeighborIndex(uint32_t numberOfBits) [member function]
    cls.add_method('ExtractNeighborIndex', 
                   'uint32_t', 
                   [param('uint32_t', 'numberOfBits')])
    ## nix-vector.h (module 'network'): uint32_t ns3::NixVector::GetEpoch() const [member function]
    cls.add_method('GetEpoch', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## nix-vector.h (module 'network'): uint32_t ns3::NixVector::GetRemainingBits() const [member function]
    cls.add_method('GetRemainingBits', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## nix-vector.h (module 'network'): uint32_t ns3::NixVector::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## nix-vector.h (module 'network'): uint32_t ns3::NixVector::Serialize(uint32_t * buffer, uint32_t maxSize) const [member function]
    cls.add_method('Serialize', 
                   'uint32_t', 
                   [param('uint32_t *', 'buffer'), param('uint32_t', 'maxSize')], 
                   is_const=True)
    ## nix-vector.h (module 'network'): void ns3::NixVector::SetEpoch(uint32_t epoch) [member function]
    cls.add_method('SetEpoch', 
                   'void', 
                   [param('uint32_t', 'epoch')])
    return

def register_Ns3Node_methods(root_module, cls):
    ## node.h (module 'network'): ns3::Node::Node(ns3::Node const & arg0) [constructor]
    cls.add_constructor([param('ns3::Node const &', 'arg0')])
    ## node.h (module 'network'): ns3::Node::Node() [constructor]
    cls.add_constructor([])
    ## node.h (module 'network'): ns3::Node::Node(uint32_t systemId) [constructor]
    cls.add_constructor([param('uint32_t', 'systemId')])
    ## node.h (module 'network'): uint32_t ns3::Node::AddApplication(ns3::Ptr<ns3::Application> application) [member function]
    cls.add_method('AddApplication', 
                   'uint32_t', 
                   [param('ns3::Ptr< ns3::Application >', 'application')])
    ## node.h (module 'network'): uint32_t ns3::Node::AddDevice(ns3::Ptr<ns3::NetDevice> device) [member function]
    cls.add_method('AddDevice', 
                   'uint32_t', 
                   [param('ns3::Ptr< ns3::NetDevice >', 'device')])
    ## node.h (module 'network'): static bool ns3::Node::ChecksumEnabled() [member function]
    cls.add_method('ChecksumEnabled', 
                   'bool', 
                   [], 
                   is_static=True)
    ## node.h (module 'network'): ns3::Ptr<ns3::Application> ns3::Node::GetApplication(uint32_t index) const [member function]
    cls.add_method('GetApplication', 
                   'ns3::Ptr< ns3::Application >', 
                   [param('uint32_t', 'index')], 
                   is_const=True)
    ## node.h (module 'network'): ns3::Ptr<ns3::NetDevice> ns3::Node::GetDevice(uint32_t index) const [member function]
    cls.add_method('GetDevice', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [param('uint32_t', 'index')], 
                   is_const=True)
    ## node.h (module 'network'): uint32_t ns3::Node::GetId() const [member function]
    cls.add_method('GetId', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## node.h (module 'network'): ns3::Time ns3::Node::GetLocalTime() const [member function]
    cls.add_method('GetLocalTime', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## node.h (module 'network'): uint32_t ns3::Node::GetNApplications() const [member function]
    cls.add_method('GetNApplications', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## node.h (module 'network'): uint32_t ns3::Node::GetNDevices() const [member function]
    cls.add_method('GetNDevices', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## node.h (module 'network'): uint32_t ns3::Node::GetSystemId() const [member function]
    cls.add_method('GetSystemId', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## node.h (module 'network'): static ns3::TypeId ns3::Node::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## node.h (module 'network'): void ns3::Node::RegisterDeviceAdditionListener(ns3::Node::DeviceAdditionListener listener) [member function]
    cls.add_method('RegisterDeviceAdditionListener', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::NetDevice >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'listener')])
    ## node.h (module 'network'): void ns3::Node::RegisterProtocolHandler(ns3::Node::ProtocolHandler handler, uint16_t protocolType, ns3::Ptr<ns3::NetDevice> device, bool promiscuous=false) [member function]
    cls.add_method('RegisterProtocolHandler', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >', 'handler'), param('uint16_t', 'protocolType'), param('ns3::Ptr< ns3::NetDevice >', 'device'), param('bool', 'promiscuous', default_value='false')])
    ## node.h (module 'network'): void ns3::Node::UnregisterDeviceAdditionListener(ns3::Node::DeviceAdditionListener listener) [member function]
    cls.add_method('UnregisterDeviceAdditionListener', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::NetDevice >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'listener')])
    ## node.h (module 'network'): void ns3::Node::UnregisterProtocolHandler(ns3::Node::ProtocolHandler handler) [member function]
    cls.add_method('UnregisterProtocolHandler', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >', 'handler')])
    ## node.h (module 'network'): void ns3::Node::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## node.h (module 'network'): void ns3::Node::DoInitialize() [member function]
    cls.add_method('DoInitialize', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3NormalRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): ns3::NormalRandomVariable::INFINITE_VALUE [variable]
    cls.add_static_attribute('INFINITE_VALUE', 'double const', is_const=True)
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::NormalRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::NormalRandomVariable::NormalRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::NormalRandomVariable::GetMean() const [member function]
    cls.add_method('GetMean', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::NormalRandomVariable::GetVariance() const [member function]
    cls.add_method('GetVariance', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::NormalRandomVariable::GetBound() const [member function]
    cls.add_method('GetBound', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::NormalRandomVariable::GetValue(double mean, double variance, double bound=ns3::NormalRandomVariable::INFINITE_VALUE) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('double', 'mean'), param('double', 'variance'), param('double', 'bound', default_value='ns3::NormalRandomVariable::INFINITE_VALUE')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::NormalRandomVariable::GetInteger(uint32_t mean, uint32_t variance, uint32_t bound) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'mean'), param('uint32_t', 'variance'), param('uint32_t', 'bound')])
    ## random-variable-stream.h (module 'core'): double ns3::NormalRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::NormalRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3ObjectFactoryChecker_methods(root_module, cls):
    ## object-factory.h (module 'core'): ns3::ObjectFactoryChecker::ObjectFactoryChecker() [constructor]
    cls.add_constructor([])
    ## object-factory.h (module 'core'): ns3::ObjectFactoryChecker::ObjectFactoryChecker(ns3::ObjectFactoryChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::ObjectFactoryChecker const &', 'arg0')])
    return

def register_Ns3ObjectFactoryValue_methods(root_module, cls):
    ## object-factory.h (module 'core'): ns3::ObjectFactoryValue::ObjectFactoryValue() [constructor]
    cls.add_constructor([])
    ## object-factory.h (module 'core'): ns3::ObjectFactoryValue::ObjectFactoryValue(ns3::ObjectFactory const & value) [constructor]
    cls.add_constructor([param('ns3::ObjectFactory const &', 'value')])
    ## object-factory.h (module 'core'): ns3::ObjectFactoryValue::ObjectFactoryValue(ns3::ObjectFactoryValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::ObjectFactoryValue const &', 'arg0')])
    ## object-factory.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::ObjectFactoryValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## object-factory.h (module 'core'): bool ns3::ObjectFactoryValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## object-factory.h (module 'core'): ns3::ObjectFactory ns3::ObjectFactoryValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::ObjectFactory', 
                   [], 
                   is_const=True)
    ## object-factory.h (module 'core'): std::string ns3::ObjectFactoryValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## object-factory.h (module 'core'): void ns3::ObjectFactoryValue::Set(ns3::ObjectFactory const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::ObjectFactory const &', 'value')])
    return

def register_Ns3Packet_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## packet.h (module 'network'): ns3::Packet::Packet() [constructor]
    cls.add_constructor([])
    ## packet.h (module 'network'): ns3::Packet::Packet(ns3::Packet const & o) [constructor]
    cls.add_constructor([param('ns3::Packet const &', 'o')])
    ## packet.h (module 'network'): ns3::Packet::Packet(uint32_t size) [constructor]
    cls.add_constructor([param('uint32_t', 'size')])
    ## packet.h (module 'network'): ns3::Packet::Packet(uint8_t const * buffer, uint32_t size, bool magic) [constructor]
    cls.add_constructor([param('uint8_t const *', 'buffer'), param('uint32_t', 'size'), param('bool', 'magic')])
    ## packet.h (module 'network'): ns3::Packet::Packet(uint8_t const * buffer, uint32_t size) [constructor]
    cls.add_constructor([param('uint8_t const *', 'buffer'), param('uint32_t', 'size')])
    ## packet.h (module 'network'): void ns3::Packet::AddAtEnd(ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('AddAtEnd', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet')])
    ## packet.h (module 'network'): void ns3::Packet::AddByteTag(ns3::Tag const & tag) const [member function]
    cls.add_method('AddByteTag', 
                   'void', 
                   [param('ns3::Tag const &', 'tag')], 
                   is_const=True)
    ## packet.h (module 'network'): void ns3::Packet::AddByteTag(ns3::Tag const & tag, uint32_t start, uint32_t end) const [member function]
    cls.add_method('AddByteTag', 
                   'void', 
                   [param('ns3::Tag const &', 'tag'), param('uint32_t', 'start'), param('uint32_t', 'end')], 
                   is_const=True)
    ## packet.h (module 'network'): void ns3::Packet::AddHeader(ns3::Header const & header) [member function]
    cls.add_method('AddHeader', 
                   'void', 
                   [param('ns3::Header const &', 'header')])
    ## packet.h (module 'network'): void ns3::Packet::AddPacketTag(ns3::Tag const & tag) const [member function]
    cls.add_method('AddPacketTag', 
                   'void', 
                   [param('ns3::Tag const &', 'tag')], 
                   is_const=True)
    ## packet.h (module 'network'): void ns3::Packet::AddPaddingAtEnd(uint32_t size) [member function]
    cls.add_method('AddPaddingAtEnd', 
                   'void', 
                   [param('uint32_t', 'size')])
    ## packet.h (module 'network'): void ns3::Packet::AddTrailer(ns3::Trailer const & trailer) [member function]
    cls.add_method('AddTrailer', 
                   'void', 
                   [param('ns3::Trailer const &', 'trailer')])
    ## packet.h (module 'network'): ns3::PacketMetadata::ItemIterator ns3::Packet::BeginItem() const [member function]
    cls.add_method('BeginItem', 
                   'ns3::PacketMetadata::ItemIterator', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::Packet::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::Packet >', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): uint32_t ns3::Packet::CopyData(uint8_t * buffer, uint32_t size) const [member function]
    cls.add_method('CopyData', 
                   'uint32_t', 
                   [param('uint8_t *', 'buffer'), param('uint32_t', 'size')], 
                   is_const=True)
    ## packet.h (module 'network'): void ns3::Packet::CopyData(std::ostream * os, uint32_t size) const [member function]
    cls.add_method('CopyData', 
                   'void', 
                   [param('std::ostream *', 'os'), param('uint32_t', 'size')], 
                   is_const=True)
    ## packet.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::Packet::CreateFragment(uint32_t start, uint32_t length) const [member function]
    cls.add_method('CreateFragment', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('uint32_t', 'start'), param('uint32_t', 'length')], 
                   is_const=True)
    ## packet.h (module 'network'): static void ns3::Packet::EnableChecking() [member function]
    cls.add_method('EnableChecking', 
                   'void', 
                   [], 
                   is_static=True)
    ## packet.h (module 'network'): static void ns3::Packet::EnablePrinting() [member function]
    cls.add_method('EnablePrinting', 
                   'void', 
                   [], 
                   is_static=True)
    ## packet.h (module 'network'): bool ns3::Packet::FindFirstMatchingByteTag(ns3::Tag & tag) const [member function]
    cls.add_method('FindFirstMatchingByteTag', 
                   'bool', 
                   [param('ns3::Tag &', 'tag')], 
                   is_const=True)
    ## packet.h (module 'network'): ns3::ByteTagIterator ns3::Packet::GetByteTagIterator() const [member function]
    cls.add_method('GetByteTagIterator', 
                   'ns3::ByteTagIterator', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): ns3::Ptr<ns3::NixVector> ns3::Packet::GetNixVector() const [member function]
    cls.add_method('GetNixVector', 
                   'ns3::Ptr< ns3::NixVector >', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): ns3::PacketTagIterator ns3::Packet::GetPacketTagIterator() const [member function]
    cls.add_method('GetPacketTagIterator', 
                   'ns3::PacketTagIterator', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): uint32_t ns3::Packet::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): uint32_t ns3::Packet::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): uint64_t ns3::Packet::GetUid() const [member function]
    cls.add_method('GetUid', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## packet.h (module 'network'): uint32_t ns3::Packet::PeekHeader(ns3::Header & header) const [member function]
    cls.add_method('PeekHeader', 
                   'uint32_t', 
                   [param('ns3::Header &', 'header')], 
                   is_const=True)
    ## packet.h (module 'network'): uint32_t ns3::Packet::PeekHeader(ns3::Header & header, uint32_t size) const [member function]
    cls.add_method('PeekHeader', 
                   'uint32_t', 
                   [param('ns3::Header &', 'header'), param('uint32_t', 'size')], 
                   is_const=True)
    ## packet.h (module 'network'): bool ns3::Packet::PeekPacketTag(ns3::Tag & tag) const [member function]
    cls.add_method('PeekPacketTag', 
                   'bool', 
                   [param('ns3::Tag &', 'tag')], 
                   is_const=True)
    ## packet.h (module 'network'): uint32_t ns3::Packet::PeekTrailer(ns3::Trailer & trailer) [member function]
    cls.add_method('PeekTrailer', 
                   'uint32_t', 
                   [param('ns3::Trailer &', 'trailer')])
    ## packet.h (module 'network'): void ns3::Packet::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## packet.h (module 'network'): void ns3::Packet::PrintByteTags(std::ostream & os) const [member function]
    cls.add_method('PrintByteTags', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## packet.h (module 'network'): void ns3::Packet::PrintPacketTags(std::ostream & os) const [member function]
    cls.add_method('PrintPacketTags', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## packet.h (module 'network'): void ns3::Packet::RemoveAllByteTags() [member function]
    cls.add_method('RemoveAllByteTags', 
                   'void', 
                   [])
    ## packet.h (module 'network'): void ns3::Packet::RemoveAllPacketTags() [member function]
    cls.add_method('RemoveAllPacketTags', 
                   'void', 
                   [])
    ## packet.h (module 'network'): void ns3::Packet::RemoveAtEnd(uint32_t size) [member function]
    cls.add_method('RemoveAtEnd', 
                   'void', 
                   [param('uint32_t', 'size')])
    ## packet.h (module 'network'): void ns3::Packet::RemoveAtStart(uint32_t size) [member function]
    cls.add_method('RemoveAtStart', 
                   'void', 
                   [param('uint32_t', 'size')])
    ## packet.h (module 'network'): uint32_t ns3::Packet::RemoveHeader(ns3::Header & header) [member function]
    cls.add_method('RemoveHeader', 
                   'uint32_t', 
                   [param('ns3::Header &', 'header')])
    ## packet.h (module 'network'): uint32_t ns3::Packet::RemoveHeader(ns3::Header & header, uint32_t size) [member function]
    cls.add_method('RemoveHeader', 
                   'uint32_t', 
                   [param('ns3::Header &', 'header'), param('uint32_t', 'size')])
    ## packet.h (module 'network'): bool ns3::Packet::RemovePacketTag(ns3::Tag & tag) [member function]
    cls.add_method('RemovePacketTag', 
                   'bool', 
                   [param('ns3::Tag &', 'tag')])
    ## packet.h (module 'network'): uint32_t ns3::Packet::RemoveTrailer(ns3::Trailer & trailer) [member function]
    cls.add_method('RemoveTrailer', 
                   'uint32_t', 
                   [param('ns3::Trailer &', 'trailer')])
    ## packet.h (module 'network'): bool ns3::Packet::ReplacePacketTag(ns3::Tag & tag) [member function]
    cls.add_method('ReplacePacketTag', 
                   'bool', 
                   [param('ns3::Tag &', 'tag')])
    ## packet.h (module 'network'): uint32_t ns3::Packet::Serialize(uint8_t * buffer, uint32_t maxSize) const [member function]
    cls.add_method('Serialize', 
                   'uint32_t', 
                   [param('uint8_t *', 'buffer'), param('uint32_t', 'maxSize')], 
                   is_const=True)
    ## packet.h (module 'network'): void ns3::Packet::SetNixVector(ns3::Ptr<ns3::NixVector> nixVector) const [member function]
    cls.add_method('SetNixVector', 
                   'void', 
                   [param('ns3::Ptr< ns3::NixVector >', 'nixVector')], 
                   is_const=True)
    ## packet.h (module 'network'): std::string ns3::Packet::ToString() const [member function]
    cls.add_method('ToString', 
                   'std::string', 
                   [], 
                   is_const=True)
    return

def register_Ns3ParetoRandomVariable_methods(root_module, cls):
    ## random-variable-stream.h (module 'core'): static ns3::TypeId ns3::ParetoRandomVariable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## random-variable-stream.h (module 'core'): ns3::ParetoRandomVariable::ParetoRandomVariable() [constructor]
    cls.add_constructor([])
    ## random-variable-stream.h (module 'core'): double ns3::ParetoRandomVariable::GetScale() const [member function]
    cls.add_method('GetScale', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ParetoRandomVariable::GetShape() const [member function]
    cls.add_method('GetShape', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ParetoRandomVariable::GetBound() const [member function]
    cls.add_method('GetBound', 
                   'double', 
                   [], 
                   is_const=True)
    ## random-variable-stream.h (module 'core'): double ns3::ParetoRandomVariable::GetValue(double scale, double shape, double bound) [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [param('double', 'scale'), param('double', 'shape'), param('double', 'bound')])
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ParetoRandomVariable::GetInteger(uint32_t scale, uint32_t shape, uint32_t bound) [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [param('uint32_t', 'scale'), param('uint32_t', 'shape'), param('uint32_t', 'bound')])
    ## random-variable-stream.h (module 'core'): double ns3::ParetoRandomVariable::GetValue() [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_virtual=True)
    ## random-variable-stream.h (module 'core'): uint32_t ns3::ParetoRandomVariable::GetInteger() [member function]
    cls.add_method('GetInteger', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3Ssid_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## ssid.h (module 'wifi'): ns3::Ssid::Ssid(ns3::Ssid const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ssid const &', 'arg0')])
    ## ssid.h (module 'wifi'): ns3::Ssid::Ssid() [constructor]
    cls.add_constructor([])
    ## ssid.h (module 'wifi'): ns3::Ssid::Ssid(std::string s) [constructor]
    cls.add_constructor([param('std::string', 's')])
    ## ssid.h (module 'wifi'): uint8_t ns3::Ssid::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ssid.h (module 'wifi'): ns3::WifiInformationElementId ns3::Ssid::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ssid.h (module 'wifi'): uint8_t ns3::Ssid::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ssid.h (module 'wifi'): bool ns3::Ssid::IsBroadcast() const [member function]
    cls.add_method('IsBroadcast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ssid.h (module 'wifi'): bool ns3::Ssid::IsEqual(ns3::Ssid const & o) const [member function]
    cls.add_method('IsEqual', 
                   'bool', 
                   [param('ns3::Ssid const &', 'o')], 
                   is_const=True)
    ## ssid.h (module 'wifi'): char * ns3::Ssid::PeekString() const [member function]
    cls.add_method('PeekString', 
                   'char *', 
                   [], 
                   is_const=True)
    ## ssid.h (module 'wifi'): void ns3::Ssid::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3SsidChecker_methods(root_module, cls):
    ## ssid.h (module 'wifi'): ns3::SsidChecker::SsidChecker() [constructor]
    cls.add_constructor([])
    ## ssid.h (module 'wifi'): ns3::SsidChecker::SsidChecker(ns3::SsidChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::SsidChecker const &', 'arg0')])
    return

def register_Ns3SsidValue_methods(root_module, cls):
    ## ssid.h (module 'wifi'): ns3::SsidValue::SsidValue() [constructor]
    cls.add_constructor([])
    ## ssid.h (module 'wifi'): ns3::SsidValue::SsidValue(ns3::Ssid const & value) [constructor]
    cls.add_constructor([param('ns3::Ssid const &', 'value')])
    ## ssid.h (module 'wifi'): ns3::SsidValue::SsidValue(ns3::SsidValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::SsidValue const &', 'arg0')])
    ## ssid.h (module 'wifi'): ns3::Ptr<ns3::AttributeValue> ns3::SsidValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ssid.h (module 'wifi'): bool ns3::SsidValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## ssid.h (module 'wifi'): ns3::Ssid ns3::SsidValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::Ssid', 
                   [], 
                   is_const=True)
    ## ssid.h (module 'wifi'): std::string ns3::SsidValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## ssid.h (module 'wifi'): void ns3::SsidValue::Set(ns3::Ssid const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::Ssid const &', 'value')])
    return

def register_Ns3SupportedRates_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## supported-rates.h (module 'wifi'): ns3::SupportedRates::SupportedRates() [constructor]
    cls.add_constructor([])
    ## supported-rates.h (module 'wifi'): ns3::SupportedRates::SupportedRates(ns3::SupportedRates const & rates) [constructor]
    cls.add_constructor([param('ns3::SupportedRates const &', 'rates')])
    ## supported-rates.h (module 'wifi'): void ns3::SupportedRates::AddBssMembershipSelectorRate(uint64_t bs) [member function]
    cls.add_method('AddBssMembershipSelectorRate', 
                   'void', 
                   [param('uint64_t', 'bs')])
    ## supported-rates.h (module 'wifi'): void ns3::SupportedRates::AddSupportedRate(uint64_t bs) [member function]
    cls.add_method('AddSupportedRate', 
                   'void', 
                   [param('uint64_t', 'bs')])
    ## supported-rates.h (module 'wifi'): uint8_t ns3::SupportedRates::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## supported-rates.h (module 'wifi'): ns3::WifiInformationElementId ns3::SupportedRates::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## supported-rates.h (module 'wifi'): uint8_t ns3::SupportedRates::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## supported-rates.h (module 'wifi'): uint8_t ns3::SupportedRates::GetNRates() const [member function]
    cls.add_method('GetNRates', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## supported-rates.h (module 'wifi'): uint32_t ns3::SupportedRates::GetRate(uint8_t i) const [member function]
    cls.add_method('GetRate', 
                   'uint32_t', 
                   [param('uint8_t', 'i')], 
                   is_const=True)
    ## supported-rates.h (module 'wifi'): bool ns3::SupportedRates::IsBasicRate(uint64_t bs) const [member function]
    cls.add_method('IsBasicRate', 
                   'bool', 
                   [param('uint64_t', 'bs')], 
                   is_const=True)
    ## supported-rates.h (module 'wifi'): bool ns3::SupportedRates::IsBssMembershipSelectorRate(uint64_t bs) const [member function]
    cls.add_method('IsBssMembershipSelectorRate', 
                   'bool', 
                   [param('uint64_t', 'bs')], 
                   is_const=True)
    ## supported-rates.h (module 'wifi'): bool ns3::SupportedRates::IsSupportedRate(uint64_t bs) const [member function]
    cls.add_method('IsSupportedRate', 
                   'bool', 
                   [param('uint64_t', 'bs')], 
                   is_const=True)
    ## supported-rates.h (module 'wifi'): void ns3::SupportedRates::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## supported-rates.h (module 'wifi'): void ns3::SupportedRates::SetBasicRate(uint64_t bs) [member function]
    cls.add_method('SetBasicRate', 
                   'void', 
                   [param('uint64_t', 'bs')])
    ## supported-rates.h (module 'wifi'): ns3::SupportedRates::MAX_SUPPORTED_RATES [variable]
    cls.add_static_attribute('MAX_SUPPORTED_RATES', 'uint8_t const', is_const=True)
    ## supported-rates.h (module 'wifi'): ns3::SupportedRates::extended [variable]
    cls.add_instance_attribute('extended', 'ns3::ExtendedSupportedRatesIE', is_const=False)
    return

def register_Ns3TimeValue_methods(root_module, cls):
    ## nstime.h (module 'core'): ns3::TimeValue::TimeValue() [constructor]
    cls.add_constructor([])
    ## nstime.h (module 'core'): ns3::TimeValue::TimeValue(ns3::Time const & value) [constructor]
    cls.add_constructor([param('ns3::Time const &', 'value')])
    ## nstime.h (module 'core'): ns3::TimeValue::TimeValue(ns3::TimeValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::TimeValue const &', 'arg0')])
    ## nstime.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::TimeValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## nstime.h (module 'core'): bool ns3::TimeValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## nstime.h (module 'core'): ns3::Time ns3::TimeValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## nstime.h (module 'core'): std::string ns3::TimeValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## nstime.h (module 'core'): void ns3::TimeValue::Set(ns3::Time const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::Time const &', 'value')])
    return

def register_Ns3TypeIdChecker_methods(root_module, cls):
    ## type-id.h (module 'core'): ns3::TypeIdChecker::TypeIdChecker() [constructor]
    cls.add_constructor([])
    ## type-id.h (module 'core'): ns3::TypeIdChecker::TypeIdChecker(ns3::TypeIdChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::TypeIdChecker const &', 'arg0')])
    return

def register_Ns3TypeIdValue_methods(root_module, cls):
    ## type-id.h (module 'core'): ns3::TypeIdValue::TypeIdValue() [constructor]
    cls.add_constructor([])
    ## type-id.h (module 'core'): ns3::TypeIdValue::TypeIdValue(ns3::TypeId const & value) [constructor]
    cls.add_constructor([param('ns3::TypeId const &', 'value')])
    ## type-id.h (module 'core'): ns3::TypeIdValue::TypeIdValue(ns3::TypeIdValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::TypeIdValue const &', 'arg0')])
    ## type-id.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::TypeIdValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## type-id.h (module 'core'): bool ns3::TypeIdValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeIdValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True)
    ## type-id.h (module 'core'): std::string ns3::TypeIdValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## type-id.h (module 'core'): void ns3::TypeIdValue::Set(ns3::TypeId const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::TypeId const &', 'value')])
    return

def register_Ns3UintegerValue_methods(root_module, cls):
    ## uinteger.h (module 'core'): ns3::UintegerValue::UintegerValue() [constructor]
    cls.add_constructor([])
    ## uinteger.h (module 'core'): ns3::UintegerValue::UintegerValue(uint64_t const & value) [constructor]
    cls.add_constructor([param('uint64_t const &', 'value')])
    ## uinteger.h (module 'core'): ns3::UintegerValue::UintegerValue(ns3::UintegerValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::UintegerValue const &', 'arg0')])
    ## uinteger.h (module 'core'): ns3::Ptr<ns3::AttributeValue> ns3::UintegerValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## uinteger.h (module 'core'): bool ns3::UintegerValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## uinteger.h (module 'core'): uint64_t ns3::UintegerValue::Get() const [member function]
    cls.add_method('Get', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## uinteger.h (module 'core'): std::string ns3::UintegerValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## uinteger.h (module 'core'): void ns3::UintegerValue::Set(uint64_t const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('uint64_t const &', 'value')])
    return

def register_Ns3VhtCapabilities_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## vht-capabilities.h (module 'wifi'): ns3::VhtCapabilities::VhtCapabilities(ns3::VhtCapabilities const & arg0) [constructor]
    cls.add_constructor([param('ns3::VhtCapabilities const &', 'arg0')])
    ## vht-capabilities.h (module 'wifi'): ns3::VhtCapabilities::VhtCapabilities() [constructor]
    cls.add_constructor([])
    ## vht-capabilities.h (module 'wifi'): uint8_t ns3::VhtCapabilities::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## vht-capabilities.h (module 'wifi'): ns3::WifiInformationElementId ns3::VhtCapabilities::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## vht-capabilities.h (module 'wifi'): uint8_t ns3::VhtCapabilities::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## vht-capabilities.h (module 'wifi'): uint32_t ns3::VhtCapabilities::GetMaxAmpduLength() const [member function]
    cls.add_method('GetMaxAmpduLength', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): uint16_t ns3::VhtCapabilities::GetMaxMpduLength() const [member function]
    cls.add_method('GetMaxMpduLength', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): uint16_t ns3::VhtCapabilities::GetRxHighestSupportedLgiDataRate() const [member function]
    cls.add_method('GetRxHighestSupportedLgiDataRate', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): uint8_t ns3::VhtCapabilities::GetRxLdpc() const [member function]
    cls.add_method('GetRxLdpc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): uint8_t ns3::VhtCapabilities::GetRxStbc() const [member function]
    cls.add_method('GetRxStbc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): uint16_t ns3::VhtCapabilities::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## vht-capabilities.h (module 'wifi'): uint8_t ns3::VhtCapabilities::GetSupportedChannelWidthSet() const [member function]
    cls.add_method('GetSupportedChannelWidthSet', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): uint64_t ns3::VhtCapabilities::GetSupportedMcsAndNssSet() const [member function]
    cls.add_method('GetSupportedMcsAndNssSet', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): uint8_t ns3::VhtCapabilities::GetTxStbc() const [member function]
    cls.add_method('GetTxStbc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): uint32_t ns3::VhtCapabilities::GetVhtCapabilitiesInfo() const [member function]
    cls.add_method('GetVhtCapabilitiesInfo', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): bool ns3::VhtCapabilities::IsSupportedMcs(uint8_t mcs, uint8_t nss) const [member function]
    cls.add_method('IsSupportedMcs', 
                   'bool', 
                   [param('uint8_t', 'mcs'), param('uint8_t', 'nss')], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): bool ns3::VhtCapabilities::IsSupportedRxMcs(uint8_t mcs) const [member function]
    cls.add_method('IsSupportedRxMcs', 
                   'bool', 
                   [param('uint8_t', 'mcs')], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): bool ns3::VhtCapabilities::IsSupportedTxMcs(uint8_t mcs) const [member function]
    cls.add_method('IsSupportedTxMcs', 
                   'bool', 
                   [param('uint8_t', 'mcs')], 
                   is_const=True)
    ## vht-capabilities.h (module 'wifi'): ns3::Buffer::Iterator ns3::VhtCapabilities::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetMaxAmpduLength(uint32_t maxAmpduLength) [member function]
    cls.add_method('SetMaxAmpduLength', 
                   'void', 
                   [param('uint32_t', 'maxAmpduLength')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetMaxMpduLength(uint16_t length) [member function]
    cls.add_method('SetMaxMpduLength', 
                   'void', 
                   [param('uint16_t', 'length')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetRxHighestSupportedLgiDataRate(uint16_t supportedDatarate) [member function]
    cls.add_method('SetRxHighestSupportedLgiDataRate', 
                   'void', 
                   [param('uint16_t', 'supportedDatarate')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetRxLdpc(uint8_t rxLdpc) [member function]
    cls.add_method('SetRxLdpc', 
                   'void', 
                   [param('uint8_t', 'rxLdpc')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetRxMcsMap(uint8_t mcs, uint8_t nss) [member function]
    cls.add_method('SetRxMcsMap', 
                   'void', 
                   [param('uint8_t', 'mcs'), param('uint8_t', 'nss')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetRxStbc(uint8_t rxStbc) [member function]
    cls.add_method('SetRxStbc', 
                   'void', 
                   [param('uint8_t', 'rxStbc')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetShortGuardIntervalFor160Mhz(uint8_t shortGuardInterval) [member function]
    cls.add_method('SetShortGuardIntervalFor160Mhz', 
                   'void', 
                   [param('uint8_t', 'shortGuardInterval')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetShortGuardIntervalFor80Mhz(uint8_t shortGuardInterval) [member function]
    cls.add_method('SetShortGuardIntervalFor80Mhz', 
                   'void', 
                   [param('uint8_t', 'shortGuardInterval')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetSupportedChannelWidthSet(uint8_t channelWidthSet) [member function]
    cls.add_method('SetSupportedChannelWidthSet', 
                   'void', 
                   [param('uint8_t', 'channelWidthSet')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetSupportedMcsAndNssSet(uint64_t ctrl) [member function]
    cls.add_method('SetSupportedMcsAndNssSet', 
                   'void', 
                   [param('uint64_t', 'ctrl')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetTxHighestSupportedLgiDataRate(uint16_t supportedDatarate) [member function]
    cls.add_method('SetTxHighestSupportedLgiDataRate', 
                   'void', 
                   [param('uint16_t', 'supportedDatarate')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetTxMcsMap(uint8_t mcs, uint8_t nss) [member function]
    cls.add_method('SetTxMcsMap', 
                   'void', 
                   [param('uint8_t', 'mcs'), param('uint8_t', 'nss')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetTxStbc(uint8_t txStbc) [member function]
    cls.add_method('SetTxStbc', 
                   'void', 
                   [param('uint8_t', 'txStbc')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetVhtCapabilitiesInfo(uint32_t ctrl) [member function]
    cls.add_method('SetVhtCapabilitiesInfo', 
                   'void', 
                   [param('uint32_t', 'ctrl')])
    ## vht-capabilities.h (module 'wifi'): void ns3::VhtCapabilities::SetVhtSupported(uint8_t vhtSupported) [member function]
    cls.add_method('SetVhtSupported', 
                   'void', 
                   [param('uint8_t', 'vhtSupported')])
    return

def register_Ns3VhtOperation_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## vht-operation.h (module 'wifi'): ns3::VhtOperation::VhtOperation(ns3::VhtOperation const & arg0) [constructor]
    cls.add_constructor([param('ns3::VhtOperation const &', 'arg0')])
    ## vht-operation.h (module 'wifi'): ns3::VhtOperation::VhtOperation() [constructor]
    cls.add_constructor([])
    ## vht-operation.h (module 'wifi'): uint8_t ns3::VhtOperation::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## vht-operation.h (module 'wifi'): ns3::WifiInformationElementId ns3::VhtOperation::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## vht-operation.h (module 'wifi'): uint16_t ns3::VhtOperation::GetBasicVhtMcsAndNssSet() const [member function]
    cls.add_method('GetBasicVhtMcsAndNssSet', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## vht-operation.h (module 'wifi'): uint8_t ns3::VhtOperation::GetChannelCenterFrequencySegment0() const [member function]
    cls.add_method('GetChannelCenterFrequencySegment0', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## vht-operation.h (module 'wifi'): uint8_t ns3::VhtOperation::GetChannelCenterFrequencySegment1() const [member function]
    cls.add_method('GetChannelCenterFrequencySegment1', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## vht-operation.h (module 'wifi'): uint8_t ns3::VhtOperation::GetChannelWidth() const [member function]
    cls.add_method('GetChannelWidth', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## vht-operation.h (module 'wifi'): uint8_t ns3::VhtOperation::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## vht-operation.h (module 'wifi'): uint16_t ns3::VhtOperation::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## vht-operation.h (module 'wifi'): ns3::Buffer::Iterator ns3::VhtOperation::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## vht-operation.h (module 'wifi'): void ns3::VhtOperation::SerializeInformationField(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## vht-operation.h (module 'wifi'): void ns3::VhtOperation::SetBasicVhtMcsAndNssSet(uint16_t basicVhtMcsAndNssSet) [member function]
    cls.add_method('SetBasicVhtMcsAndNssSet', 
                   'void', 
                   [param('uint16_t', 'basicVhtMcsAndNssSet')])
    ## vht-operation.h (module 'wifi'): void ns3::VhtOperation::SetChannelCenterFrequencySegment0(uint8_t channelCenterFrequencySegment0) [member function]
    cls.add_method('SetChannelCenterFrequencySegment0', 
                   'void', 
                   [param('uint8_t', 'channelCenterFrequencySegment0')])
    ## vht-operation.h (module 'wifi'): void ns3::VhtOperation::SetChannelCenterFrequencySegment1(uint8_t channelCenterFrequencySegment1) [member function]
    cls.add_method('SetChannelCenterFrequencySegment1', 
                   'void', 
                   [param('uint8_t', 'channelCenterFrequencySegment1')])
    ## vht-operation.h (module 'wifi'): void ns3::VhtOperation::SetChannelWidth(uint8_t channelWidth) [member function]
    cls.add_method('SetChannelWidth', 
                   'void', 
                   [param('uint8_t', 'channelWidth')])
    ## vht-operation.h (module 'wifi'): void ns3::VhtOperation::SetMaxVhtMcsPerNss(uint8_t nss, uint8_t maxVhtMcs) [member function]
    cls.add_method('SetMaxVhtMcsPerNss', 
                   'void', 
                   [param('uint8_t', 'nss'), param('uint8_t', 'maxVhtMcs')])
    ## vht-operation.h (module 'wifi'): void ns3::VhtOperation::SetVhtSupported(uint8_t vhtSupported) [member function]
    cls.add_method('SetVhtSupported', 
                   'void', 
                   [param('uint8_t', 'vhtSupported')])
    return

def register_Ns3WifiModeChecker_methods(root_module, cls):
    ## wifi-mode.h (module 'wifi'): ns3::WifiModeChecker::WifiModeChecker() [constructor]
    cls.add_constructor([])
    ## wifi-mode.h (module 'wifi'): ns3::WifiModeChecker::WifiModeChecker(ns3::WifiModeChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiModeChecker const &', 'arg0')])
    return

def register_Ns3WifiModeValue_methods(root_module, cls):
    ## wifi-mode.h (module 'wifi'): ns3::WifiModeValue::WifiModeValue() [constructor]
    cls.add_constructor([])
    ## wifi-mode.h (module 'wifi'): ns3::WifiModeValue::WifiModeValue(ns3::WifiMode const & value) [constructor]
    cls.add_constructor([param('ns3::WifiMode const &', 'value')])
    ## wifi-mode.h (module 'wifi'): ns3::WifiModeValue::WifiModeValue(ns3::WifiModeValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::WifiModeValue const &', 'arg0')])
    ## wifi-mode.h (module 'wifi'): ns3::Ptr<ns3::AttributeValue> ns3::WifiModeValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wifi-mode.h (module 'wifi'): bool ns3::WifiModeValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## wifi-mode.h (module 'wifi'): ns3::WifiMode ns3::WifiModeValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::WifiMode', 
                   [], 
                   is_const=True)
    ## wifi-mode.h (module 'wifi'): std::string ns3::WifiModeValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## wifi-mode.h (module 'wifi'): void ns3::WifiModeValue::Set(ns3::WifiMode const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::WifiMode const &', 'value')])
    return

def register_Ns3AddressChecker_methods(root_module, cls):
    ## address.h (module 'network'): ns3::AddressChecker::AddressChecker() [constructor]
    cls.add_constructor([])
    ## address.h (module 'network'): ns3::AddressChecker::AddressChecker(ns3::AddressChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::AddressChecker const &', 'arg0')])
    return

def register_Ns3AddressValue_methods(root_module, cls):
    ## address.h (module 'network'): ns3::AddressValue::AddressValue() [constructor]
    cls.add_constructor([])
    ## address.h (module 'network'): ns3::AddressValue::AddressValue(ns3::Address const & value) [constructor]
    cls.add_constructor([param('ns3::Address const &', 'value')])
    ## address.h (module 'network'): ns3::AddressValue::AddressValue(ns3::AddressValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::AddressValue const &', 'arg0')])
    ## address.h (module 'network'): ns3::Ptr<ns3::AttributeValue> ns3::AddressValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## address.h (module 'network'): bool ns3::AddressValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## address.h (module 'network'): ns3::Address ns3::AddressValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::Address', 
                   [], 
                   is_const=True)
    ## address.h (module 'network'): std::string ns3::AddressValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## address.h (module 'network'): void ns3::AddressValue::Set(ns3::Address const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::Address const &', 'value')])
    return

def register_Ns3BridgeChannel_methods(root_module, cls):
    ## bridge-channel.h (module 'bridge'): static ns3::TypeId ns3::BridgeChannel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## bridge-channel.h (module 'bridge'): ns3::BridgeChannel::BridgeChannel() [constructor]
    cls.add_constructor([])
    ## bridge-channel.h (module 'bridge'): void ns3::BridgeChannel::AddChannel(ns3::Ptr<ns3::Channel> bridgedChannel) [member function]
    cls.add_method('AddChannel', 
                   'void', 
                   [param('ns3::Ptr< ns3::Channel >', 'bridgedChannel')])
    ## bridge-channel.h (module 'bridge'): std::size_t ns3::BridgeChannel::GetNDevices() const [member function]
    cls.add_method('GetNDevices', 
                   'std::size_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bridge-channel.h (module 'bridge'): ns3::Ptr<ns3::NetDevice> ns3::BridgeChannel::GetDevice(std::size_t i) const [member function]
    cls.add_method('GetDevice', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [param('std::size_t', 'i')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Bool_Const_ns3WifiTxVector___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<bool, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< bool, ns3::WifiTxVector const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<bool, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<bool, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): bool ns3::CallbackImpl<bool, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::WifiTxVector const & arg0) [member operator]
    cls.add_method('operator()', 
                   'bool', 
                   [param('ns3::WifiTxVector const &', 'arg0')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Bool_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Ptr__lt__const_ns3Packet__gt___Unsigned_short_Const_ns3Address___amp___Const_ns3Address___amp___Ns3NetDevicePacketType_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): bool ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<ns3::NetDevice> arg0, ns3::Ptr<const ns3::Packet> arg1, short unsigned int arg2, ns3::Address const & arg3, ns3::Address const & arg4, ns3::NetDevice::PacketType arg5) [member operator]
    cls.add_method('operator()', 
                   'bool', 
                   [param('ns3::Ptr< ns3::NetDevice >', 'arg0'), param('ns3::Ptr< ns3::Packet const >', 'arg1'), param('short unsigned int', 'arg2'), param('ns3::Address const &', 'arg3'), param('ns3::Address const &', 'arg4'), param('ns3::NetDevice::PacketType', 'arg5')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Bool_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Ptr__lt__const_ns3Packet__gt___Unsigned_short_Const_ns3Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): bool ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<ns3::NetDevice> arg0, ns3::Ptr<const ns3::Packet> arg1, short unsigned int arg2, ns3::Address const & arg3) [member operator]
    cls.add_method('operator()', 
                   'bool', 
                   [param('ns3::Ptr< ns3::NetDevice >', 'arg0'), param('ns3::Ptr< ns3::Packet const >', 'arg1'), param('short unsigned int', 'arg2'), param('ns3::Address const &', 'arg3')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Ns3ObjectBase___star___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): ns3::ObjectBase * ns3::CallbackImpl<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()() [member operator]
    cls.add_method('operator()', 
                   'ns3::ObjectBase *', 
                   [], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Unsigned_long_Const_ns3WifiTxVector___amp___Unsigned_short_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<unsigned long long, const ns3::WifiTxVector &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<unsigned long long, const ns3::WifiTxVector &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<unsigned long long, const ns3::WifiTxVector &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< unsigned long long, ns3::WifiTxVector const &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<unsigned long long, const ns3::WifiTxVector &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<unsigned long long, const ns3::WifiTxVector &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): long unsigned int ns3::CallbackImpl<unsigned long long, const ns3::WifiTxVector &, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::WifiTxVector const & arg0, short unsigned int arg1) [member operator]
    cls.add_method('operator()', 
                   'long unsigned int', 
                   [param('ns3::WifiTxVector const &', 'arg0'), param('short unsigned int', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Unsigned_long_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): long unsigned int ns3::CallbackImpl<unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()() [member operator]
    cls.add_method('operator()', 
                   'long unsigned int', 
                   [], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Unsigned_short_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): short unsigned int ns3::CallbackImpl<unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()() [member operator]
    cls.add_method('operator()', 
                   'short unsigned int', 
                   [], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Bool_Ns3Ptr__lt__ns3Packet__gt___Ns3Mac48Address_Ns3Mac48Address_Unsigned_short_Unsigned_int_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, bool, ns3::Ptr<ns3::Packet>, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, bool, ns3::Ptr<ns3::Packet>, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, bool, ns3::Ptr<ns3::Packet>, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, bool, ns3::Ptr< ns3::Packet >, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, bool, ns3::Ptr<ns3::Packet>, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, bool, ns3::Ptr<ns3::Packet>, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, bool, ns3::Ptr<ns3::Packet>, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty>::operator()(bool arg0, ns3::Ptr<ns3::Packet> arg1, ns3::Mac48Address arg2, ns3::Mac48Address arg3, short unsigned int arg4, unsigned int arg5) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('bool', 'arg0'), param('ns3::Ptr< ns3::Packet >', 'arg1'), param('ns3::Mac48Address', 'arg2'), param('ns3::Mac48Address', 'arg3'), param('short unsigned int', 'arg4'), param('unsigned int', 'arg5')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Const_ns3WifiMacHeader___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, const ns3::WifiMacHeader &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, const ns3::WifiMacHeader &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, const ns3::WifiMacHeader &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::WifiMacHeader const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, const ns3::WifiMacHeader &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, const ns3::WifiMacHeader &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, const ns3::WifiMacHeader &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::WifiMacHeader const & arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::WifiMacHeader const &', 'arg0')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Mac48Address_Ns3Mac48Address_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Mac48Address, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Mac48Address arg0, ns3::Mac48Address arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Mac48Address', 'arg0'), param('ns3::Mac48Address', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Mac48Address_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Mac48Address, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Mac48Address arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Mac48Address', 'arg0')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::Packet const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<const ns3::Packet> arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'arg0')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3WifiMacQueueItem__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::WifiMacQueueItem const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<const ns3::WifiMacQueueItem> arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'arg0')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Ptr__lt__const_ns3Packet__gt___Unsigned_short_Const_ns3Address___amp___Const_ns3Address___amp___Ns3NetDevicePacketType_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<ns3::NetDevice> arg0, ns3::Ptr<const ns3::Packet> arg1, short unsigned int arg2, ns3::Address const & arg3, ns3::Address const & arg4, ns3::NetDevice::PacketType arg5) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::NetDevice >', 'arg0'), param('ns3::Ptr< ns3::Packet const >', 'arg1'), param('short unsigned int', 'arg2'), param('ns3::Address const &', 'arg3'), param('ns3::Address const &', 'arg4'), param('ns3::NetDevice::PacketType', 'arg5')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::NetDevice >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<ns3::NetDevice> arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::NetDevice >', 'arg0')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Time_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Time, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Time, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Time, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Time, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Time, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Time, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Time, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Time arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Time', 'arg0')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3WifiMacDropReason_Ns3Ptr__lt__const_ns3WifiMacQueueItem__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::WifiMacDropReason, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::WifiMacDropReason, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::WifiMacDropReason, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::WifiMacDropReason, ns3::Ptr< ns3::WifiMacQueueItem const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::WifiMacDropReason, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::WifiMacDropReason, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::WifiMacDropReason, ns3::Ptr<const ns3::WifiMacQueueItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::WifiMacDropReason arg0, ns3::Ptr<const ns3::WifiMacQueueItem> arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::WifiMacDropReason', 'arg0'), param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Dot11sRouteChange_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::dot11s::RouteChange, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::dot11s::RouteChange, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::dot11s::RouteChange, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::dot11s::RouteChange, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::dot11s::RouteChange, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::dot11s::RouteChange, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::dot11s::RouteChange, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::dot11s::RouteChange arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::dot11s::RouteChange', 'arg0')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Unsigned_char_Ns3Ptr__lt__const_ns3WifiMacQueueItem__gt___Const_ns3WifiTxVector___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiMacQueueItem>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiMacQueueItem>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiMacQueueItem>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, unsigned char, ns3::Ptr< ns3::WifiMacQueueItem const >, ns3::WifiTxVector const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiMacQueueItem>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiMacQueueItem>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiMacQueueItem>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(unsigned char arg0, ns3::Ptr<const ns3::WifiMacQueueItem> arg1, ns3::WifiTxVector const & arg2) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('unsigned char', 'arg0'), param('ns3::Ptr< ns3::WifiMacQueueItem const >', 'arg1'), param('ns3::WifiTxVector const &', 'arg2')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Unsigned_char_Ns3Ptr__lt__const_ns3WifiPsdu__gt___Const_ns3WifiTxVector___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiPsdu>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiPsdu>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiPsdu>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, unsigned char, ns3::Ptr< ns3::WifiPsdu const >, ns3::WifiTxVector const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiPsdu>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiPsdu>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, unsigned char, ns3::Ptr<const ns3::WifiPsdu>, const ns3::WifiTxVector &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(unsigned char arg0, ns3::Ptr<const ns3::WifiPsdu> arg1, ns3::WifiTxVector const & arg2) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('unsigned char', 'arg0'), param('ns3::Ptr< ns3::WifiPsdu const >', 'arg1'), param('ns3::WifiTxVector const &', 'arg2')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Unsigned_char_StdUnordered_map__lt__unsigned_short__ns3Ptr__lt__ns3WifiPsdu__gt____stdHash__lt__unsigned_short__gt____stdEqual_to__lt__unsigned_short__gt____stdAllocator__lt__stdPair__lt__const_unsigned_short__ns3Ptr__lt__ns3WifiPsdu__gt____gt____gt____gt_____star___Const_stdSet__lt__ns3Mac48Address__stdLess__lt__ns3Mac48Address__gt____stdAllocator__lt__ns3Mac48Address__gt____gt_____star___Unsigned_long_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, std::unordered_map<unsigned short, ns3::Ptr<ns3::WifiPsdu>, std::hash<unsigned short>, std::equal_to<unsigned short>, std::allocator<std::pair<const unsigned short, ns3::Ptr<ns3::WifiPsdu>>>> *, const std::set<ns3::Mac48Address, std::less<ns3::Mac48Address>, std::allocator<ns3::Mac48Address>> *, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, std::unordered_map<unsigned short, ns3::Ptr<ns3::WifiPsdu>, std::hash<unsigned short>, std::equal_to<unsigned short>, std::allocator<std::pair<const unsigned short, ns3::Ptr<ns3::WifiPsdu>>>> *, const std::set<ns3::Mac48Address, std::less<ns3::Mac48Address>, std::allocator<ns3::Mac48Address>> *, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, unsigned char, std::unordered_map<unsigned short, ns3::Ptr<ns3::WifiPsdu>, std::hash<unsigned short>, std::equal_to<unsigned short>, std::allocator<std::pair<const unsigned short, ns3::Ptr<ns3::WifiPsdu>>>> *, const std::set<ns3::Mac48Address, std::less<ns3::Mac48Address>, std::allocator<ns3::Mac48Address>> *, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, unsigned char, std::unordered_map< unsigned short, ns3::Ptr< ns3::WifiPsdu >, std::hash< unsigned short >, std::equal_to< unsigned short >, std::allocator< std::pair< unsigned short const, ns3::Ptr< ns3::WifiPsdu > > > > *, std::set< ns3::Mac48Address, std::less< ns3::Mac48Address >, std::allocator< ns3::Mac48Address > > const, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, unsigned char, std::unordered_map<unsigned short, ns3::Ptr<ns3::WifiPsdu>, std::hash<unsigned short>, std::equal_to<unsigned short>, std::allocator<std::pair<const unsigned short, ns3::Ptr<ns3::WifiPsdu>>>> *, const std::set<ns3::Mac48Address, std::less<ns3::Mac48Address>, std::allocator<ns3::Mac48Address>> *, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, unsigned char, std::unordered_map<unsigned short, ns3::Ptr<ns3::WifiPsdu>, std::hash<unsigned short>, std::equal_to<unsigned short>, std::allocator<std::pair<const unsigned short, ns3::Ptr<ns3::WifiPsdu>>>> *, const std::set<ns3::Mac48Address, std::less<ns3::Mac48Address>, std::allocator<ns3::Mac48Address>> *, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, unsigned char, std::unordered_map<unsigned short, ns3::Ptr<ns3::WifiPsdu>, std::hash<unsigned short>, std::equal_to<unsigned short>, std::allocator<std::pair<const unsigned short, ns3::Ptr<ns3::WifiPsdu>>>> *, const std::set<ns3::Mac48Address, std::less<ns3::Mac48Address>, std::allocator<ns3::Mac48Address>> *, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(unsigned char arg0, std::unordered_map<unsigned short, ns3::Ptr<ns3::WifiPsdu>, std::hash<unsigned short>, std::equal_to<unsigned short>, std::allocator<std::pair<const unsigned short, ns3::Ptr<ns3::WifiPsdu>>>> * arg1, std::set<ns3::Mac48Address, std::less<ns3::Mac48Address>, std::allocator<ns3::Mac48Address>> const * arg2, long unsigned int arg3) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('unsigned char', 'arg0'), param('std::unordered_map< unsigned short, ns3::Ptr< ns3::WifiPsdu > > *', 'arg1'), param('std::set< ns3::Mac48Address > const *', 'arg2'), param('long unsigned int', 'arg3')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    cls.add_copy_constructor()
    return

def register_Ns3Dot11sStack_methods(root_module, cls):
    ## dot11s-installer.h (module 'mesh'): ns3::Dot11sStack::Dot11sStack(ns3::Dot11sStack const & arg0) [constructor]
    cls.add_constructor([param('ns3::Dot11sStack const &', 'arg0')])
    ## dot11s-installer.h (module 'mesh'): ns3::Dot11sStack::Dot11sStack() [constructor]
    cls.add_constructor([])
    ## dot11s-installer.h (module 'mesh'): void ns3::Dot11sStack::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## dot11s-installer.h (module 'mesh'): static ns3::TypeId ns3::Dot11sStack::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## dot11s-installer.h (module 'mesh'): bool ns3::Dot11sStack::InstallStack(ns3::Ptr<ns3::MeshPointDevice> mp) [member function]
    cls.add_method('InstallStack', 
                   'bool', 
                   [param('ns3::Ptr< ns3::MeshPointDevice >', 'mp')], 
                   is_virtual=True)
    ## dot11s-installer.h (module 'mesh'): void ns3::Dot11sStack::Report(ns3::Ptr<ns3::MeshPointDevice> const mp, std::ostream & arg1) [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshPointDevice > const', 'mp'), param('std::ostream &', 'arg1')], 
                   is_virtual=True)
    ## dot11s-installer.h (module 'mesh'): void ns3::Dot11sStack::ResetStats(ns3::Ptr<ns3::MeshPointDevice> const mp) [member function]
    cls.add_method('ResetStats', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshPointDevice > const', 'mp')], 
                   is_virtual=True)
    return

def register_Ns3FlameStack_methods(root_module, cls):
    ## flame-installer.h (module 'mesh'): ns3::FlameStack::FlameStack(ns3::FlameStack const & arg0) [constructor]
    cls.add_constructor([param('ns3::FlameStack const &', 'arg0')])
    ## flame-installer.h (module 'mesh'): ns3::FlameStack::FlameStack() [constructor]
    cls.add_constructor([])
    ## flame-installer.h (module 'mesh'): void ns3::FlameStack::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## flame-installer.h (module 'mesh'): static ns3::TypeId ns3::FlameStack::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## flame-installer.h (module 'mesh'): bool ns3::FlameStack::InstallStack(ns3::Ptr<ns3::MeshPointDevice> mp) [member function]
    cls.add_method('InstallStack', 
                   'bool', 
                   [param('ns3::Ptr< ns3::MeshPointDevice >', 'mp')], 
                   is_virtual=True)
    ## flame-installer.h (module 'mesh'): void ns3::FlameStack::Report(ns3::Ptr<ns3::MeshPointDevice> const mp, std::ostream & arg1) [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshPointDevice > const', 'mp'), param('std::ostream &', 'arg1')], 
                   is_virtual=True)
    ## flame-installer.h (module 'mesh'): void ns3::FlameStack::ResetStats(ns3::Ptr<ns3::MeshPointDevice> const mp) [member function]
    cls.add_method('ResetStats', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshPointDevice > const', 'mp')], 
                   is_virtual=True)
    return

def register_Ns3MeshPointDevice_methods(root_module, cls):
    ## mesh-point-device.h (module 'mesh'): ns3::MeshPointDevice::MeshPointDevice(ns3::MeshPointDevice const & arg0) [constructor]
    cls.add_constructor([param('ns3::MeshPointDevice const &', 'arg0')])
    ## mesh-point-device.h (module 'mesh'): ns3::MeshPointDevice::MeshPointDevice() [constructor]
    cls.add_constructor([])
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::AddInterface(ns3::Ptr<ns3::NetDevice> port) [member function]
    cls.add_method('AddInterface', 
                   'void', 
                   [param('ns3::Ptr< ns3::NetDevice >', 'port')])
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::AddLinkChangeCallback(ns3::Callback<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> callback) [member function]
    cls.add_method('AddLinkChangeCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'callback')], 
                   is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): int64_t ns3::MeshPointDevice::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')])
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): ns3::Address ns3::MeshPointDevice::GetAddress() const [member function]
    cls.add_method('GetAddress', 
                   'ns3::Address', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): ns3::Address ns3::MeshPointDevice::GetBroadcast() const [member function]
    cls.add_method('GetBroadcast', 
                   'ns3::Address', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): ns3::Ptr<ns3::Channel> ns3::MeshPointDevice::GetChannel() const [member function]
    cls.add_method('GetChannel', 
                   'ns3::Ptr< ns3::Channel >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): ns3::Time ns3::MeshPointDevice::GetForwardingDelay() const [member function]
    cls.add_method('GetForwardingDelay', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## mesh-point-device.h (module 'mesh'): uint32_t ns3::MeshPointDevice::GetIfIndex() const [member function]
    cls.add_method('GetIfIndex', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): ns3::Ptr<ns3::NetDevice> ns3::MeshPointDevice::GetInterface(uint32_t id) const [member function]
    cls.add_method('GetInterface', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [param('uint32_t', 'id')], 
                   is_const=True)
    ## mesh-point-device.h (module 'mesh'): std::vector<ns3::Ptr<ns3::NetDevice>, std::allocator<ns3::Ptr<ns3::NetDevice>>> ns3::MeshPointDevice::GetInterfaces() const [member function]
    cls.add_method('GetInterfaces', 
                   'std::vector< ns3::Ptr< ns3::NetDevice > >', 
                   [], 
                   is_const=True)
    ## mesh-point-device.h (module 'mesh'): uint16_t ns3::MeshPointDevice::GetMtu() const [member function]
    cls.add_method('GetMtu', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): ns3::Address ns3::MeshPointDevice::GetMulticast(ns3::Ipv4Address multicastGroup) const [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Address', 
                   [param('ns3::Ipv4Address', 'multicastGroup')], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): ns3::Address ns3::MeshPointDevice::GetMulticast(ns3::Ipv6Address addr) const [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Address', 
                   [param('ns3::Ipv6Address', 'addr')], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): uint32_t ns3::MeshPointDevice::GetNInterfaces() const [member function]
    cls.add_method('GetNInterfaces', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## mesh-point-device.h (module 'mesh'): ns3::Ptr<ns3::Node> ns3::MeshPointDevice::GetNode() const [member function]
    cls.add_method('GetNode', 
                   'ns3::Ptr< ns3::Node >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): ns3::Ptr<ns3::MeshL2RoutingProtocol> ns3::MeshPointDevice::GetRoutingProtocol() const [member function]
    cls.add_method('GetRoutingProtocol', 
                   'ns3::Ptr< ns3::MeshL2RoutingProtocol >', 
                   [], 
                   is_const=True)
    ## mesh-point-device.h (module 'mesh'): static ns3::TypeId ns3::MeshPointDevice::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mesh-point-device.h (module 'mesh'): bool ns3::MeshPointDevice::IsBridge() const [member function]
    cls.add_method('IsBridge', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): bool ns3::MeshPointDevice::IsBroadcast() const [member function]
    cls.add_method('IsBroadcast', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): bool ns3::MeshPointDevice::IsLinkUp() const [member function]
    cls.add_method('IsLinkUp', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): bool ns3::MeshPointDevice::IsMulticast() const [member function]
    cls.add_method('IsMulticast', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): bool ns3::MeshPointDevice::IsPointToPoint() const [member function]
    cls.add_method('IsPointToPoint', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): bool ns3::MeshPointDevice::NeedsArp() const [member function]
    cls.add_method('NeedsArp', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::Report(std::ostream & os) const [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::ResetStats() [member function]
    cls.add_method('ResetStats', 
                   'void', 
                   [])
    ## mesh-point-device.h (module 'mesh'): bool ns3::MeshPointDevice::Send(ns3::Ptr<ns3::Packet> packet, ns3::Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('Send', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): bool ns3::MeshPointDevice::SendFrom(ns3::Ptr<ns3::Packet> packet, ns3::Address const & source, ns3::Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('SendFrom', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Address const &', 'source'), param('ns3::Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::SetAddress(ns3::Address a) [member function]
    cls.add_method('SetAddress', 
                   'void', 
                   [param('ns3::Address', 'a')], 
                   is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::SetIfIndex(uint32_t const index) [member function]
    cls.add_method('SetIfIndex', 
                   'void', 
                   [param('uint32_t const', 'index')], 
                   is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): bool ns3::MeshPointDevice::SetMtu(uint16_t const mtu) [member function]
    cls.add_method('SetMtu', 
                   'bool', 
                   [param('uint16_t const', 'mtu')], 
                   is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::SetNode(ns3::Ptr<ns3::Node> node) [member function]
    cls.add_method('SetNode', 
                   'void', 
                   [param('ns3::Ptr< ns3::Node >', 'node')], 
                   is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::SetPromiscReceiveCallback(ns3::NetDevice::PromiscReceiveCallback cb) [member function]
    cls.add_method('SetPromiscReceiveCallback', 
                   'void', 
                   [param('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >', 'cb')], 
                   is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::SetReceiveCallback(ns3::NetDevice::ReceiveCallback cb) [member function]
    cls.add_method('SetReceiveCallback', 
                   'void', 
                   [param('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')], 
                   is_virtual=True)
    ## mesh-point-device.h (module 'mesh'): void ns3::MeshPointDevice::SetRoutingProtocol(ns3::Ptr<ns3::MeshL2RoutingProtocol> protocol) [member function]
    cls.add_method('SetRoutingProtocol', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshL2RoutingProtocol >', 'protocol')])
    ## mesh-point-device.h (module 'mesh'): bool ns3::MeshPointDevice::SupportsSendFrom() const [member function]
    cls.add_method('SupportsSendFrom', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3HashImplementation_methods(root_module, cls):
    ## hash-function.h (module 'core'): ns3::Hash::Implementation::Implementation(ns3::Hash::Implementation const & arg0) [constructor]
    cls.add_constructor([param('ns3::Hash::Implementation const &', 'arg0')])
    ## hash-function.h (module 'core'): ns3::Hash::Implementation::Implementation() [constructor]
    cls.add_constructor([])
    ## hash-function.h (module 'core'): uint32_t ns3::Hash::Implementation::GetHash32(char const * buffer, std::size_t const size) [member function]
    cls.add_method('GetHash32', 
                   'uint32_t', 
                   [param('char const *', 'buffer'), param('std::size_t const', 'size')], 
                   is_pure_virtual=True, is_virtual=True)
    ## hash-function.h (module 'core'): uint64_t ns3::Hash::Implementation::GetHash64(char const * buffer, std::size_t const size) [member function]
    cls.add_method('GetHash64', 
                   'uint64_t', 
                   [param('char const *', 'buffer'), param('std::size_t const', 'size')], 
                   is_virtual=True)
    ## hash-function.h (module 'core'): void ns3::Hash::Implementation::clear() [member function]
    cls.add_method('clear', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3HashFunctionFnv1a_methods(root_module, cls):
    ## hash-fnv.h (module 'core'): ns3::Hash::Function::Fnv1a::Fnv1a(ns3::Hash::Function::Fnv1a const & arg0) [constructor]
    cls.add_constructor([param('ns3::Hash::Function::Fnv1a const &', 'arg0')])
    ## hash-fnv.h (module 'core'): ns3::Hash::Function::Fnv1a::Fnv1a() [constructor]
    cls.add_constructor([])
    ## hash-fnv.h (module 'core'): uint32_t ns3::Hash::Function::Fnv1a::GetHash32(char const * buffer, size_t const size) [member function]
    cls.add_method('GetHash32', 
                   'uint32_t', 
                   [param('char const *', 'buffer'), param('size_t const', 'size')], 
                   is_virtual=True)
    ## hash-fnv.h (module 'core'): uint64_t ns3::Hash::Function::Fnv1a::GetHash64(char const * buffer, size_t const size) [member function]
    cls.add_method('GetHash64', 
                   'uint64_t', 
                   [param('char const *', 'buffer'), param('size_t const', 'size')], 
                   is_virtual=True)
    ## hash-fnv.h (module 'core'): void ns3::Hash::Function::Fnv1a::clear() [member function]
    cls.add_method('clear', 
                   'void', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3HashFunctionHash32_methods(root_module, cls):
    ## hash-function.h (module 'core'): ns3::Hash::Function::Hash32::Hash32(ns3::Hash::Function::Hash32 const & arg0) [constructor]
    cls.add_constructor([param('ns3::Hash::Function::Hash32 const &', 'arg0')])
    ## hash-function.h (module 'core'): ns3::Hash::Function::Hash32::Hash32(ns3::Hash::Hash32Function_ptr hp) [constructor]
    cls.add_constructor([param('ns3::Hash::Hash32Function_ptr', 'hp')])
    ## hash-function.h (module 'core'): uint32_t ns3::Hash::Function::Hash32::GetHash32(char const * buffer, std::size_t const size) [member function]
    cls.add_method('GetHash32', 
                   'uint32_t', 
                   [param('char const *', 'buffer'), param('std::size_t const', 'size')], 
                   is_virtual=True)
    ## hash-function.h (module 'core'): void ns3::Hash::Function::Hash32::clear() [member function]
    cls.add_method('clear', 
                   'void', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3HashFunctionHash64_methods(root_module, cls):
    ## hash-function.h (module 'core'): ns3::Hash::Function::Hash64::Hash64(ns3::Hash::Function::Hash64 const & arg0) [constructor]
    cls.add_constructor([param('ns3::Hash::Function::Hash64 const &', 'arg0')])
    ## hash-function.h (module 'core'): ns3::Hash::Function::Hash64::Hash64(ns3::Hash::Hash64Function_ptr hp) [constructor]
    cls.add_constructor([param('ns3::Hash::Hash64Function_ptr', 'hp')])
    ## hash-function.h (module 'core'): uint32_t ns3::Hash::Function::Hash64::GetHash32(char const * buffer, std::size_t const size) [member function]
    cls.add_method('GetHash32', 
                   'uint32_t', 
                   [param('char const *', 'buffer'), param('std::size_t const', 'size')], 
                   is_virtual=True)
    ## hash-function.h (module 'core'): uint64_t ns3::Hash::Function::Hash64::GetHash64(char const * buffer, std::size_t const size) [member function]
    cls.add_method('GetHash64', 
                   'uint64_t', 
                   [param('char const *', 'buffer'), param('std::size_t const', 'size')], 
                   is_virtual=True)
    ## hash-function.h (module 'core'): void ns3::Hash::Function::Hash64::clear() [member function]
    cls.add_method('clear', 
                   'void', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3HashFunctionMurmur3_methods(root_module, cls):
    ## hash-murmur3.h (module 'core'): ns3::Hash::Function::Murmur3::Murmur3(ns3::Hash::Function::Murmur3 const & arg0) [constructor]
    cls.add_constructor([param('ns3::Hash::Function::Murmur3 const &', 'arg0')])
    ## hash-murmur3.h (module 'core'): ns3::Hash::Function::Murmur3::Murmur3() [constructor]
    cls.add_constructor([])
    ## hash-murmur3.h (module 'core'): uint32_t ns3::Hash::Function::Murmur3::GetHash32(char const * buffer, std::size_t const size) [member function]
    cls.add_method('GetHash32', 
                   'uint32_t', 
                   [param('char const *', 'buffer'), param('std::size_t const', 'size')], 
                   is_virtual=True)
    ## hash-murmur3.h (module 'core'): uint64_t ns3::Hash::Function::Murmur3::GetHash64(char const * buffer, std::size_t const size) [member function]
    cls.add_method('GetHash64', 
                   'uint64_t', 
                   [param('char const *', 'buffer'), param('std::size_t const', 'size')], 
                   is_virtual=True)
    ## hash-murmur3.h (module 'core'): void ns3::Hash::Function::Murmur3::clear() [member function]
    cls.add_method('clear', 
                   'void', 
                   [], 
                   is_virtual=True)
    return

def register_Ns3Dot11sDestinationAddressUnit_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## ie-dot11s-preq.h (module 'mesh'): ns3::dot11s::DestinationAddressUnit::DestinationAddressUnit(ns3::dot11s::DestinationAddressUnit const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::DestinationAddressUnit const &', 'arg0')])
    ## ie-dot11s-preq.h (module 'mesh'): ns3::dot11s::DestinationAddressUnit::DestinationAddressUnit() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-preq.h (module 'mesh'): uint32_t ns3::dot11s::DestinationAddressUnit::GetDestSeqNumber() const [member function]
    cls.add_method('GetDestSeqNumber', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): ns3::Mac48Address ns3::dot11s::DestinationAddressUnit::GetDestinationAddress() const [member function]
    cls.add_method('GetDestinationAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): bool ns3::dot11s::DestinationAddressUnit::IsDo() [member function]
    cls.add_method('IsDo', 
                   'bool', 
                   [])
    ## ie-dot11s-preq.h (module 'mesh'): bool ns3::dot11s::DestinationAddressUnit::IsRf() [member function]
    cls.add_method('IsRf', 
                   'bool', 
                   [])
    ## ie-dot11s-preq.h (module 'mesh'): bool ns3::dot11s::DestinationAddressUnit::IsUsn() [member function]
    cls.add_method('IsUsn', 
                   'bool', 
                   [])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::DestinationAddressUnit::SetDestSeqNumber(uint32_t dest_seq_number) [member function]
    cls.add_method('SetDestSeqNumber', 
                   'void', 
                   [param('uint32_t', 'dest_seq_number')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::DestinationAddressUnit::SetDestinationAddress(ns3::Mac48Address dest_address) [member function]
    cls.add_method('SetDestinationAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'dest_address')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::DestinationAddressUnit::SetFlags(bool doFlag, bool rfFlag, bool usnFlag) [member function]
    cls.add_method('SetFlags', 
                   'void', 
                   [param('bool', 'doFlag'), param('bool', 'rfFlag'), param('bool', 'usnFlag')])
    return

def register_Ns3Dot11sDot11sMeshCapability_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability::Dot11sMeshCapability(ns3::dot11s::Dot11sMeshCapability const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::Dot11sMeshCapability const &', 'arg0')])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability::Dot11sMeshCapability() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::Buffer::Iterator ns3::dot11s::Dot11sMeshCapability::Deserialize(ns3::Buffer::Iterator i) [member function]
    cls.add_method('Deserialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'i')])
    ## ie-dot11s-configuration.h (module 'mesh'): uint8_t ns3::dot11s::Dot11sMeshCapability::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-configuration.h (module 'mesh'): uint8_t ns3::dot11s::Dot11sMeshCapability::GetUint8() const [member function]
    cls.add_method('GetUint8', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-configuration.h (module 'mesh'): bool ns3::dot11s::Dot11sMeshCapability::Is(uint8_t cap, uint8_t n) const [member function]
    cls.add_method('Is', 
                   'bool', 
                   [param('uint8_t', 'cap'), param('uint8_t', 'n')], 
                   is_const=True)
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::Buffer::Iterator ns3::dot11s::Dot11sMeshCapability::Serialize(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('Serialize', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True)
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability::MCCAEnabled [variable]
    cls.add_instance_attribute('MCCAEnabled', 'bool', is_const=False)
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability::MCCASupported [variable]
    cls.add_instance_attribute('MCCASupported', 'bool', is_const=False)
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability::TBTTAdjustment [variable]
    cls.add_instance_attribute('TBTTAdjustment', 'bool', is_const=False)
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability::acceptPeerLinks [variable]
    cls.add_instance_attribute('acceptPeerLinks', 'bool', is_const=False)
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability::beaconTimingReport [variable]
    cls.add_instance_attribute('beaconTimingReport', 'bool', is_const=False)
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability::forwarding [variable]
    cls.add_instance_attribute('forwarding', 'bool', is_const=False)
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability::powerSaveLevel [variable]
    cls.add_instance_attribute('powerSaveLevel', 'bool', is_const=False)
    return

def register_Ns3Dot11sHwmpProtocol_methods(root_module, cls):
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::HwmpProtocol::HwmpProtocol() [constructor]
    cls.add_constructor([])
    ## hwmp-protocol.h (module 'mesh'): int64_t ns3::dot11s::HwmpProtocol::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')])
    ## hwmp-protocol.h (module 'mesh'): void ns3::dot11s::HwmpProtocol::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## hwmp-protocol.h (module 'mesh'): ns3::Ptr<ns3::dot11s::HwmpRtable> ns3::dot11s::HwmpProtocol::GetRoutingTable() const [member function]
    cls.add_method('GetRoutingTable', 
                   'ns3::Ptr< ns3::dot11s::HwmpRtable >', 
                   [], 
                   is_const=True)
    ## hwmp-protocol.h (module 'mesh'): static ns3::TypeId ns3::dot11s::HwmpProtocol::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## hwmp-protocol.h (module 'mesh'): bool ns3::dot11s::HwmpProtocol::Install(ns3::Ptr<ns3::MeshPointDevice> mp) [member function]
    cls.add_method('Install', 
                   'bool', 
                   [param('ns3::Ptr< ns3::MeshPointDevice >', 'mp')])
    ## hwmp-protocol.h (module 'mesh'): void ns3::dot11s::HwmpProtocol::PeerLinkStatus(ns3::Mac48Address meshPontAddress, ns3::Mac48Address peerAddress, uint32_t interface, bool status) [member function]
    cls.add_method('PeerLinkStatus', 
                   'void', 
                   [param('ns3::Mac48Address', 'meshPontAddress'), param('ns3::Mac48Address', 'peerAddress'), param('uint32_t', 'interface'), param('bool', 'status')])
    ## hwmp-protocol.h (module 'mesh'): bool ns3::dot11s::HwmpProtocol::RemoveRoutingStuff(uint32_t fromIface, ns3::Mac48Address const source, ns3::Mac48Address const destination, ns3::Ptr<ns3::Packet> packet, uint16_t & protocolType) [member function]
    cls.add_method('RemoveRoutingStuff', 
                   'bool', 
                   [param('uint32_t', 'fromIface'), param('ns3::Mac48Address const', 'source'), param('ns3::Mac48Address const', 'destination'), param('ns3::Ptr< ns3::Packet >', 'packet'), param('uint16_t &', 'protocolType')], 
                   is_virtual=True)
    ## hwmp-protocol.h (module 'mesh'): void ns3::dot11s::HwmpProtocol::Report(std::ostream & os) const [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## hwmp-protocol.h (module 'mesh'): bool ns3::dot11s::HwmpProtocol::RequestRoute(uint32_t sourceIface, ns3::Mac48Address const source, ns3::Mac48Address const destination, ns3::Ptr<const ns3::Packet> packet, uint16_t protocolType, ns3::MeshL2RoutingProtocol::RouteReplyCallback routeReply) [member function]
    cls.add_method('RequestRoute', 
                   'bool', 
                   [param('uint32_t', 'sourceIface'), param('ns3::Mac48Address const', 'source'), param('ns3::Mac48Address const', 'destination'), param('ns3::Ptr< ns3::Packet const >', 'packet'), param('uint16_t', 'protocolType'), param('ns3::Callback< void, bool, ns3::Ptr< ns3::Packet >, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty >', 'routeReply')], 
                   is_virtual=True)
    ## hwmp-protocol.h (module 'mesh'): void ns3::dot11s::HwmpProtocol::ResetStats() [member function]
    cls.add_method('ResetStats', 
                   'void', 
                   [])
    ## hwmp-protocol.h (module 'mesh'): void ns3::dot11s::HwmpProtocol::SetNeighboursCallback(ns3::Callback<std::vector<ns3::Mac48Address, std::allocator<ns3::Mac48Address>>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> cb) [member function]
    cls.add_method('SetNeighboursCallback', 
                   'void', 
                   [param('ns3::Callback< std::vector< ns3::Mac48Address >, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')])
    ## hwmp-protocol.h (module 'mesh'): void ns3::dot11s::HwmpProtocol::SetRoot() [member function]
    cls.add_method('SetRoot', 
                   'void', 
                   [])
    ## hwmp-protocol.h (module 'mesh'): void ns3::dot11s::HwmpProtocol::UnsetRoot() [member function]
    cls.add_method('UnsetRoot', 
                   'void', 
                   [])
    ## hwmp-protocol.h (module 'mesh'): void ns3::dot11s::HwmpProtocol::DoInitialize() [member function]
    cls.add_method('DoInitialize', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3Dot11sHwmpProtocolFailedDestination_methods(root_module, cls):
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::HwmpProtocol::FailedDestination::FailedDestination() [constructor]
    cls.add_constructor([])
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::HwmpProtocol::FailedDestination::FailedDestination(ns3::dot11s::HwmpProtocol::FailedDestination const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::HwmpProtocol::FailedDestination const &', 'arg0')])
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::HwmpProtocol::FailedDestination::destination [variable]
    cls.add_instance_attribute('destination', 'ns3::Mac48Address', is_const=False)
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::HwmpProtocol::FailedDestination::seqnum [variable]
    cls.add_instance_attribute('seqnum', 'uint32_t', is_const=False)
    return

def register_Ns3Dot11sHwmpRtable_methods(root_module, cls):
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::HwmpRtable(ns3::dot11s::HwmpRtable const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::HwmpRtable const &', 'arg0')])
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::HwmpRtable() [constructor]
    cls.add_constructor([])
    ## hwmp-rtable.h (module 'mesh'): void ns3::dot11s::HwmpRtable::AddPrecursor(ns3::Mac48Address destination, uint32_t precursorInterface, ns3::Mac48Address precursorAddress, ns3::Time lifetime) [member function]
    cls.add_method('AddPrecursor', 
                   'void', 
                   [param('ns3::Mac48Address', 'destination'), param('uint32_t', 'precursorInterface'), param('ns3::Mac48Address', 'precursorAddress'), param('ns3::Time', 'lifetime')])
    ## hwmp-rtable.h (module 'mesh'): void ns3::dot11s::HwmpRtable::AddProactivePath(uint32_t metric, ns3::Mac48Address root, ns3::Mac48Address retransmitter, uint32_t interface, ns3::Time lifetime, uint32_t seqnum) [member function]
    cls.add_method('AddProactivePath', 
                   'void', 
                   [param('uint32_t', 'metric'), param('ns3::Mac48Address', 'root'), param('ns3::Mac48Address', 'retransmitter'), param('uint32_t', 'interface'), param('ns3::Time', 'lifetime'), param('uint32_t', 'seqnum')])
    ## hwmp-rtable.h (module 'mesh'): void ns3::dot11s::HwmpRtable::AddReactivePath(ns3::Mac48Address destination, ns3::Mac48Address retransmitter, uint32_t interface, uint32_t metric, ns3::Time lifetime, uint32_t seqnum) [member function]
    cls.add_method('AddReactivePath', 
                   'void', 
                   [param('ns3::Mac48Address', 'destination'), param('ns3::Mac48Address', 'retransmitter'), param('uint32_t', 'interface'), param('uint32_t', 'metric'), param('ns3::Time', 'lifetime'), param('uint32_t', 'seqnum')])
    ## hwmp-rtable.h (module 'mesh'): void ns3::dot11s::HwmpRtable::DeleteProactivePath() [member function]
    cls.add_method('DeleteProactivePath', 
                   'void', 
                   [])
    ## hwmp-rtable.h (module 'mesh'): void ns3::dot11s::HwmpRtable::DeleteProactivePath(ns3::Mac48Address root) [member function]
    cls.add_method('DeleteProactivePath', 
                   'void', 
                   [param('ns3::Mac48Address', 'root')])
    ## hwmp-rtable.h (module 'mesh'): void ns3::dot11s::HwmpRtable::DeleteReactivePath(ns3::Mac48Address destination) [member function]
    cls.add_method('DeleteReactivePath', 
                   'void', 
                   [param('ns3::Mac48Address', 'destination')])
    ## hwmp-rtable.h (module 'mesh'): void ns3::dot11s::HwmpRtable::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::PrecursorList ns3::dot11s::HwmpRtable::GetPrecursors(ns3::Mac48Address destination) [member function]
    cls.add_method('GetPrecursors', 
                   'ns3::dot11s::HwmpRtable::PrecursorList', 
                   [param('ns3::Mac48Address', 'destination')])
    ## hwmp-rtable.h (module 'mesh'): static ns3::TypeId ns3::dot11s::HwmpRtable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## hwmp-rtable.h (module 'mesh'): std::vector<ns3::dot11s::HwmpProtocol::FailedDestination, std::allocator<ns3::dot11s::HwmpProtocol::FailedDestination>> ns3::dot11s::HwmpRtable::GetUnreachableDestinations(ns3::Mac48Address peerAddress) [member function]
    cls.add_method('GetUnreachableDestinations', 
                   'std::vector< ns3::dot11s::HwmpProtocol::FailedDestination >', 
                   [param('ns3::Mac48Address', 'peerAddress')])
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult ns3::dot11s::HwmpRtable::LookupProactive() [member function]
    cls.add_method('LookupProactive', 
                   'ns3::dot11s::HwmpRtable::LookupResult', 
                   [])
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult ns3::dot11s::HwmpRtable::LookupProactiveExpired() [member function]
    cls.add_method('LookupProactiveExpired', 
                   'ns3::dot11s::HwmpRtable::LookupResult', 
                   [])
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult ns3::dot11s::HwmpRtable::LookupReactive(ns3::Mac48Address destination) [member function]
    cls.add_method('LookupReactive', 
                   'ns3::dot11s::HwmpRtable::LookupResult', 
                   [param('ns3::Mac48Address', 'destination')])
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult ns3::dot11s::HwmpRtable::LookupReactiveExpired(ns3::Mac48Address destination) [member function]
    cls.add_method('LookupReactiveExpired', 
                   'ns3::dot11s::HwmpRtable::LookupResult', 
                   [param('ns3::Mac48Address', 'destination')])
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::INTERFACE_ANY [variable]
    cls.add_static_attribute('INTERFACE_ANY', 'uint32_t const', is_const=True)
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::MAX_METRIC [variable]
    cls.add_static_attribute('MAX_METRIC', 'uint32_t const', is_const=True)
    return

def register_Ns3Dot11sHwmpRtableLookupResult_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult::LookupResult(ns3::dot11s::HwmpRtable::LookupResult const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::HwmpRtable::LookupResult const &', 'arg0')])
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult::LookupResult(ns3::Mac48Address r=ns3::Mac48Address::GetBroadcast(), uint32_t i=ns3::dot11s::HwmpRtable::INTERFACE_ANY, uint32_t m=ns3::dot11s::HwmpRtable::MAX_METRIC, uint32_t s=0, ns3::Time l=ns3::Seconds(0.)) [constructor]
    cls.add_constructor([param('ns3::Mac48Address', 'r', default_value='ns3::Mac48Address::GetBroadcast()'), param('uint32_t', 'i', default_value='ns3::dot11s::HwmpRtable::INTERFACE_ANY'), param('uint32_t', 'm', default_value='ns3::dot11s::HwmpRtable::MAX_METRIC'), param('uint32_t', 's', default_value='0'), param('ns3::Time', 'l', default_value='ns3::Seconds(0.)')])
    ## hwmp-rtable.h (module 'mesh'): bool ns3::dot11s::HwmpRtable::LookupResult::IsValid() const [member function]
    cls.add_method('IsValid', 
                   'bool', 
                   [], 
                   is_const=True)
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult::ifIndex [variable]
    cls.add_instance_attribute('ifIndex', 'uint32_t', is_const=False)
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult::lifetime [variable]
    cls.add_instance_attribute('lifetime', 'ns3::Time', is_const=False)
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult::metric [variable]
    cls.add_instance_attribute('metric', 'uint32_t', is_const=False)
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult::retransmitter [variable]
    cls.add_instance_attribute('retransmitter', 'ns3::Mac48Address', is_const=False)
    ## hwmp-rtable.h (module 'mesh'): ns3::dot11s::HwmpRtable::LookupResult::seqnum [variable]
    cls.add_instance_attribute('seqnum', 'uint32_t', is_const=False)
    return

def register_Ns3Dot11sIeBeaconTiming_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## ie-dot11s-beacon-timing.h (module 'mesh'): ns3::dot11s::IeBeaconTiming::IeBeaconTiming(ns3::dot11s::IeBeaconTiming const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IeBeaconTiming const &', 'arg0')])
    ## ie-dot11s-beacon-timing.h (module 'mesh'): ns3::dot11s::IeBeaconTiming::IeBeaconTiming() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-beacon-timing.h (module 'mesh'): void ns3::dot11s::IeBeaconTiming::AddNeighboursTimingElementUnit(uint16_t aid, ns3::Time last_beacon, ns3::Time beacon_interval) [member function]
    cls.add_method('AddNeighboursTimingElementUnit', 
                   'void', 
                   [param('uint16_t', 'aid'), param('ns3::Time', 'last_beacon'), param('ns3::Time', 'beacon_interval')])
    ## ie-dot11s-beacon-timing.h (module 'mesh'): void ns3::dot11s::IeBeaconTiming::ClearTimingElement() [member function]
    cls.add_method('ClearTimingElement', 
                   'void', 
                   [])
    ## ie-dot11s-beacon-timing.h (module 'mesh'): void ns3::dot11s::IeBeaconTiming::DelNeighboursTimingElementUnit(uint16_t aid, ns3::Time last_beacon, ns3::Time beacon_interval) [member function]
    cls.add_method('DelNeighboursTimingElementUnit', 
                   'void', 
                   [param('uint16_t', 'aid'), param('ns3::Time', 'last_beacon'), param('ns3::Time', 'beacon_interval')])
    ## ie-dot11s-beacon-timing.h (module 'mesh'): uint8_t ns3::dot11s::IeBeaconTiming::DeserializeInformationField(ns3::Buffer::Iterator i, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'i'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ie-dot11s-beacon-timing.h (module 'mesh'): ns3::WifiInformationElementId ns3::dot11s::IeBeaconTiming::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-beacon-timing.h (module 'mesh'): uint8_t ns3::dot11s::IeBeaconTiming::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-beacon-timing.h (module 'mesh'): ns3::dot11s::IeBeaconTiming::NeighboursTimingUnitsList ns3::dot11s::IeBeaconTiming::GetNeighboursTimingElementsList() [member function]
    cls.add_method('GetNeighboursTimingElementsList', 
                   'ns3::dot11s::IeBeaconTiming::NeighboursTimingUnitsList', 
                   [])
    ## ie-dot11s-beacon-timing.h (module 'mesh'): void ns3::dot11s::IeBeaconTiming::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-beacon-timing.h (module 'mesh'): void ns3::dot11s::IeBeaconTiming::SerializeInformationField(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3Dot11sIeBeaconTimingUnit_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## ie-dot11s-beacon-timing.h (module 'mesh'): ns3::dot11s::IeBeaconTimingUnit::IeBeaconTimingUnit(ns3::dot11s::IeBeaconTimingUnit const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IeBeaconTimingUnit const &', 'arg0')])
    ## ie-dot11s-beacon-timing.h (module 'mesh'): ns3::dot11s::IeBeaconTimingUnit::IeBeaconTimingUnit() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-beacon-timing.h (module 'mesh'): uint8_t ns3::dot11s::IeBeaconTimingUnit::GetAid() const [member function]
    cls.add_method('GetAid', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-beacon-timing.h (module 'mesh'): uint16_t ns3::dot11s::IeBeaconTimingUnit::GetBeaconInterval() const [member function]
    cls.add_method('GetBeaconInterval', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-beacon-timing.h (module 'mesh'): uint16_t ns3::dot11s::IeBeaconTimingUnit::GetLastBeacon() const [member function]
    cls.add_method('GetLastBeacon', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-beacon-timing.h (module 'mesh'): void ns3::dot11s::IeBeaconTimingUnit::SetAid(uint8_t aid) [member function]
    cls.add_method('SetAid', 
                   'void', 
                   [param('uint8_t', 'aid')])
    ## ie-dot11s-beacon-timing.h (module 'mesh'): void ns3::dot11s::IeBeaconTimingUnit::SetBeaconInterval(uint16_t beaconInterval) [member function]
    cls.add_method('SetBeaconInterval', 
                   'void', 
                   [param('uint16_t', 'beaconInterval')])
    ## ie-dot11s-beacon-timing.h (module 'mesh'): void ns3::dot11s::IeBeaconTimingUnit::SetLastBeacon(uint16_t lastBeacon) [member function]
    cls.add_method('SetLastBeacon', 
                   'void', 
                   [param('uint16_t', 'lastBeacon')])
    return

def register_Ns3Dot11sIeConfiguration_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_output_stream_operator()
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::IeConfiguration::IeConfiguration(ns3::dot11s::IeConfiguration const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IeConfiguration const &', 'arg0')])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::IeConfiguration::IeConfiguration() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-configuration.h (module 'mesh'): uint8_t ns3::dot11s::IeConfiguration::DeserializeInformationField(ns3::Buffer::Iterator i, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'i'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::WifiInformationElementId ns3::dot11s::IeConfiguration::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-configuration.h (module 'mesh'): uint8_t ns3::dot11s::IeConfiguration::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-configuration.h (module 'mesh'): uint8_t ns3::dot11s::IeConfiguration::GetNeighborCount() [member function]
    cls.add_method('GetNeighborCount', 
                   'uint8_t', 
                   [])
    ## ie-dot11s-configuration.h (module 'mesh'): bool ns3::dot11s::IeConfiguration::IsAirtime() [member function]
    cls.add_method('IsAirtime', 
                   'bool', 
                   [])
    ## ie-dot11s-configuration.h (module 'mesh'): bool ns3::dot11s::IeConfiguration::IsHWMP() [member function]
    cls.add_method('IsHWMP', 
                   'bool', 
                   [])
    ## ie-dot11s-configuration.h (module 'mesh'): ns3::dot11s::Dot11sMeshCapability const & ns3::dot11s::IeConfiguration::MeshCapability() [member function]
    cls.add_method('MeshCapability', 
                   'ns3::dot11s::Dot11sMeshCapability const &', 
                   [])
    ## ie-dot11s-configuration.h (module 'mesh'): void ns3::dot11s::IeConfiguration::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-configuration.h (module 'mesh'): void ns3::dot11s::IeConfiguration::SerializeInformationField(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-configuration.h (module 'mesh'): void ns3::dot11s::IeConfiguration::SetMetric(ns3::dot11s::dot11sPathSelectionMetric metricId) [member function]
    cls.add_method('SetMetric', 
                   'void', 
                   [param('ns3::dot11s::dot11sPathSelectionMetric', 'metricId')])
    ## ie-dot11s-configuration.h (module 'mesh'): void ns3::dot11s::IeConfiguration::SetNeighborCount(uint8_t neighbors) [member function]
    cls.add_method('SetNeighborCount', 
                   'void', 
                   [param('uint8_t', 'neighbors')])
    ## ie-dot11s-configuration.h (module 'mesh'): void ns3::dot11s::IeConfiguration::SetRouting(ns3::dot11s::dot11sPathSelectionProtocol routingId) [member function]
    cls.add_method('SetRouting', 
                   'void', 
                   [param('ns3::dot11s::dot11sPathSelectionProtocol', 'routingId')])
    return

def register_Ns3Dot11sIeLinkMetricReport_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('>')
    cls.add_binary_comparison_operator('<')
    cls.add_output_stream_operator()
    ## ie-dot11s-metric-report.h (module 'mesh'): ns3::dot11s::IeLinkMetricReport::IeLinkMetricReport(ns3::dot11s::IeLinkMetricReport const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IeLinkMetricReport const &', 'arg0')])
    ## ie-dot11s-metric-report.h (module 'mesh'): ns3::dot11s::IeLinkMetricReport::IeLinkMetricReport() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-metric-report.h (module 'mesh'): ns3::dot11s::IeLinkMetricReport::IeLinkMetricReport(uint32_t metric) [constructor]
    cls.add_constructor([param('uint32_t', 'metric')])
    ## ie-dot11s-metric-report.h (module 'mesh'): uint8_t ns3::dot11s::IeLinkMetricReport::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ie-dot11s-metric-report.h (module 'mesh'): ns3::WifiInformationElementId ns3::dot11s::IeLinkMetricReport::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-metric-report.h (module 'mesh'): uint8_t ns3::dot11s::IeLinkMetricReport::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-metric-report.h (module 'mesh'): uint32_t ns3::dot11s::IeLinkMetricReport::GetMetric() [member function]
    cls.add_method('GetMetric', 
                   'uint32_t', 
                   [])
    ## ie-dot11s-metric-report.h (module 'mesh'): void ns3::dot11s::IeLinkMetricReport::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-metric-report.h (module 'mesh'): void ns3::dot11s::IeLinkMetricReport::SerializeInformationField(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-metric-report.h (module 'mesh'): void ns3::dot11s::IeLinkMetricReport::SetMetric(uint32_t metric) [member function]
    cls.add_method('SetMetric', 
                   'void', 
                   [param('uint32_t', 'metric')])
    return

def register_Ns3Dot11sIeMeshId_methods(root_module, cls):
    cls.add_output_stream_operator()
    cls.add_binary_comparison_operator('==')
    ## ie-dot11s-id.h (module 'mesh'): ns3::dot11s::IeMeshId::IeMeshId(ns3::dot11s::IeMeshId const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IeMeshId const &', 'arg0')])
    ## ie-dot11s-id.h (module 'mesh'): ns3::dot11s::IeMeshId::IeMeshId() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-id.h (module 'mesh'): ns3::dot11s::IeMeshId::IeMeshId(std::string s) [constructor]
    cls.add_constructor([param('std::string', 's')])
    ## ie-dot11s-id.h (module 'mesh'): uint8_t ns3::dot11s::IeMeshId::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ie-dot11s-id.h (module 'mesh'): ns3::WifiInformationElementId ns3::dot11s::IeMeshId::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-id.h (module 'mesh'): uint8_t ns3::dot11s::IeMeshId::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-id.h (module 'mesh'): bool ns3::dot11s::IeMeshId::IsBroadcast() const [member function]
    cls.add_method('IsBroadcast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ie-dot11s-id.h (module 'mesh'): bool ns3::dot11s::IeMeshId::IsEqual(ns3::dot11s::IeMeshId const & o) const [member function]
    cls.add_method('IsEqual', 
                   'bool', 
                   [param('ns3::dot11s::IeMeshId const &', 'o')], 
                   is_const=True)
    ## ie-dot11s-id.h (module 'mesh'): char * ns3::dot11s::IeMeshId::PeekString() const [member function]
    cls.add_method('PeekString', 
                   'char *', 
                   [], 
                   is_const=True)
    ## ie-dot11s-id.h (module 'mesh'): void ns3::dot11s::IeMeshId::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-id.h (module 'mesh'): void ns3::dot11s::IeMeshId::SerializeInformationField(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3Dot11sIePeerManagement_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_output_stream_operator()
    ## ie-dot11s-peer-management.h (module 'mesh'): ns3::dot11s::IePeerManagement::IePeerManagement(ns3::dot11s::IePeerManagement const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IePeerManagement const &', 'arg0')])
    ## ie-dot11s-peer-management.h (module 'mesh'): ns3::dot11s::IePeerManagement::IePeerManagement() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-peer-management.h (module 'mesh'): uint8_t ns3::dot11s::IePeerManagement::DeserializeInformationField(ns3::Buffer::Iterator i, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'i'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): ns3::WifiInformationElementId ns3::dot11s::IePeerManagement::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): uint8_t ns3::dot11s::IePeerManagement::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): uint16_t ns3::dot11s::IePeerManagement::GetLocalLinkId() const [member function]
    cls.add_method('GetLocalLinkId', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): uint16_t ns3::dot11s::IePeerManagement::GetPeerLinkId() const [member function]
    cls.add_method('GetPeerLinkId', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): ns3::dot11s::PmpReasonCode ns3::dot11s::IePeerManagement::GetReasonCode() const [member function]
    cls.add_method('GetReasonCode', 
                   'ns3::dot11s::PmpReasonCode', 
                   [], 
                   is_const=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): uint8_t ns3::dot11s::IePeerManagement::GetSubtype() const [member function]
    cls.add_method('GetSubtype', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): void ns3::dot11s::IePeerManagement::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): void ns3::dot11s::IePeerManagement::SerializeInformationField(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): void ns3::dot11s::IePeerManagement::SetPeerClose(uint16_t localLinkID, uint16_t peerLinkId, ns3::dot11s::PmpReasonCode reasonCode) [member function]
    cls.add_method('SetPeerClose', 
                   'void', 
                   [param('uint16_t', 'localLinkID'), param('uint16_t', 'peerLinkId'), param('ns3::dot11s::PmpReasonCode', 'reasonCode')])
    ## ie-dot11s-peer-management.h (module 'mesh'): void ns3::dot11s::IePeerManagement::SetPeerConfirm(uint16_t localLinkID, uint16_t peerLinkId) [member function]
    cls.add_method('SetPeerConfirm', 
                   'void', 
                   [param('uint16_t', 'localLinkID'), param('uint16_t', 'peerLinkId')])
    ## ie-dot11s-peer-management.h (module 'mesh'): void ns3::dot11s::IePeerManagement::SetPeerOpen(uint16_t localLinkId) [member function]
    cls.add_method('SetPeerOpen', 
                   'void', 
                   [param('uint16_t', 'localLinkId')])
    ## ie-dot11s-peer-management.h (module 'mesh'): bool ns3::dot11s::IePeerManagement::SubtypeIsClose() const [member function]
    cls.add_method('SubtypeIsClose', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): bool ns3::dot11s::IePeerManagement::SubtypeIsConfirm() const [member function]
    cls.add_method('SubtypeIsConfirm', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ie-dot11s-peer-management.h (module 'mesh'): bool ns3::dot11s::IePeerManagement::SubtypeIsOpen() const [member function]
    cls.add_method('SubtypeIsOpen', 
                   'bool', 
                   [], 
                   is_const=True)
    return

def register_Ns3Dot11sIePeeringProtocol_methods(root_module, cls):
    ## ie-dot11s-peering-protocol.h (module 'mesh'): ns3::dot11s::IePeeringProtocol::IePeeringProtocol(ns3::dot11s::IePeeringProtocol const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IePeeringProtocol const &', 'arg0')])
    ## ie-dot11s-peering-protocol.h (module 'mesh'): ns3::dot11s::IePeeringProtocol::IePeeringProtocol() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-peering-protocol.h (module 'mesh'): uint8_t ns3::dot11s::IePeeringProtocol::DeserializeInformationField(ns3::Buffer::Iterator i, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'i'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ie-dot11s-peering-protocol.h (module 'mesh'): ns3::WifiInformationElementId ns3::dot11s::IePeeringProtocol::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-peering-protocol.h (module 'mesh'): uint8_t ns3::dot11s::IePeeringProtocol::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-peering-protocol.h (module 'mesh'): void ns3::dot11s::IePeeringProtocol::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-peering-protocol.h (module 'mesh'): void ns3::dot11s::IePeeringProtocol::SerializeInformationField(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3Dot11sIePerr_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_output_stream_operator()
    ## ie-dot11s-perr.h (module 'mesh'): ns3::dot11s::IePerr::IePerr(ns3::dot11s::IePerr const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IePerr const &', 'arg0')])
    ## ie-dot11s-perr.h (module 'mesh'): ns3::dot11s::IePerr::IePerr() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-perr.h (module 'mesh'): void ns3::dot11s::IePerr::AddAddressUnit(ns3::dot11s::HwmpProtocol::FailedDestination unit) [member function]
    cls.add_method('AddAddressUnit', 
                   'void', 
                   [param('ns3::dot11s::HwmpProtocol::FailedDestination', 'unit')])
    ## ie-dot11s-perr.h (module 'mesh'): void ns3::dot11s::IePerr::DeleteAddressUnit(ns3::Mac48Address address) [member function]
    cls.add_method('DeleteAddressUnit', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## ie-dot11s-perr.h (module 'mesh'): uint8_t ns3::dot11s::IePerr::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ie-dot11s-perr.h (module 'mesh'): ns3::WifiInformationElementId ns3::dot11s::IePerr::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-perr.h (module 'mesh'): std::vector<ns3::dot11s::HwmpProtocol::FailedDestination, std::allocator<ns3::dot11s::HwmpProtocol::FailedDestination>> ns3::dot11s::IePerr::GetAddressUnitVector() const [member function]
    cls.add_method('GetAddressUnitVector', 
                   'std::vector< ns3::dot11s::HwmpProtocol::FailedDestination >', 
                   [], 
                   is_const=True)
    ## ie-dot11s-perr.h (module 'mesh'): uint8_t ns3::dot11s::IePerr::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-perr.h (module 'mesh'): uint8_t ns3::dot11s::IePerr::GetNumOfDest() const [member function]
    cls.add_method('GetNumOfDest', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-perr.h (module 'mesh'): bool ns3::dot11s::IePerr::IsFull() const [member function]
    cls.add_method('IsFull', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ie-dot11s-perr.h (module 'mesh'): void ns3::dot11s::IePerr::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-perr.h (module 'mesh'): void ns3::dot11s::IePerr::ResetPerr() [member function]
    cls.add_method('ResetPerr', 
                   'void', 
                   [])
    ## ie-dot11s-perr.h (module 'mesh'): void ns3::dot11s::IePerr::SerializeInformationField(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3Dot11sIePrep_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_output_stream_operator()
    ## ie-dot11s-prep.h (module 'mesh'): ns3::dot11s::IePrep::IePrep(ns3::dot11s::IePrep const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IePrep const &', 'arg0')])
    ## ie-dot11s-prep.h (module 'mesh'): ns3::dot11s::IePrep::IePrep() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::DecrementTtl() [member function]
    cls.add_method('DecrementTtl', 
                   'void', 
                   [])
    ## ie-dot11s-prep.h (module 'mesh'): uint8_t ns3::dot11s::IePrep::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ie-dot11s-prep.h (module 'mesh'): ns3::WifiInformationElementId ns3::dot11s::IePrep::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-prep.h (module 'mesh'): ns3::Mac48Address ns3::dot11s::IePrep::GetDestinationAddress() const [member function]
    cls.add_method('GetDestinationAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## ie-dot11s-prep.h (module 'mesh'): uint32_t ns3::dot11s::IePrep::GetDestinationSeqNumber() const [member function]
    cls.add_method('GetDestinationSeqNumber', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-prep.h (module 'mesh'): uint8_t ns3::dot11s::IePrep::GetFlags() const [member function]
    cls.add_method('GetFlags', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-prep.h (module 'mesh'): uint8_t ns3::dot11s::IePrep::GetHopcount() const [member function]
    cls.add_method('GetHopcount', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-prep.h (module 'mesh'): uint8_t ns3::dot11s::IePrep::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-prep.h (module 'mesh'): uint32_t ns3::dot11s::IePrep::GetLifetime() const [member function]
    cls.add_method('GetLifetime', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-prep.h (module 'mesh'): uint32_t ns3::dot11s::IePrep::GetMetric() const [member function]
    cls.add_method('GetMetric', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-prep.h (module 'mesh'): ns3::Mac48Address ns3::dot11s::IePrep::GetOriginatorAddress() const [member function]
    cls.add_method('GetOriginatorAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## ie-dot11s-prep.h (module 'mesh'): uint32_t ns3::dot11s::IePrep::GetOriginatorSeqNumber() const [member function]
    cls.add_method('GetOriginatorSeqNumber', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-prep.h (module 'mesh'): uint32_t ns3::dot11s::IePrep::GetTtl() const [member function]
    cls.add_method('GetTtl', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::IncrementMetric(uint32_t metric) [member function]
    cls.add_method('IncrementMetric', 
                   'void', 
                   [param('uint32_t', 'metric')])
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::SerializeInformationField(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::SetDestinationAddress(ns3::Mac48Address dest_address) [member function]
    cls.add_method('SetDestinationAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'dest_address')])
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::SetDestinationSeqNumber(uint32_t dest_seq_number) [member function]
    cls.add_method('SetDestinationSeqNumber', 
                   'void', 
                   [param('uint32_t', 'dest_seq_number')])
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::SetFlags(uint8_t flags) [member function]
    cls.add_method('SetFlags', 
                   'void', 
                   [param('uint8_t', 'flags')])
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::SetHopcount(uint8_t hopcount) [member function]
    cls.add_method('SetHopcount', 
                   'void', 
                   [param('uint8_t', 'hopcount')])
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::SetLifetime(uint32_t lifetime) [member function]
    cls.add_method('SetLifetime', 
                   'void', 
                   [param('uint32_t', 'lifetime')])
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::SetMetric(uint32_t metric) [member function]
    cls.add_method('SetMetric', 
                   'void', 
                   [param('uint32_t', 'metric')])
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::SetOriginatorAddress(ns3::Mac48Address originator_address) [member function]
    cls.add_method('SetOriginatorAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'originator_address')])
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::SetOriginatorSeqNumber(uint32_t originator_seq_number) [member function]
    cls.add_method('SetOriginatorSeqNumber', 
                   'void', 
                   [param('uint32_t', 'originator_seq_number')])
    ## ie-dot11s-prep.h (module 'mesh'): void ns3::dot11s::IePrep::SetTtl(uint8_t ttl) [member function]
    cls.add_method('SetTtl', 
                   'void', 
                   [param('uint8_t', 'ttl')])
    return

def register_Ns3Dot11sIePreq_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_output_stream_operator()
    ## ie-dot11s-preq.h (module 'mesh'): ns3::dot11s::IePreq::IePreq(ns3::dot11s::IePreq const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IePreq const &', 'arg0')])
    ## ie-dot11s-preq.h (module 'mesh'): ns3::dot11s::IePreq::IePreq() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::AddDestinationAddressElement(bool doFlag, bool rfFlag, ns3::Mac48Address dest_address, uint32_t dest_seq_number) [member function]
    cls.add_method('AddDestinationAddressElement', 
                   'void', 
                   [param('bool', 'doFlag'), param('bool', 'rfFlag'), param('ns3::Mac48Address', 'dest_address'), param('uint32_t', 'dest_seq_number')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::ClearDestinationAddressElements() [member function]
    cls.add_method('ClearDestinationAddressElements', 
                   'void', 
                   [])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::DecrementTtl() [member function]
    cls.add_method('DecrementTtl', 
                   'void', 
                   [])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::DelDestinationAddressElement(ns3::Mac48Address dest_address) [member function]
    cls.add_method('DelDestinationAddressElement', 
                   'void', 
                   [param('ns3::Mac48Address', 'dest_address')])
    ## ie-dot11s-preq.h (module 'mesh'): uint8_t ns3::dot11s::IePreq::DeserializeInformationField(ns3::Buffer::Iterator i, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'i'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ie-dot11s-preq.h (module 'mesh'): ns3::WifiInformationElementId ns3::dot11s::IePreq::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-preq.h (module 'mesh'): uint8_t ns3::dot11s::IePreq::GetDestCount() const [member function]
    cls.add_method('GetDestCount', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): std::vector<ns3::Ptr<ns3::dot11s::DestinationAddressUnit>, std::allocator<ns3::Ptr<ns3::dot11s::DestinationAddressUnit>>> ns3::dot11s::IePreq::GetDestinationList() [member function]
    cls.add_method('GetDestinationList', 
                   'std::vector< ns3::Ptr< ns3::dot11s::DestinationAddressUnit > >', 
                   [])
    ## ie-dot11s-preq.h (module 'mesh'): uint8_t ns3::dot11s::IePreq::GetHopCount() const [member function]
    cls.add_method('GetHopCount', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): uint8_t ns3::dot11s::IePreq::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-preq.h (module 'mesh'): uint32_t ns3::dot11s::IePreq::GetLifetime() const [member function]
    cls.add_method('GetLifetime', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): uint32_t ns3::dot11s::IePreq::GetMetric() const [member function]
    cls.add_method('GetMetric', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): ns3::Mac48Address ns3::dot11s::IePreq::GetOriginatorAddress() const [member function]
    cls.add_method('GetOriginatorAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): uint32_t ns3::dot11s::IePreq::GetOriginatorSeqNumber() const [member function]
    cls.add_method('GetOriginatorSeqNumber', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): uint32_t ns3::dot11s::IePreq::GetPreqID() const [member function]
    cls.add_method('GetPreqID', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): uint8_t ns3::dot11s::IePreq::GetTtl() const [member function]
    cls.add_method('GetTtl', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::IncrementMetric(uint32_t metric) [member function]
    cls.add_method('IncrementMetric', 
                   'void', 
                   [param('uint32_t', 'metric')])
    ## ie-dot11s-preq.h (module 'mesh'): bool ns3::dot11s::IePreq::IsFull() const [member function]
    cls.add_method('IsFull', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): bool ns3::dot11s::IePreq::IsNeedNotPrep() const [member function]
    cls.add_method('IsNeedNotPrep', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): bool ns3::dot11s::IePreq::IsUnicastPreq() const [member function]
    cls.add_method('IsUnicastPreq', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ie-dot11s-preq.h (module 'mesh'): bool ns3::dot11s::IePreq::MayAddAddress(ns3::Mac48Address originator) [member function]
    cls.add_method('MayAddAddress', 
                   'bool', 
                   [param('ns3::Mac48Address', 'originator')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SerializeInformationField(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SetDestCount(uint8_t dest_count) [member function]
    cls.add_method('SetDestCount', 
                   'void', 
                   [param('uint8_t', 'dest_count')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SetHopcount(uint8_t hopcount) [member function]
    cls.add_method('SetHopcount', 
                   'void', 
                   [param('uint8_t', 'hopcount')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SetLifetime(uint32_t lifetime) [member function]
    cls.add_method('SetLifetime', 
                   'void', 
                   [param('uint32_t', 'lifetime')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SetMetric(uint32_t metric) [member function]
    cls.add_method('SetMetric', 
                   'void', 
                   [param('uint32_t', 'metric')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SetNeedNotPrep() [member function]
    cls.add_method('SetNeedNotPrep', 
                   'void', 
                   [])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SetOriginatorAddress(ns3::Mac48Address originator_address) [member function]
    cls.add_method('SetOriginatorAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'originator_address')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SetOriginatorSeqNumber(uint32_t originator_seq_number) [member function]
    cls.add_method('SetOriginatorSeqNumber', 
                   'void', 
                   [param('uint32_t', 'originator_seq_number')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SetPreqID(uint32_t id) [member function]
    cls.add_method('SetPreqID', 
                   'void', 
                   [param('uint32_t', 'id')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SetTTL(uint8_t ttl) [member function]
    cls.add_method('SetTTL', 
                   'void', 
                   [param('uint8_t', 'ttl')])
    ## ie-dot11s-preq.h (module 'mesh'): void ns3::dot11s::IePreq::SetUnicastPreq() [member function]
    cls.add_method('SetUnicastPreq', 
                   'void', 
                   [])
    return

def register_Ns3Dot11sIeRann_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_output_stream_operator()
    ## ie-dot11s-rann.h (module 'mesh'): ns3::dot11s::IeRann::IeRann(ns3::dot11s::IeRann const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::IeRann const &', 'arg0')])
    ## ie-dot11s-rann.h (module 'mesh'): ns3::dot11s::IeRann::IeRann() [constructor]
    cls.add_constructor([])
    ## ie-dot11s-rann.h (module 'mesh'): void ns3::dot11s::IeRann::DecrementTtl() [member function]
    cls.add_method('DecrementTtl', 
                   'void', 
                   [])
    ## ie-dot11s-rann.h (module 'mesh'): uint8_t ns3::dot11s::IeRann::DeserializeInformationField(ns3::Buffer::Iterator start, uint8_t length) [member function]
    cls.add_method('DeserializeInformationField', 
                   'uint8_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint8_t', 'length')], 
                   is_virtual=True)
    ## ie-dot11s-rann.h (module 'mesh'): ns3::WifiInformationElementId ns3::dot11s::IeRann::ElementId() const [member function]
    cls.add_method('ElementId', 
                   'ns3::WifiInformationElementId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-rann.h (module 'mesh'): uint32_t ns3::dot11s::IeRann::GetDestSeqNumber() [member function]
    cls.add_method('GetDestSeqNumber', 
                   'uint32_t', 
                   [])
    ## ie-dot11s-rann.h (module 'mesh'): uint8_t ns3::dot11s::IeRann::GetFlags() [member function]
    cls.add_method('GetFlags', 
                   'uint8_t', 
                   [])
    ## ie-dot11s-rann.h (module 'mesh'): uint8_t ns3::dot11s::IeRann::GetHopcount() [member function]
    cls.add_method('GetHopcount', 
                   'uint8_t', 
                   [])
    ## ie-dot11s-rann.h (module 'mesh'): uint8_t ns3::dot11s::IeRann::GetInformationFieldSize() const [member function]
    cls.add_method('GetInformationFieldSize', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-rann.h (module 'mesh'): uint32_t ns3::dot11s::IeRann::GetMetric() [member function]
    cls.add_method('GetMetric', 
                   'uint32_t', 
                   [])
    ## ie-dot11s-rann.h (module 'mesh'): ns3::Mac48Address ns3::dot11s::IeRann::GetOriginatorAddress() [member function]
    cls.add_method('GetOriginatorAddress', 
                   'ns3::Mac48Address', 
                   [])
    ## ie-dot11s-rann.h (module 'mesh'): uint8_t ns3::dot11s::IeRann::GetTtl() [member function]
    cls.add_method('GetTtl', 
                   'uint8_t', 
                   [])
    ## ie-dot11s-rann.h (module 'mesh'): void ns3::dot11s::IeRann::IncrementMetric(uint32_t metric) [member function]
    cls.add_method('IncrementMetric', 
                   'void', 
                   [param('uint32_t', 'metric')])
    ## ie-dot11s-rann.h (module 'mesh'): void ns3::dot11s::IeRann::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-rann.h (module 'mesh'): void ns3::dot11s::IeRann::SerializeInformationField(ns3::Buffer::Iterator i) const [member function]
    cls.add_method('SerializeInformationField', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'i')], 
                   is_const=True, is_virtual=True)
    ## ie-dot11s-rann.h (module 'mesh'): void ns3::dot11s::IeRann::SetDestSeqNumber(uint32_t dest_seq_number) [member function]
    cls.add_method('SetDestSeqNumber', 
                   'void', 
                   [param('uint32_t', 'dest_seq_number')])
    ## ie-dot11s-rann.h (module 'mesh'): void ns3::dot11s::IeRann::SetFlags(uint8_t flags) [member function]
    cls.add_method('SetFlags', 
                   'void', 
                   [param('uint8_t', 'flags')])
    ## ie-dot11s-rann.h (module 'mesh'): void ns3::dot11s::IeRann::SetHopcount(uint8_t hopcount) [member function]
    cls.add_method('SetHopcount', 
                   'void', 
                   [param('uint8_t', 'hopcount')])
    ## ie-dot11s-rann.h (module 'mesh'): void ns3::dot11s::IeRann::SetMetric(uint32_t metric) [member function]
    cls.add_method('SetMetric', 
                   'void', 
                   [param('uint32_t', 'metric')])
    ## ie-dot11s-rann.h (module 'mesh'): void ns3::dot11s::IeRann::SetOriginatorAddress(ns3::Mac48Address originator_address) [member function]
    cls.add_method('SetOriginatorAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'originator_address')])
    ## ie-dot11s-rann.h (module 'mesh'): void ns3::dot11s::IeRann::SetTTL(uint8_t ttl) [member function]
    cls.add_method('SetTTL', 
                   'void', 
                   [param('uint8_t', 'ttl')])
    return

def register_Ns3Dot11sMeshHeader_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## dot11s-mac-header.h (module 'mesh'): ns3::dot11s::MeshHeader::MeshHeader(ns3::dot11s::MeshHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::MeshHeader const &', 'arg0')])
    ## dot11s-mac-header.h (module 'mesh'): ns3::dot11s::MeshHeader::MeshHeader() [constructor]
    cls.add_constructor([])
    ## dot11s-mac-header.h (module 'mesh'): uint32_t ns3::dot11s::MeshHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## dot11s-mac-header.h (module 'mesh'): ns3::Mac48Address ns3::dot11s::MeshHeader::GetAddr4() const [member function]
    cls.add_method('GetAddr4', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## dot11s-mac-header.h (module 'mesh'): ns3::Mac48Address ns3::dot11s::MeshHeader::GetAddr5() const [member function]
    cls.add_method('GetAddr5', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## dot11s-mac-header.h (module 'mesh'): ns3::Mac48Address ns3::dot11s::MeshHeader::GetAddr6() const [member function]
    cls.add_method('GetAddr6', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## dot11s-mac-header.h (module 'mesh'): uint8_t ns3::dot11s::MeshHeader::GetAddressExt() const [member function]
    cls.add_method('GetAddressExt', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dot11s-mac-header.h (module 'mesh'): ns3::TypeId ns3::dot11s::MeshHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## dot11s-mac-header.h (module 'mesh'): uint32_t ns3::dot11s::MeshHeader::GetMeshSeqno() const [member function]
    cls.add_method('GetMeshSeqno', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## dot11s-mac-header.h (module 'mesh'): uint8_t ns3::dot11s::MeshHeader::GetMeshTtl() const [member function]
    cls.add_method('GetMeshTtl', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dot11s-mac-header.h (module 'mesh'): uint32_t ns3::dot11s::MeshHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## dot11s-mac-header.h (module 'mesh'): static ns3::TypeId ns3::dot11s::MeshHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## dot11s-mac-header.h (module 'mesh'): void ns3::dot11s::MeshHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## dot11s-mac-header.h (module 'mesh'): void ns3::dot11s::MeshHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## dot11s-mac-header.h (module 'mesh'): void ns3::dot11s::MeshHeader::SetAddr4(ns3::Mac48Address address) [member function]
    cls.add_method('SetAddr4', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## dot11s-mac-header.h (module 'mesh'): void ns3::dot11s::MeshHeader::SetAddr5(ns3::Mac48Address address) [member function]
    cls.add_method('SetAddr5', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## dot11s-mac-header.h (module 'mesh'): void ns3::dot11s::MeshHeader::SetAddr6(ns3::Mac48Address address) [member function]
    cls.add_method('SetAddr6', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## dot11s-mac-header.h (module 'mesh'): void ns3::dot11s::MeshHeader::SetAddressExt(uint8_t num_of_addresses) [member function]
    cls.add_method('SetAddressExt', 
                   'void', 
                   [param('uint8_t', 'num_of_addresses')])
    ## dot11s-mac-header.h (module 'mesh'): void ns3::dot11s::MeshHeader::SetMeshSeqno(uint32_t seqno) [member function]
    cls.add_method('SetMeshSeqno', 
                   'void', 
                   [param('uint32_t', 'seqno')])
    ## dot11s-mac-header.h (module 'mesh'): void ns3::dot11s::MeshHeader::SetMeshTtl(uint8_t TTL) [member function]
    cls.add_method('SetMeshTtl', 
                   'void', 
                   [param('uint8_t', 'TTL')])
    return

def register_Ns3Dot11sPeerLink_methods(root_module, cls):
    ## peer-link.h (module 'mesh'): static ns3::TypeId ns3::dot11s::PeerLink::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## peer-link.h (module 'mesh'): ns3::dot11s::PeerLink::PeerLink() [constructor]
    cls.add_constructor([])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## peer-link.h (module 'mesh'): ns3::dot11s::PeerLink::PeerStateNames [variable]
    cls.add_static_attribute('PeerStateNames', 'char const * [ 6 ] const', is_const=True)
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::SetBeaconInformation(ns3::Time lastBeacon, ns3::Time BeaconInterval) [member function]
    cls.add_method('SetBeaconInformation', 
                   'void', 
                   [param('ns3::Time', 'lastBeacon'), param('ns3::Time', 'BeaconInterval')])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::SetLinkStatusCallback(ns3::Callback<void, unsigned int, ns3::Mac48Address, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> cb) [member function]
    cls.add_method('SetLinkStatusCallback', 
                   'void', 
                   [param('ns3::Callback< void, unsigned int, ns3::Mac48Address, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::SetPeerAddress(ns3::Mac48Address macaddr) [member function]
    cls.add_method('SetPeerAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'macaddr')])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::SetPeerMeshPointAddress(ns3::Mac48Address macaddr) [member function]
    cls.add_method('SetPeerMeshPointAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'macaddr')])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::SetInterface(uint32_t interface) [member function]
    cls.add_method('SetInterface', 
                   'void', 
                   [param('uint32_t', 'interface')])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::SetLocalLinkId(uint16_t id) [member function]
    cls.add_method('SetLocalLinkId', 
                   'void', 
                   [param('uint16_t', 'id')])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::SetLocalAid(uint16_t aid) [member function]
    cls.add_method('SetLocalAid', 
                   'void', 
                   [param('uint16_t', 'aid')])
    ## peer-link.h (module 'mesh'): uint16_t ns3::dot11s::PeerLink::GetPeerAid() const [member function]
    cls.add_method('GetPeerAid', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::SetBeaconTimingElement(ns3::dot11s::IeBeaconTiming beaconTiming) [member function]
    cls.add_method('SetBeaconTimingElement', 
                   'void', 
                   [param('ns3::dot11s::IeBeaconTiming', 'beaconTiming')])
    ## peer-link.h (module 'mesh'): ns3::Mac48Address ns3::dot11s::PeerLink::GetPeerAddress() const [member function]
    cls.add_method('GetPeerAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## peer-link.h (module 'mesh'): uint16_t ns3::dot11s::PeerLink::GetLocalAid() const [member function]
    cls.add_method('GetLocalAid', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## peer-link.h (module 'mesh'): ns3::Time ns3::dot11s::PeerLink::GetLastBeacon() const [member function]
    cls.add_method('GetLastBeacon', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## peer-link.h (module 'mesh'): ns3::Time ns3::dot11s::PeerLink::GetBeaconInterval() const [member function]
    cls.add_method('GetBeaconInterval', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## peer-link.h (module 'mesh'): ns3::dot11s::IeBeaconTiming ns3::dot11s::PeerLink::GetBeaconTimingElement() const [member function]
    cls.add_method('GetBeaconTimingElement', 
                   'ns3::dot11s::IeBeaconTiming', 
                   [], 
                   is_const=True)
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::MLMECancelPeerLink(ns3::dot11s::PmpReasonCode reason) [member function]
    cls.add_method('MLMECancelPeerLink', 
                   'void', 
                   [param('ns3::dot11s::PmpReasonCode', 'reason')])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::MLMEActivePeerLinkOpen() [member function]
    cls.add_method('MLMEActivePeerLinkOpen', 
                   'void', 
                   [])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::MLMEPeeringRequestReject() [member function]
    cls.add_method('MLMEPeeringRequestReject', 
                   'void', 
                   [])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::MLMESetSignalStatusCallback(ns3::dot11s::PeerLink::SignalStatusCallback cb) [member function]
    cls.add_method('MLMESetSignalStatusCallback', 
                   'void', 
                   [param('ns3::Callback< void, unsigned int, ns3::Mac48Address, ns3::Mac48Address, ns3::dot11s::PeerLink::PeerState, ns3::dot11s::PeerLink::PeerState, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::TransmissionSuccess() [member function]
    cls.add_method('TransmissionSuccess', 
                   'void', 
                   [])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::TransmissionFailure() [member function]
    cls.add_method('TransmissionFailure', 
                   'void', 
                   [])
    ## peer-link.h (module 'mesh'): void ns3::dot11s::PeerLink::Report(std::ostream & os) const [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    return

def register_Ns3Dot11sPeerLinkCloseStart_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkCloseStart::PeerLinkCloseStart() [constructor]
    cls.add_constructor([])
    ## peer-link-frame.h (module 'mesh'): uint32_t ns3::dot11s::PeerLinkCloseStart::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields ns3::dot11s::PeerLinkCloseStart::GetFields() const [member function]
    cls.add_method('GetFields', 
                   'ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields', 
                   [], 
                   is_const=True)
    ## peer-link-frame.h (module 'mesh'): ns3::TypeId ns3::dot11s::PeerLinkCloseStart::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): uint32_t ns3::dot11s::PeerLinkCloseStart::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): static ns3::TypeId ns3::dot11s::PeerLinkCloseStart::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## peer-link-frame.h (module 'mesh'): void ns3::dot11s::PeerLinkCloseStart::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): void ns3::dot11s::PeerLinkCloseStart::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): void ns3::dot11s::PeerLinkCloseStart::SetPlinkCloseStart(ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields fields) [member function]
    cls.add_method('SetPlinkCloseStart', 
                   'void', 
                   [param('ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields', 'fields')])
    return

def register_Ns3Dot11sPeerLinkCloseStartPlinkCloseStartFields_methods(root_module, cls):
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields::PlinkCloseStartFields() [constructor]
    cls.add_constructor([])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields::PlinkCloseStartFields(ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields const &', 'arg0')])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields::meshId [variable]
    cls.add_instance_attribute('meshId', 'ns3::dot11s::IeMeshId', is_const=False)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkCloseStart::PlinkCloseStartFields::protocol [variable]
    cls.add_instance_attribute('protocol', 'ns3::dot11s::IePeeringProtocol', is_const=False)
    return

def register_Ns3Dot11sPeerLinkConfirmStart_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart::PeerLinkConfirmStart() [constructor]
    cls.add_constructor([])
    ## peer-link-frame.h (module 'mesh'): uint32_t ns3::dot11s::PeerLinkConfirmStart::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields ns3::dot11s::PeerLinkConfirmStart::GetFields() const [member function]
    cls.add_method('GetFields', 
                   'ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields', 
                   [], 
                   is_const=True)
    ## peer-link-frame.h (module 'mesh'): ns3::TypeId ns3::dot11s::PeerLinkConfirmStart::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): uint32_t ns3::dot11s::PeerLinkConfirmStart::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): static ns3::TypeId ns3::dot11s::PeerLinkConfirmStart::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## peer-link-frame.h (module 'mesh'): void ns3::dot11s::PeerLinkConfirmStart::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): void ns3::dot11s::PeerLinkConfirmStart::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): void ns3::dot11s::PeerLinkConfirmStart::SetPlinkConfirmStart(ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields fields) [member function]
    cls.add_method('SetPlinkConfirmStart', 
                   'void', 
                   [param('ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields', 'fields')])
    return

def register_Ns3Dot11sPeerLinkConfirmStartPlinkConfirmStartFields_methods(root_module, cls):
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields::PlinkConfirmStartFields() [constructor]
    cls.add_constructor([])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields::PlinkConfirmStartFields(ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields const &', 'arg0')])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields::aid [variable]
    cls.add_instance_attribute('aid', 'uint16_t', is_const=False)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields::capability [variable]
    cls.add_instance_attribute('capability', 'uint16_t', is_const=False)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields::config [variable]
    cls.add_instance_attribute('config', 'ns3::dot11s::IeConfiguration', is_const=False)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields::protocol [variable]
    cls.add_instance_attribute('protocol', 'ns3::dot11s::IePeeringProtocol', is_const=False)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkConfirmStart::PlinkConfirmStartFields::rates [variable]
    cls.add_instance_attribute('rates', 'ns3::SupportedRates', is_const=False)
    return

def register_Ns3Dot11sPeerLinkOpenStart_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart::PeerLinkOpenStart() [constructor]
    cls.add_constructor([])
    ## peer-link-frame.h (module 'mesh'): uint32_t ns3::dot11s::PeerLinkOpenStart::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields ns3::dot11s::PeerLinkOpenStart::GetFields() const [member function]
    cls.add_method('GetFields', 
                   'ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields', 
                   [], 
                   is_const=True)
    ## peer-link-frame.h (module 'mesh'): ns3::TypeId ns3::dot11s::PeerLinkOpenStart::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): uint32_t ns3::dot11s::PeerLinkOpenStart::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): static ns3::TypeId ns3::dot11s::PeerLinkOpenStart::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## peer-link-frame.h (module 'mesh'): void ns3::dot11s::PeerLinkOpenStart::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): void ns3::dot11s::PeerLinkOpenStart::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## peer-link-frame.h (module 'mesh'): void ns3::dot11s::PeerLinkOpenStart::SetPlinkOpenStart(ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields fields) [member function]
    cls.add_method('SetPlinkOpenStart', 
                   'void', 
                   [param('ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields', 'fields')])
    return

def register_Ns3Dot11sPeerLinkOpenStartPlinkOpenStartFields_methods(root_module, cls):
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields::PlinkOpenStartFields() [constructor]
    cls.add_constructor([])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields::PlinkOpenStartFields(ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields const &', 'arg0')])
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields::capability [variable]
    cls.add_instance_attribute('capability', 'uint16_t', is_const=False)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields::config [variable]
    cls.add_instance_attribute('config', 'ns3::dot11s::IeConfiguration', is_const=False)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields::meshId [variable]
    cls.add_instance_attribute('meshId', 'ns3::dot11s::IeMeshId', is_const=False)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields::protocol [variable]
    cls.add_instance_attribute('protocol', 'ns3::dot11s::IePeeringProtocol', is_const=False)
    ## peer-link-frame.h (module 'mesh'): ns3::dot11s::PeerLinkOpenStart::PlinkOpenStartFields::rates [variable]
    cls.add_instance_attribute('rates', 'ns3::SupportedRates', is_const=False)
    return

def register_Ns3Dot11sPeerManagementProtocol_methods(root_module, cls):
    ## peer-management-protocol.h (module 'mesh'): ns3::dot11s::PeerManagementProtocol::PeerManagementProtocol() [constructor]
    cls.add_constructor([])
    ## peer-management-protocol.h (module 'mesh'): int64_t ns3::dot11s::PeerManagementProtocol::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::ConfigurationMismatch(uint32_t interface, ns3::Mac48Address peerAddress) [member function]
    cls.add_method('ConfigurationMismatch', 
                   'void', 
                   [param('uint32_t', 'interface'), param('ns3::Mac48Address', 'peerAddress')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## peer-management-protocol.h (module 'mesh'): ns3::Ptr<ns3::dot11s::PeerLink> ns3::dot11s::PeerManagementProtocol::FindPeerLink(uint32_t interface, ns3::Mac48Address peerAddress) [member function]
    cls.add_method('FindPeerLink', 
                   'ns3::Ptr< ns3::dot11s::PeerLink >', 
                   [param('uint32_t', 'interface'), param('ns3::Mac48Address', 'peerAddress')])
    ## peer-management-protocol.h (module 'mesh'): ns3::Mac48Address ns3::dot11s::PeerManagementProtocol::GetAddress() [member function]
    cls.add_method('GetAddress', 
                   'ns3::Mac48Address', 
                   [])
    ## peer-management-protocol.h (module 'mesh'): bool ns3::dot11s::PeerManagementProtocol::GetBeaconCollisionAvoidance() const [member function]
    cls.add_method('GetBeaconCollisionAvoidance', 
                   'bool', 
                   [], 
                   is_const=True)
    ## peer-management-protocol.h (module 'mesh'): ns3::Ptr<ns3::dot11s::IeBeaconTiming> ns3::dot11s::PeerManagementProtocol::GetBeaconTimingElement(uint32_t interface) [member function]
    cls.add_method('GetBeaconTimingElement', 
                   'ns3::Ptr< ns3::dot11s::IeBeaconTiming >', 
                   [param('uint32_t', 'interface')])
    ## peer-management-protocol.h (module 'mesh'): ns3::Ptr<ns3::dot11s::IeMeshId> ns3::dot11s::PeerManagementProtocol::GetMeshId() const [member function]
    cls.add_method('GetMeshId', 
                   'ns3::Ptr< ns3::dot11s::IeMeshId >', 
                   [], 
                   is_const=True)
    ## peer-management-protocol.h (module 'mesh'): uint8_t ns3::dot11s::PeerManagementProtocol::GetNumberOfLinks() [member function]
    cls.add_method('GetNumberOfLinks', 
                   'uint8_t', 
                   [])
    ## peer-management-protocol.h (module 'mesh'): std::vector<ns3::Ptr<ns3::dot11s::PeerLink>, std::allocator<ns3::Ptr<ns3::dot11s::PeerLink>>> ns3::dot11s::PeerManagementProtocol::GetPeerLinks() const [member function]
    cls.add_method('GetPeerLinks', 
                   'std::vector< ns3::Ptr< ns3::dot11s::PeerLink > >', 
                   [], 
                   is_const=True)
    ## peer-management-protocol.h (module 'mesh'): std::vector<ns3::Mac48Address, std::allocator<ns3::Mac48Address>> ns3::dot11s::PeerManagementProtocol::GetPeers(uint32_t interface) const [member function]
    cls.add_method('GetPeers', 
                   'std::vector< ns3::Mac48Address >', 
                   [param('uint32_t', 'interface')], 
                   is_const=True)
    ## peer-management-protocol.h (module 'mesh'): static ns3::TypeId ns3::dot11s::PeerManagementProtocol::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## peer-management-protocol.h (module 'mesh'): bool ns3::dot11s::PeerManagementProtocol::Install(ns3::Ptr<ns3::MeshPointDevice> mp) [member function]
    cls.add_method('Install', 
                   'bool', 
                   [param('ns3::Ptr< ns3::MeshPointDevice >', 'mp')])
    ## peer-management-protocol.h (module 'mesh'): bool ns3::dot11s::PeerManagementProtocol::IsActiveLink(uint32_t interface, ns3::Mac48Address peerAddress) [member function]
    cls.add_method('IsActiveLink', 
                   'bool', 
                   [param('uint32_t', 'interface'), param('ns3::Mac48Address', 'peerAddress')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::NotifyBeaconSent(uint32_t interface, ns3::Time beaconInterval) [member function]
    cls.add_method('NotifyBeaconSent', 
                   'void', 
                   [param('uint32_t', 'interface'), param('ns3::Time', 'beaconInterval')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::ReceiveBeacon(uint32_t interface, ns3::Mac48Address peerAddress, ns3::Time beaconInterval, ns3::Ptr<ns3::dot11s::IeBeaconTiming> beaconTiming) [member function]
    cls.add_method('ReceiveBeacon', 
                   'void', 
                   [param('uint32_t', 'interface'), param('ns3::Mac48Address', 'peerAddress'), param('ns3::Time', 'beaconInterval'), param('ns3::Ptr< ns3::dot11s::IeBeaconTiming >', 'beaconTiming')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::ReceivePeerLinkFrame(uint32_t interface, ns3::Mac48Address peerAddress, ns3::Mac48Address peerMeshPointAddress, uint16_t aid, ns3::dot11s::IePeerManagement peerManagementElement, ns3::dot11s::IeConfiguration meshConfig) [member function]
    cls.add_method('ReceivePeerLinkFrame', 
                   'void', 
                   [param('uint32_t', 'interface'), param('ns3::Mac48Address', 'peerAddress'), param('ns3::Mac48Address', 'peerMeshPointAddress'), param('uint16_t', 'aid'), param('ns3::dot11s::IePeerManagement', 'peerManagementElement'), param('ns3::dot11s::IeConfiguration', 'meshConfig')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::Report(std::ostream & os) const [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::ResetStats() [member function]
    cls.add_method('ResetStats', 
                   'void', 
                   [])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::SetBeaconCollisionAvoidance(bool enable) [member function]
    cls.add_method('SetBeaconCollisionAvoidance', 
                   'void', 
                   [param('bool', 'enable')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::SetMeshId(std::string s) [member function]
    cls.add_method('SetMeshId', 
                   'void', 
                   [param('std::string', 's')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::SetPeerLinkStatusCallback(ns3::Callback<void, ns3::Mac48Address, ns3::Mac48Address, unsigned int, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> cb) [member function]
    cls.add_method('SetPeerLinkStatusCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Mac48Address, ns3::Mac48Address, unsigned int, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::TransmissionFailure(uint32_t interface, ns3::Mac48Address const peerAddress) [member function]
    cls.add_method('TransmissionFailure', 
                   'void', 
                   [param('uint32_t', 'interface'), param('ns3::Mac48Address const', 'peerAddress')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::TransmissionSuccess(uint32_t interface, ns3::Mac48Address const peerAddress) [member function]
    cls.add_method('TransmissionSuccess', 
                   'void', 
                   [param('uint32_t', 'interface'), param('ns3::Mac48Address const', 'peerAddress')])
    ## peer-management-protocol.h (module 'mesh'): void ns3::dot11s::PeerManagementProtocol::DoInitialize() [member function]
    cls.add_method('DoInitialize', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3Dot11sRouteChange_methods(root_module, cls):
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::RouteChange::RouteChange() [constructor]
    cls.add_constructor([])
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::RouteChange::RouteChange(ns3::dot11s::RouteChange const & arg0) [constructor]
    cls.add_constructor([param('ns3::dot11s::RouteChange const &', 'arg0')])
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::RouteChange::destination [variable]
    cls.add_instance_attribute('destination', 'ns3::Mac48Address', is_const=False)
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::RouteChange::interface [variable]
    cls.add_instance_attribute('interface', 'uint32_t', is_const=False)
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::RouteChange::lifetime [variable]
    cls.add_instance_attribute('lifetime', 'ns3::Time', is_const=False)
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::RouteChange::metric [variable]
    cls.add_instance_attribute('metric', 'uint32_t', is_const=False)
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::RouteChange::retransmitter [variable]
    cls.add_instance_attribute('retransmitter', 'ns3::Mac48Address', is_const=False)
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::RouteChange::seqnum [variable]
    cls.add_instance_attribute('seqnum', 'uint32_t', is_const=False)
    ## hwmp-protocol.h (module 'mesh'): ns3::dot11s::RouteChange::type [variable]
    cls.add_instance_attribute('type', 'std::string', is_const=False)
    return

def register_Ns3FlameFlameHeader_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## flame-header.h (module 'mesh'): ns3::flame::FlameHeader::FlameHeader(ns3::flame::FlameHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::flame::FlameHeader const &', 'arg0')])
    ## flame-header.h (module 'mesh'): ns3::flame::FlameHeader::FlameHeader() [constructor]
    cls.add_constructor([])
    ## flame-header.h (module 'mesh'): void ns3::flame::FlameHeader::AddCost(uint8_t cost) [member function]
    cls.add_method('AddCost', 
                   'void', 
                   [param('uint8_t', 'cost')])
    ## flame-header.h (module 'mesh'): uint32_t ns3::flame::FlameHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## flame-header.h (module 'mesh'): uint8_t ns3::flame::FlameHeader::GetCost() const [member function]
    cls.add_method('GetCost', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## flame-header.h (module 'mesh'): ns3::TypeId ns3::flame::FlameHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## flame-header.h (module 'mesh'): ns3::Mac48Address ns3::flame::FlameHeader::GetOrigDst() const [member function]
    cls.add_method('GetOrigDst', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## flame-header.h (module 'mesh'): ns3::Mac48Address ns3::flame::FlameHeader::GetOrigSrc() const [member function]
    cls.add_method('GetOrigSrc', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## flame-header.h (module 'mesh'): uint16_t ns3::flame::FlameHeader::GetProtocol() const [member function]
    cls.add_method('GetProtocol', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## flame-header.h (module 'mesh'): uint16_t ns3::flame::FlameHeader::GetSeqno() const [member function]
    cls.add_method('GetSeqno', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## flame-header.h (module 'mesh'): uint32_t ns3::flame::FlameHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## flame-header.h (module 'mesh'): static ns3::TypeId ns3::flame::FlameHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## flame-header.h (module 'mesh'): void ns3::flame::FlameHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## flame-header.h (module 'mesh'): void ns3::flame::FlameHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## flame-header.h (module 'mesh'): void ns3::flame::FlameHeader::SetOrigDst(ns3::Mac48Address dst) [member function]
    cls.add_method('SetOrigDst', 
                   'void', 
                   [param('ns3::Mac48Address', 'dst')])
    ## flame-header.h (module 'mesh'): void ns3::flame::FlameHeader::SetOrigSrc(ns3::Mac48Address OrigSrc) [member function]
    cls.add_method('SetOrigSrc', 
                   'void', 
                   [param('ns3::Mac48Address', 'OrigSrc')])
    ## flame-header.h (module 'mesh'): void ns3::flame::FlameHeader::SetProtocol(uint16_t protocol) [member function]
    cls.add_method('SetProtocol', 
                   'void', 
                   [param('uint16_t', 'protocol')])
    ## flame-header.h (module 'mesh'): void ns3::flame::FlameHeader::SetSeqno(uint16_t seqno) [member function]
    cls.add_method('SetSeqno', 
                   'void', 
                   [param('uint16_t', 'seqno')])
    return

def register_Ns3FlameFlameProtocol_methods(root_module, cls):
    ## flame-protocol.h (module 'mesh'): ns3::flame::FlameProtocol::FlameProtocol() [constructor]
    cls.add_constructor([])
    ## flame-protocol.h (module 'mesh'): void ns3::flame::FlameProtocol::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## flame-protocol.h (module 'mesh'): ns3::Mac48Address ns3::flame::FlameProtocol::GetAddress() [member function]
    cls.add_method('GetAddress', 
                   'ns3::Mac48Address', 
                   [])
    ## flame-protocol.h (module 'mesh'): static ns3::TypeId ns3::flame::FlameProtocol::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## flame-protocol.h (module 'mesh'): bool ns3::flame::FlameProtocol::Install(ns3::Ptr<ns3::MeshPointDevice> mp) [member function]
    cls.add_method('Install', 
                   'bool', 
                   [param('ns3::Ptr< ns3::MeshPointDevice >', 'mp')])
    ## flame-protocol.h (module 'mesh'): bool ns3::flame::FlameProtocol::RemoveRoutingStuff(uint32_t fromIface, ns3::Mac48Address const source, ns3::Mac48Address const destination, ns3::Ptr<ns3::Packet> packet, uint16_t & protocolType) [member function]
    cls.add_method('RemoveRoutingStuff', 
                   'bool', 
                   [param('uint32_t', 'fromIface'), param('ns3::Mac48Address const', 'source'), param('ns3::Mac48Address const', 'destination'), param('ns3::Ptr< ns3::Packet >', 'packet'), param('uint16_t &', 'protocolType')], 
                   is_virtual=True)
    ## flame-protocol.h (module 'mesh'): void ns3::flame::FlameProtocol::Report(std::ostream & os) const [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## flame-protocol.h (module 'mesh'): bool ns3::flame::FlameProtocol::RequestRoute(uint32_t sourceIface, ns3::Mac48Address const source, ns3::Mac48Address const destination, ns3::Ptr<const ns3::Packet> packet, uint16_t protocolType, ns3::MeshL2RoutingProtocol::RouteReplyCallback routeReply) [member function]
    cls.add_method('RequestRoute', 
                   'bool', 
                   [param('uint32_t', 'sourceIface'), param('ns3::Mac48Address const', 'source'), param('ns3::Mac48Address const', 'destination'), param('ns3::Ptr< ns3::Packet const >', 'packet'), param('uint16_t', 'protocolType'), param('ns3::Callback< void, bool, ns3::Ptr< ns3::Packet >, ns3::Mac48Address, ns3::Mac48Address, unsigned short, unsigned int, ns3::empty, ns3::empty, ns3::empty >', 'routeReply')], 
                   is_virtual=True)
    ## flame-protocol.h (module 'mesh'): void ns3::flame::FlameProtocol::ResetStats() [member function]
    cls.add_method('ResetStats', 
                   'void', 
                   [])
    return

def register_Ns3FlameFlameProtocolMac_methods(root_module, cls):
    ## flame-protocol-mac.h (module 'mesh'): ns3::flame::FlameProtocolMac::FlameProtocolMac(ns3::flame::FlameProtocolMac const & arg0) [constructor]
    cls.add_constructor([param('ns3::flame::FlameProtocolMac const &', 'arg0')])
    ## flame-protocol-mac.h (module 'mesh'): ns3::flame::FlameProtocolMac::FlameProtocolMac(ns3::Ptr<ns3::flame::FlameProtocol> protocol) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::flame::FlameProtocol >', 'protocol')])
    ## flame-protocol-mac.h (module 'mesh'): int64_t ns3::flame::FlameProtocolMac::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True)
    ## flame-protocol-mac.h (module 'mesh'): uint16_t ns3::flame::FlameProtocolMac::GetChannelId() const [member function]
    cls.add_method('GetChannelId', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## flame-protocol-mac.h (module 'mesh'): bool ns3::flame::FlameProtocolMac::Receive(ns3::Ptr<ns3::Packet> packet, ns3::WifiMacHeader const & header) [member function]
    cls.add_method('Receive', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::WifiMacHeader const &', 'header')], 
                   is_virtual=True)
    ## flame-protocol-mac.h (module 'mesh'): void ns3::flame::FlameProtocolMac::Report(std::ostream & os) const [member function]
    cls.add_method('Report', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## flame-protocol-mac.h (module 'mesh'): void ns3::flame::FlameProtocolMac::ResetStats() [member function]
    cls.add_method('ResetStats', 
                   'void', 
                   [])
    ## flame-protocol-mac.h (module 'mesh'): void ns3::flame::FlameProtocolMac::SetParent(ns3::Ptr<ns3::MeshWifiInterfaceMac> parent) [member function]
    cls.add_method('SetParent', 
                   'void', 
                   [param('ns3::Ptr< ns3::MeshWifiInterfaceMac >', 'parent')], 
                   is_virtual=True)
    ## flame-protocol-mac.h (module 'mesh'): void ns3::flame::FlameProtocolMac::UpdateBeacon(ns3::MeshWifiBeacon & beacon) const [member function]
    cls.add_method('UpdateBeacon', 
                   'void', 
                   [param('ns3::MeshWifiBeacon &', 'beacon')], 
                   is_const=True, is_virtual=True)
    ## flame-protocol-mac.h (module 'mesh'): bool ns3::flame::FlameProtocolMac::UpdateOutcomingFrame(ns3::Ptr<ns3::Packet> packet, ns3::WifiMacHeader & header, ns3::Mac48Address from, ns3::Mac48Address to) [member function]
    cls.add_method('UpdateOutcomingFrame', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::WifiMacHeader &', 'header'), param('ns3::Mac48Address', 'from'), param('ns3::Mac48Address', 'to')], 
                   is_virtual=True)
    return

def register_Ns3FlameFlameRtable_methods(root_module, cls):
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::FlameRtable() [constructor]
    cls.add_constructor([])
    ## flame-rtable.h (module 'mesh'): void ns3::flame::FlameRtable::AddPath(ns3::Mac48Address const destination, ns3::Mac48Address const retransmitter, uint32_t const interface, uint8_t const cost, uint16_t const seqnum) [member function]
    cls.add_method('AddPath', 
                   'void', 
                   [param('ns3::Mac48Address const', 'destination'), param('ns3::Mac48Address const', 'retransmitter'), param('uint32_t const', 'interface'), param('uint8_t const', 'cost'), param('uint16_t const', 'seqnum')])
    ## flame-rtable.h (module 'mesh'): void ns3::flame::FlameRtable::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## flame-rtable.h (module 'mesh'): static ns3::TypeId ns3::flame::FlameRtable::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::LookupResult ns3::flame::FlameRtable::Lookup(ns3::Mac48Address destination) [member function]
    cls.add_method('Lookup', 
                   'ns3::flame::FlameRtable::LookupResult', 
                   [param('ns3::Mac48Address', 'destination')])
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::INTERFACE_ANY [variable]
    cls.add_static_attribute('INTERFACE_ANY', 'uint32_t const', is_const=True)
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::MAX_COST [variable]
    cls.add_static_attribute('MAX_COST', 'uint32_t const', is_const=True)
    return

def register_Ns3FlameFlameRtableLookupResult_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::LookupResult::LookupResult(ns3::flame::FlameRtable::LookupResult const & arg0) [constructor]
    cls.add_constructor([param('ns3::flame::FlameRtable::LookupResult const &', 'arg0')])
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::LookupResult::LookupResult(ns3::Mac48Address r=ns3::Mac48Address::GetBroadcast(), uint32_t i=ns3::flame::FlameRtable::INTERFACE_ANY, uint8_t c=ns3::flame::FlameRtable::MAX_COST, uint16_t s=0) [constructor]
    cls.add_constructor([param('ns3::Mac48Address', 'r', default_value='ns3::Mac48Address::GetBroadcast()'), param('uint32_t', 'i', default_value='ns3::flame::FlameRtable::INTERFACE_ANY'), param('uint8_t', 'c', default_value='ns3::flame::FlameRtable::MAX_COST'), param('uint16_t', 's', default_value='0')])
    ## flame-rtable.h (module 'mesh'): bool ns3::flame::FlameRtable::LookupResult::IsValid() const [member function]
    cls.add_method('IsValid', 
                   'bool', 
                   [], 
                   is_const=True)
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::LookupResult::cost [variable]
    cls.add_instance_attribute('cost', 'uint8_t', is_const=False)
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::LookupResult::ifIndex [variable]
    cls.add_instance_attribute('ifIndex', 'uint32_t', is_const=False)
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::LookupResult::retransmitter [variable]
    cls.add_instance_attribute('retransmitter', 'ns3::Mac48Address', is_const=False)
    ## flame-rtable.h (module 'mesh'): ns3::flame::FlameRtable::LookupResult::seqnum [variable]
    cls.add_instance_attribute('seqnum', 'uint16_t', is_const=False)
    return

def register_Ns3FlameFlameTag_methods(root_module, cls):
    ## flame-protocol.h (module 'mesh'): ns3::flame::FlameTag::FlameTag(ns3::flame::FlameTag const & arg0) [constructor]
    cls.add_constructor([param('ns3::flame::FlameTag const &', 'arg0')])
    ## flame-protocol.h (module 'mesh'): ns3::flame::FlameTag::FlameTag(ns3::Mac48Address a=ns3::Mac48Address()) [constructor]
    cls.add_constructor([param('ns3::Mac48Address', 'a', default_value='ns3::Mac48Address()')])
    ## flame-protocol.h (module 'mesh'): void ns3::flame::FlameTag::Deserialize(ns3::TagBuffer i) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_virtual=True)
    ## flame-protocol.h (module 'mesh'): ns3::TypeId ns3::flame::FlameTag::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## flame-protocol.h (module 'mesh'): uint32_t ns3::flame::FlameTag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## flame-protocol.h (module 'mesh'): static ns3::TypeId ns3::flame::FlameTag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## flame-protocol.h (module 'mesh'): void ns3::flame::FlameTag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## flame-protocol.h (module 'mesh'): void ns3::flame::FlameTag::Serialize(ns3::TagBuffer i) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_const=True, is_virtual=True)
    ## flame-protocol.h (module 'mesh'): ns3::flame::FlameTag::receiver [variable]
    cls.add_instance_attribute('receiver', 'ns3::Mac48Address', is_const=False)
    ## flame-protocol.h (module 'mesh'): ns3::flame::FlameTag::transmitter [variable]
    cls.add_instance_attribute('transmitter', 'ns3::Mac48Address', is_const=False)
    return

def register_functions(root_module):
    module = root_module
    register_functions_ns3_FatalImpl(module.add_cpp_namespace('FatalImpl'), root_module)
    register_functions_ns3_Hash(module.add_cpp_namespace('Hash'), root_module)
    register_functions_ns3_TracedValueCallback(module.add_cpp_namespace('TracedValueCallback'), root_module)
    register_functions_ns3_dot11s(module.add_cpp_namespace('dot11s'), root_module)
    register_functions_ns3_flame(module.add_cpp_namespace('flame'), root_module)
    register_functions_ns3_internal(module.add_cpp_namespace('internal'), root_module)
    return

def register_functions_ns3_FatalImpl(module, root_module):
    return

def register_functions_ns3_Hash(module, root_module):
    register_functions_ns3_Hash_Function(module.add_cpp_namespace('Function'), root_module)
    return

def register_functions_ns3_Hash_Function(module, root_module):
    return

def register_functions_ns3_TracedValueCallback(module, root_module):
    return

def register_functions_ns3_dot11s(module, root_module):
    return

def register_functions_ns3_flame(module, root_module):
    return

def register_functions_ns3_internal(module, root_module):
    return

def main():
    out = FileCodeSink(sys.stdout)
    root_module = module_init()
    register_types(root_module)
    register_methods(root_module)
    register_functions(root_module)
    root_module.generate(out)

if __name__ == '__main__':
    main()

