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
    root_module = Module('ns.wimax', cpp_namespace='::ns3')
    return root_module

def register_types(module):
    root_module = module.get_root()
    
    ## ul-job.h (module 'wimax'): ns3::ReqType [enumeration]
    module.add_enum('ReqType', ['DATA', 'UNICAST_POLLING'])
    ## log.h (module 'core'): ns3::LogLevel [enumeration]
    module.add_enum('LogLevel', ['LOG_NONE', 'LOG_ERROR', 'LOG_LEVEL_ERROR', 'LOG_WARN', 'LOG_LEVEL_WARN', 'LOG_DEBUG', 'LOG_LEVEL_DEBUG', 'LOG_INFO', 'LOG_LEVEL_INFO', 'LOG_FUNCTION', 'LOG_LEVEL_FUNCTION', 'LOG_LOGIC', 'LOG_LEVEL_LOGIC', 'LOG_ALL', 'LOG_LEVEL_ALL', 'LOG_PREFIX_FUNC', 'LOG_PREFIX_TIME', 'LOG_PREFIX_NODE', 'LOG_PREFIX_LEVEL', 'LOG_PREFIX_ALL'], import_from_module='ns.core')
    ## address.h (module 'network'): ns3::Address [class]
    module.add_class('Address', import_from_module='ns.network')
    ## address.h (module 'network'): ns3::Address::MaxSize_e [enumeration]
    module.add_enum('MaxSize_e', ['MAX_SIZE'], outer_class=root_module['ns3::Address'], import_from_module='ns.network')
    ## trace-helper.h (module 'network'): ns3::AsciiTraceHelper [class]
    module.add_class('AsciiTraceHelper', import_from_module='ns.network')
    ## trace-helper.h (module 'network'): ns3::AsciiTraceHelperForDevice [class]
    module.add_class('AsciiTraceHelperForDevice', allow_subclassing=True, import_from_module='ns.network')
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList [class]
    module.add_class('AttributeConstructionList', import_from_module='ns.core')
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::Item [struct]
    module.add_class('Item', import_from_module='ns.core', outer_class=root_module['ns3::AttributeConstructionList'])
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator', 'ns3::AttributeConstructionList::CIterator')
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator*', 'ns3::AttributeConstructionList::CIterator*')
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator&', 'ns3::AttributeConstructionList::CIterator&')
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
    ## cid.h (module 'wimax'): ns3::Cid [class]
    module.add_class('Cid')
    ## cid.h (module 'wimax'): ns3::Cid::Type [enumeration]
    module.add_enum('Type', ['BROADCAST', 'INITIAL_RANGING', 'BASIC', 'PRIMARY', 'TRANSPORT', 'MULTICAST', 'PADDING'], outer_class=root_module['ns3::Cid'])
    ## cid-factory.h (module 'wimax'): ns3::CidFactory [class]
    module.add_class('CidFactory')
    ## cs-parameters.h (module 'wimax'): ns3::CsParameters [class]
    module.add_class('CsParameters')
    ## cs-parameters.h (module 'wimax'): ns3::CsParameters::Action [enumeration]
    module.add_enum('Action', ['ADD', 'REPLACE', 'DELETE'], outer_class=root_module['ns3::CsParameters'])
    ## dl-mac-messages.h (module 'wimax'): ns3::DcdChannelEncodings [class]
    module.add_class('DcdChannelEncodings', allow_subclassing=True)
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
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::NixVector> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::NixVector'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::OutputStreamWrapper> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::OutputStreamWrapper'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::Packet> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::Packet'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::TraceSourceAccessor> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::TraceSourceAccessor'])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): ns3::DlFramePrefixIe [class]
    module.add_class('DlFramePrefixIe')
    ## event-id.h (module 'core'): ns3::EventId [class]
    module.add_class('EventId', import_from_module='ns.core')
    ## event-id.h (module 'core'): ns3::EventId::UID [enumeration]
    module.add_enum('UID', ['INVALID', 'NOW', 'DESTROY', 'RESERVED', 'VALID'], outer_class=root_module['ns3::EventId'], import_from_module='ns.core')
    ## hash.h (module 'core'): ns3::Hasher [class]
    module.add_class('Hasher', import_from_module='ns.core')
    ## ipcs-classifier-record.h (module 'wimax'): ns3::IpcsClassifierRecord [class]
    module.add_class('IpcsClassifierRecord')
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
    ## net-device-container.h (module 'network'): ns3::NetDeviceContainer [class]
    module.add_class('NetDeviceContainer', import_from_module='ns.network')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::NetDevice > > const_iterator', 'ns3::NetDeviceContainer::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::NetDevice > > const_iterator*', 'ns3::NetDeviceContainer::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::NetDevice > > const_iterator&', 'ns3::NetDeviceContainer::Iterator&')
    ## node-container.h (module 'network'): ns3::NodeContainer [class]
    module.add_class('NodeContainer', import_from_module='ns.network')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Node > > const_iterator', 'ns3::NodeContainer::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Node > > const_iterator*', 'ns3::NodeContainer::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Node > > const_iterator&', 'ns3::NodeContainer::Iterator&')
    ## object-base.h (module 'core'): ns3::ObjectBase [class]
    module.add_class('ObjectBase', allow_subclassing=True, import_from_module='ns.core')
    ## object.h (module 'core'): ns3::ObjectDeleter [struct]
    module.add_class('ObjectDeleter', import_from_module='ns.core')
    ## object-factory.h (module 'core'): ns3::ObjectFactory [class]
    module.add_class('ObjectFactory', import_from_module='ns.core')
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDcdChannelEncodings [class]
    module.add_class('OfdmDcdChannelEncodings', parent=root_module['ns3::DcdChannelEncodings'])
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDlBurstProfile [class]
    module.add_class('OfdmDlBurstProfile')
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDlBurstProfile::Diuc [enumeration]
    module.add_enum('Diuc', ['DIUC_STC_ZONE', 'DIUC_BURST_PROFILE_1', 'DIUC_BURST_PROFILE_2', 'DIUC_BURST_PROFILE_3', 'DIUC_BURST_PROFILE_4', 'DIUC_BURST_PROFILE_5', 'DIUC_BURST_PROFILE_6', 'DIUC_BURST_PROFILE_7', 'DIUC_BURST_PROFILE_8', 'DIUC_BURST_PROFILE_9', 'DIUC_BURST_PROFILE_10', 'DIUC_BURST_PROFILE_11', 'DIUC_GAP', 'DIUC_END_OF_MAP'], outer_class=root_module['ns3::OfdmDlBurstProfile'])
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDlMapIe [class]
    module.add_class('OfdmDlMapIe')
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUlBurstProfile [class]
    module.add_class('OfdmUlBurstProfile')
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUlBurstProfile::Uiuc [enumeration]
    module.add_enum('Uiuc', ['UIUC_INITIAL_RANGING', 'UIUC_REQ_REGION_FULL', 'UIUC_REQ_REGION_FOCUSED', 'UIUC_FOCUSED_CONTENTION_IE', 'UIUC_BURST_PROFILE_5', 'UIUC_BURST_PROFILE_6', 'UIUC_BURST_PROFILE_7', 'UIUC_BURST_PROFILE_8', 'UIUC_BURST_PROFILE_9', 'UIUC_BURST_PROFILE_10', 'UIUC_BURST_PROFILE_11', 'UIUC_BURST_PROFILE_12', 'UIUC_SUBCH_NETWORK_ENTRY', 'UIUC_END_OF_MAP'], outer_class=root_module['ns3::OfdmUlBurstProfile'])
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUlMapIe [class]
    module.add_class('OfdmUlMapIe')
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
    ## pcap-file.h (module 'network'): ns3::PcapFile [class]
    module.add_class('PcapFile', import_from_module='ns.network')
    ## trace-helper.h (module 'network'): ns3::PcapHelper [class]
    module.add_class('PcapHelper', import_from_module='ns.network')
    ## trace-helper.h (module 'network'): ns3::PcapHelper::DataLinkType [enumeration]
    module.add_enum('DataLinkType', ['DLT_NULL', 'DLT_EN10MB', 'DLT_PPP', 'DLT_RAW', 'DLT_IEEE802_11', 'DLT_LINUX_SLL', 'DLT_PRISM_HEADER', 'DLT_IEEE802_11_RADIO', 'DLT_IEEE802_15_4', 'DLT_NETLINK', 'DLT_LORATAP'], outer_class=root_module['ns3::PcapHelper'], import_from_module='ns.network')
    ## trace-helper.h (module 'network'): ns3::PcapHelperForDevice [class]
    module.add_class('PcapHelperForDevice', allow_subclassing=True, import_from_module='ns.network')
    ## snr-to-block-error-rate-manager.h (module 'wimax'): ns3::SNRToBlockErrorRateManager [class]
    module.add_class('SNRToBlockErrorRateManager')
    ## snr-to-block-error-rate-record.h (module 'wimax'): ns3::SNRToBlockErrorRateRecord [class]
    module.add_class('SNRToBlockErrorRateRecord')
    ## ss-record.h (module 'wimax'): ns3::SSRecord [class]
    module.add_class('SSRecord')
    ## send-params.h (module 'wimax'): ns3::SendParams [class]
    module.add_class('SendParams')
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow [class]
    module.add_class('ServiceFlow')
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::Direction [enumeration]
    module.add_enum('Direction', ['SF_DIRECTION_DOWN', 'SF_DIRECTION_UP'], outer_class=root_module['ns3::ServiceFlow'])
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::Type [enumeration]
    module.add_enum('Type', ['SF_TYPE_PROVISIONED', 'SF_TYPE_ADMITTED', 'SF_TYPE_ACTIVE'], outer_class=root_module['ns3::ServiceFlow'])
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::SchedulingType [enumeration]
    module.add_enum('SchedulingType', ['SF_TYPE_NONE', 'SF_TYPE_UNDEF', 'SF_TYPE_BE', 'SF_TYPE_NRTPS', 'SF_TYPE_RTPS', 'SF_TYPE_UGS', 'SF_TYPE_ALL'], outer_class=root_module['ns3::ServiceFlow'])
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::CsSpecification [enumeration]
    module.add_enum('CsSpecification', ['ATM', 'IPV4', 'IPV6', 'ETHERNET', 'VLAN', 'IPV4_OVER_ETHERNET', 'IPV6_OVER_ETHERNET', 'IPV4_OVER_VLAN', 'IPV6_OVER_VLAN'], outer_class=root_module['ns3::ServiceFlow'])
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::ModulationType [enumeration]
    module.add_enum('ModulationType', ['MODULATION_TYPE_BPSK_12', 'MODULATION_TYPE_QPSK_12', 'MODULATION_TYPE_QPSK_34', 'MODULATION_TYPE_QAM16_12', 'MODULATION_TYPE_QAM16_34', 'MODULATION_TYPE_QAM64_23', 'MODULATION_TYPE_QAM64_34'], outer_class=root_module['ns3::ServiceFlow'])
    ## service-flow-record.h (module 'wimax'): ns3::ServiceFlowRecord [class]
    module.add_class('ServiceFlowRecord')
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::ObjectBase'], template_parameters=['ns3::Object', 'ns3::ObjectBase', 'ns3::ObjectDeleter'])
    ## simulator.h (module 'core'): ns3::Simulator [class]
    module.add_class('Simulator', destructor_visibility='private', import_from_module='ns.core')
    ## simulator.h (module 'core'): ns3::Simulator [enumeration]
    module.add_enum('', ['NO_CONTEXT'], outer_class=root_module['ns3::Simulator'], import_from_module='ns.core')
    ## ul-job.h (module 'wimax'): ns3::SortProcess [struct]
    module.add_class('SortProcess')
    ## ul-job.h (module 'wimax'): ns3::SortProcessPtr [struct]
    module.add_class('SortProcessPtr')
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
    ## wimax-tlv.h (module 'wimax'): ns3::TlvValue [class]
    module.add_class('TlvValue', allow_subclassing=True)
    ## wimax-tlv.h (module 'wimax'): ns3::TosTlvValue [class]
    module.add_class('TosTlvValue', parent=root_module['ns3::TlvValue'])
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
    ## wimax-tlv.h (module 'wimax'): ns3::U16TlvValue [class]
    module.add_class('U16TlvValue', parent=root_module['ns3::TlvValue'])
    ## wimax-tlv.h (module 'wimax'): ns3::U32TlvValue [class]
    module.add_class('U32TlvValue', parent=root_module['ns3::TlvValue'])
    ## wimax-tlv.h (module 'wimax'): ns3::U8TlvValue [class]
    module.add_class('U8TlvValue', parent=root_module['ns3::TlvValue'])
    ## ul-mac-messages.h (module 'wimax'): ns3::UcdChannelEncodings [class]
    module.add_class('UcdChannelEncodings', allow_subclassing=True)
    ## wimax-tlv.h (module 'wimax'): ns3::VectorTlvValue [class]
    module.add_class('VectorTlvValue', parent=root_module['ns3::TlvValue'])
    typehandlers.add_type_alias('std::vector< ns3::Tlv * > const_iterator', 'ns3::VectorTlvValue::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::Tlv * > const_iterator*', 'ns3::VectorTlvValue::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::Tlv * > const_iterator&', 'ns3::VectorTlvValue::Iterator&')
    ## wimax-helper.h (module 'wimax'): ns3::WimaxHelper [class]
    module.add_class('WimaxHelper', parent=[root_module['ns3::PcapHelperForDevice'], root_module['ns3::AsciiTraceHelperForDevice']])
    ## wimax-helper.h (module 'wimax'): ns3::WimaxHelper::NetDeviceType [enumeration]
    module.add_enum('NetDeviceType', ['DEVICE_TYPE_SUBSCRIBER_STATION', 'DEVICE_TYPE_BASE_STATION'], outer_class=root_module['ns3::WimaxHelper'])
    ## wimax-helper.h (module 'wimax'): ns3::WimaxHelper::PhyType [enumeration]
    module.add_enum('PhyType', ['SIMPLE_PHY_TYPE_OFDM'], outer_class=root_module['ns3::WimaxHelper'])
    ## wimax-helper.h (module 'wimax'): ns3::WimaxHelper::SchedulerType [enumeration]
    module.add_enum('SchedulerType', ['SCHED_TYPE_SIMPLE', 'SCHED_TYPE_RTPS', 'SCHED_TYPE_MBQOS'], outer_class=root_module['ns3::WimaxHelper'])
    ## empty.h (module 'core'): ns3::empty [class]
    module.add_class('empty', import_from_module='ns.core')
    ## int64x64-128.h (module 'core'): ns3::int64x64_t [class]
    module.add_class('int64x64_t', import_from_module='ns.core')
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::impl_type [enumeration]
    module.add_enum('impl_type', ['int128_impl', 'cairo_impl', 'ld_impl'], outer_class=root_module['ns3::int64x64_t'], import_from_module='ns.core')
    ## simple-ofdm-send-param.h (module 'wimax'): ns3::simpleOfdmSendParam [class]
    module.add_class('simpleOfdmSendParam')
    ## chunk.h (module 'network'): ns3::Chunk [class]
    module.add_class('Chunk', import_from_module='ns.network', parent=root_module['ns3::ObjectBase'])
    ## wimax-tlv.h (module 'wimax'): ns3::ClassificationRuleVectorTlvValue [class]
    module.add_class('ClassificationRuleVectorTlvValue', parent=root_module['ns3::VectorTlvValue'])
    ## wimax-tlv.h (module 'wimax'): ns3::ClassificationRuleVectorTlvValue::ClassificationRuleTlvType [enumeration]
    module.add_enum('ClassificationRuleTlvType', ['Priority', 'ToS', 'Protocol', 'IP_src', 'IP_dst', 'Port_src', 'Port_dst', 'Index'], outer_class=root_module['ns3::ClassificationRuleVectorTlvValue'])
    ## wimax-tlv.h (module 'wimax'): ns3::CsParamVectorTlvValue [class]
    module.add_class('CsParamVectorTlvValue', parent=root_module['ns3::VectorTlvValue'])
    ## wimax-tlv.h (module 'wimax'): ns3::CsParamVectorTlvValue::Type [enumeration]
    module.add_enum('Type', ['Classifier_DSC_Action', 'Packet_Classification_Rule'], outer_class=root_module['ns3::CsParamVectorTlvValue'])
    ## header.h (module 'network'): ns3::Header [class]
    module.add_class('Header', import_from_module='ns.network', parent=root_module['ns3::Chunk'])
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue [class]
    module.add_class('Ipv4AddressTlvValue', parent=root_module['ns3::TlvValue'])
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue::ipv4Addr [struct]
    module.add_class('ipv4Addr', outer_class=root_module['ns3::Ipv4AddressTlvValue'])
    typehandlers.add_type_alias('std::vector< ns3::Ipv4AddressTlvValue::ipv4Addr > const_iterator', 'ns3::Ipv4AddressTlvValue::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::Ipv4AddressTlvValue::ipv4Addr > const_iterator*', 'ns3::Ipv4AddressTlvValue::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::Ipv4AddressTlvValue::ipv4Addr > const_iterator&', 'ns3::Ipv4AddressTlvValue::Iterator&')
    ## wimax-mac-header.h (module 'wimax'): ns3::MacHeaderType [class]
    module.add_class('MacHeaderType', parent=root_module['ns3::Header'])
    ## wimax-mac-header.h (module 'wimax'): ns3::MacHeaderType::HeaderType [enumeration]
    module.add_enum('HeaderType', ['HEADER_TYPE_GENERIC', 'HEADER_TYPE_BANDWIDTH'], outer_class=root_module['ns3::MacHeaderType'])
    ## mac-messages.h (module 'wimax'): ns3::ManagementMessageType [class]
    module.add_class('ManagementMessageType', parent=root_module['ns3::Header'])
    ## mac-messages.h (module 'wimax'): ns3::ManagementMessageType::MessageType [enumeration]
    module.add_enum('MessageType', ['MESSAGE_TYPE_UCD', 'MESSAGE_TYPE_DCD', 'MESSAGE_TYPE_DL_MAP', 'MESSAGE_TYPE_UL_MAP', 'MESSAGE_TYPE_RNG_REQ', 'MESSAGE_TYPE_RNG_RSP', 'MESSAGE_TYPE_REG_REQ', 'MESSAGE_TYPE_REG_RSP', 'MESSAGE_TYPE_DSA_REQ', 'MESSAGE_TYPE_DSA_RSP', 'MESSAGE_TYPE_DSA_ACK'], outer_class=root_module['ns3::ManagementMessageType'])
    ## object.h (module 'core'): ns3::Object [class]
    module.add_class('Object', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter >'])
    ## object.h (module 'core'): ns3::Object::AggregateIterator [class]
    module.add_class('AggregateIterator', import_from_module='ns.core', outer_class=root_module['ns3::Object'])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): ns3::OfdmDownlinkFramePrefix [class]
    module.add_class('OfdmDownlinkFramePrefix', parent=root_module['ns3::Header'])
    ## send-params.h (module 'wimax'): ns3::OfdmSendParams [class]
    module.add_class('OfdmSendParams', parent=root_module['ns3::SendParams'])
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUcdChannelEncodings [class]
    module.add_class('OfdmUcdChannelEncodings', parent=root_module['ns3::UcdChannelEncodings'])
    ## packet-burst.h (module 'network'): ns3::PacketBurst [class]
    module.add_class('PacketBurst', import_from_module='ns.network', parent=root_module['ns3::Object'])
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::PacketBurst const > )', 'ns3::PacketBurst::TracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::PacketBurst const > )*', 'ns3::PacketBurst::TracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::PacketBurst const > )&', 'ns3::PacketBurst::TracedCallback&')
    ## pcap-file-wrapper.h (module 'network'): ns3::PcapFileWrapper [class]
    module.add_class('PcapFileWrapper', import_from_module='ns.network', parent=root_module['ns3::Object'])
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue [class]
    module.add_class('PortRangeTlvValue', parent=root_module['ns3::TlvValue'])
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue::PortRange [struct]
    module.add_class('PortRange', outer_class=root_module['ns3::PortRangeTlvValue'])
    typehandlers.add_type_alias('std::vector< ns3::PortRangeTlvValue::PortRange > const_iterator', 'ns3::PortRangeTlvValue::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::PortRangeTlvValue::PortRange > const_iterator*', 'ns3::PortRangeTlvValue::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::PortRangeTlvValue::PortRange > const_iterator&', 'ns3::PortRangeTlvValue::Iterator&')
    ## ul-job.h (module 'wimax'): ns3::PriorityUlJob [class]
    module.add_class('PriorityUlJob', parent=root_module['ns3::Object'])
    ## propagation-loss-model.h (module 'propagation'): ns3::PropagationLossModel [class]
    module.add_class('PropagationLossModel', import_from_module='ns.propagation', parent=root_module['ns3::Object'])
    ## wimax-tlv.h (module 'wimax'): ns3::ProtocolTlvValue [class]
    module.add_class('ProtocolTlvValue', parent=root_module['ns3::TlvValue'])
    typehandlers.add_type_alias('std::vector< unsigned char > const_iterator', 'ns3::ProtocolTlvValue::Iterator')
    typehandlers.add_type_alias('std::vector< unsigned char > const_iterator*', 'ns3::ProtocolTlvValue::Iterator*')
    typehandlers.add_type_alias('std::vector< unsigned char > const_iterator&', 'ns3::ProtocolTlvValue::Iterator&')
    ## propagation-loss-model.h (module 'propagation'): ns3::RandomPropagationLossModel [class]
    module.add_class('RandomPropagationLossModel', import_from_module='ns.propagation', parent=root_module['ns3::PropagationLossModel'])
    ## random-variable-stream.h (module 'core'): ns3::RandomVariableStream [class]
    module.add_class('RandomVariableStream', import_from_module='ns.core', parent=root_module['ns3::Object'])
    ## propagation-loss-model.h (module 'propagation'): ns3::RangePropagationLossModel [class]
    module.add_class('RangePropagationLossModel', import_from_module='ns.propagation', parent=root_module['ns3::PropagationLossModel'])
    ## mac-messages.h (module 'wimax'): ns3::RngReq [class]
    module.add_class('RngReq', parent=root_module['ns3::Header'])
    ## mac-messages.h (module 'wimax'): ns3::RngRsp [class]
    module.add_class('RngRsp', parent=root_module['ns3::Header'])
    ## ss-manager.h (module 'wimax'): ns3::SSManager [class]
    module.add_class('SSManager', parent=root_module['ns3::Object'])
    ## random-variable-stream.h (module 'core'): ns3::SequentialRandomVariable [class]
    module.add_class('SequentialRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## service-flow-manager.h (module 'wimax'): ns3::ServiceFlowManager [class]
    module.add_class('ServiceFlowManager', parent=root_module['ns3::Object'])
    ## service-flow-manager.h (module 'wimax'): ns3::ServiceFlowManager::ConfirmationCode [enumeration]
    module.add_enum('ConfirmationCode', ['CONFIRMATION_CODE_SUCCESS', 'CONFIRMATION_CODE_REJECT'], outer_class=root_module['ns3::ServiceFlowManager'])
    ## wimax-tlv.h (module 'wimax'): ns3::SfVectorTlvValue [class]
    module.add_class('SfVectorTlvValue', parent=root_module['ns3::VectorTlvValue'])
    ## wimax-tlv.h (module 'wimax'): ns3::SfVectorTlvValue::Type [enumeration]
    module.add_enum('Type', ['SFID', 'CID', 'Service_Class_Name', 'reserved1', 'QoS_Parameter_Set_Type', 'Traffic_Priority', 'Maximum_Sustained_Traffic_Rate', 'Maximum_Traffic_Burst', 'Minimum_Reserved_Traffic_Rate', 'Minimum_Tolerable_Traffic_Rate', 'Service_Flow_Scheduling_Type', 'Request_Transmission_Policy', 'Tolerated_Jitter', 'Maximum_Latency', 'Fixed_length_versus_Variable_length_SDU_Indicator', 'SDU_Size', 'Target_SAID', 'ARQ_Enable', 'ARQ_WINDOW_SIZE', 'ARQ_RETRY_TIMEOUT_Transmitter_Delay', 'ARQ_RETRY_TIMEOUT_Receiver_Delay', 'ARQ_BLOCK_LIFETIME', 'ARQ_SYNC_LOSS', 'ARQ_DELIVER_IN_ORDER', 'ARQ_PURGE_TIMEOUT', 'ARQ_BLOCK_SIZE', 'reserved2', 'CS_Specification', 'IPV4_CS_Parameters'], outer_class=root_module['ns3::SfVectorTlvValue'])
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
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::NixVector', 'ns3::empty', 'ns3::DefaultDeleter<ns3::NixVector>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::OutputStreamWrapper, ns3::empty, ns3::DefaultDeleter<ns3::OutputStreamWrapper>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::OutputStreamWrapper', 'ns3::empty', 'ns3::DefaultDeleter<ns3::OutputStreamWrapper>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::Packet', 'ns3::empty', 'ns3::DefaultDeleter<ns3::Packet>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::TraceSourceAccessor', 'ns3::empty', 'ns3::DefaultDeleter<ns3::TraceSourceAccessor>'])
    ## ss-service-flow-manager.h (module 'wimax'): ns3::SsServiceFlowManager [class]
    module.add_class('SsServiceFlowManager', parent=root_module['ns3::ServiceFlowManager'])
    ## ss-service-flow-manager.h (module 'wimax'): ns3::SsServiceFlowManager::ConfirmationCode [enumeration]
    module.add_enum('ConfirmationCode', ['CONFIRMATION_CODE_SUCCESS', 'CONFIRMATION_CODE_REJECT'], outer_class=root_module['ns3::SsServiceFlowManager'])
    ## propagation-loss-model.h (module 'propagation'): ns3::ThreeLogDistancePropagationLossModel [class]
    module.add_class('ThreeLogDistancePropagationLossModel', import_from_module='ns.propagation', parent=root_module['ns3::PropagationLossModel'])
    ## wimax-tlv.h (module 'wimax'): ns3::Tlv [class]
    module.add_class('Tlv', parent=root_module['ns3::Header'])
    ## wimax-tlv.h (module 'wimax'): ns3::Tlv::CommonTypes [enumeration]
    module.add_enum('CommonTypes', ['HMAC_TUPLE', 'MAC_VERSION_ENCODING', 'CURRENT_TRANSMIT_POWER', 'DOWNLINK_SERVICE_FLOW', 'UPLINK_SERVICE_FLOW', 'VENDOR_ID_EMCODING', 'VENDOR_SPECIFIC_INFORMATION'], outer_class=root_module['ns3::Tlv'])
    ## trace-source-accessor.h (module 'core'): ns3::TraceSourceAccessor [class]
    module.add_class('TraceSourceAccessor', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor> >'])
    ## trailer.h (module 'network'): ns3::Trailer [class]
    module.add_class('Trailer', import_from_module='ns.network', parent=root_module['ns3::Chunk'])
    ## random-variable-stream.h (module 'core'): ns3::TriangularRandomVariable [class]
    module.add_class('TriangularRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## propagation-loss-model.h (module 'propagation'): ns3::TwoRayGroundPropagationLossModel [class]
    module.add_class('TwoRayGroundPropagationLossModel', import_from_module='ns.propagation', parent=root_module['ns3::PropagationLossModel'])
    ## ul-mac-messages.h (module 'wimax'): ns3::Ucd [class]
    module.add_class('Ucd', parent=root_module['ns3::Header'])
    ## ul-job.h (module 'wimax'): ns3::UlJob [class]
    module.add_class('UlJob', parent=root_module['ns3::Object'])
    ## ul-job.h (module 'wimax'): ns3::UlJob::JobPriority [enumeration]
    module.add_enum('JobPriority', ['LOW', 'INTERMEDIATE', 'HIGH'], outer_class=root_module['ns3::UlJob'])
    ## ul-mac-messages.h (module 'wimax'): ns3::UlMap [class]
    module.add_class('UlMap', parent=root_module['ns3::Header'])
    ## random-variable-stream.h (module 'core'): ns3::UniformRandomVariable [class]
    module.add_class('UniformRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## bs-uplink-scheduler.h (module 'wimax'): ns3::UplinkScheduler [class]
    module.add_class('UplinkScheduler', parent=root_module['ns3::Object'])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): ns3::UplinkSchedulerMBQoS [class]
    module.add_class('UplinkSchedulerMBQoS', parent=root_module['ns3::UplinkScheduler'])
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): ns3::UplinkSchedulerRtps [class]
    module.add_class('UplinkSchedulerRtps', parent=root_module['ns3::UplinkScheduler'])
    ## bs-uplink-scheduler-simple.h (module 'wimax'): ns3::UplinkSchedulerSimple [class]
    module.add_class('UplinkSchedulerSimple', parent=root_module['ns3::UplinkScheduler'])
    ## random-variable-stream.h (module 'core'): ns3::WeibullRandomVariable [class]
    module.add_class('WeibullRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## wimax-connection.h (module 'wimax'): ns3::WimaxConnection [class]
    module.add_class('WimaxConnection', parent=root_module['ns3::Object'])
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::Packet const > >', 'ns3::WimaxConnection::FragmentsQueue')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::Packet const > >*', 'ns3::WimaxConnection::FragmentsQueue*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::Packet const > >&', 'ns3::WimaxConnection::FragmentsQueue&')
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue [class]
    module.add_class('WimaxMacQueue', parent=root_module['ns3::Object'])
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement [struct]
    module.add_class('QueueElement', outer_class=root_module['ns3::WimaxMacQueue'])
    ## wimax-mac-to-mac-header.h (module 'wimax'): ns3::WimaxMacToMacHeader [class]
    module.add_class('WimaxMacToMacHeader', parent=root_module['ns3::Header'])
    ## wimax-phy.h (module 'wimax'): ns3::WimaxPhy [class]
    module.add_class('WimaxPhy', parent=root_module['ns3::Object'])
    ## wimax-phy.h (module 'wimax'): ns3::WimaxPhy::ModulationType [enumeration]
    module.add_enum('ModulationType', ['MODULATION_TYPE_BPSK_12', 'MODULATION_TYPE_QPSK_12', 'MODULATION_TYPE_QPSK_34', 'MODULATION_TYPE_QAM16_12', 'MODULATION_TYPE_QAM16_34', 'MODULATION_TYPE_QAM64_23', 'MODULATION_TYPE_QAM64_34'], outer_class=root_module['ns3::WimaxPhy'])
    ## wimax-phy.h (module 'wimax'): ns3::WimaxPhy::PhyState [enumeration]
    module.add_enum('PhyState', ['PHY_STATE_IDLE', 'PHY_STATE_SCANNING', 'PHY_STATE_TX', 'PHY_STATE_RX'], outer_class=root_module['ns3::WimaxPhy'])
    ## wimax-phy.h (module 'wimax'): ns3::WimaxPhy::PhyType [enumeration]
    module.add_enum('PhyType', ['SimpleWimaxPhy', 'simpleOfdmWimaxPhy'], outer_class=root_module['ns3::WimaxPhy'])
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
    ## bs-scheduler.h (module 'wimax'): ns3::BSScheduler [class]
    module.add_class('BSScheduler', parent=root_module['ns3::Object'])
    ## bs-scheduler-rtps.h (module 'wimax'): ns3::BSSchedulerRtps [class]
    module.add_class('BSSchedulerRtps', parent=root_module['ns3::BSScheduler'])
    ## bs-scheduler-simple.h (module 'wimax'): ns3::BSSchedulerSimple [class]
    module.add_class('BSSchedulerSimple', parent=root_module['ns3::BSScheduler'])
    ## wimax-mac-header.h (module 'wimax'): ns3::BandwidthRequestHeader [class]
    module.add_class('BandwidthRequestHeader', parent=root_module['ns3::Header'])
    ## wimax-mac-header.h (module 'wimax'): ns3::BandwidthRequestHeader::HeaderType [enumeration]
    module.add_enum('HeaderType', ['HEADER_TYPE_INCREMENTAL', 'HEADER_TYPE_AGGREGATE'], outer_class=root_module['ns3::BandwidthRequestHeader'])
    ## bs-service-flow-manager.h (module 'wimax'): ns3::BsServiceFlowManager [class]
    module.add_class('BsServiceFlowManager', parent=root_module['ns3::ServiceFlowManager'])
    ## bs-service-flow-manager.h (module 'wimax'): ns3::BsServiceFlowManager::ConfirmationCode [enumeration]
    module.add_enum('ConfirmationCode', ['CONFIRMATION_CODE_SUCCESS', 'CONFIRMATION_CODE_REJECT'], outer_class=root_module['ns3::BsServiceFlowManager'])
    ## callback.h (module 'core'): ns3::CallbackChecker [class]
    module.add_class('CallbackChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## callback.h (module 'core'): ns3::CallbackImplBase [class]
    module.add_class('CallbackImplBase', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter<ns3::CallbackImplBase> >'])
    ## callback.h (module 'core'): ns3::CallbackValue [class]
    module.add_class('CallbackValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## channel.h (module 'network'): ns3::Channel [class]
    module.add_class('Channel', import_from_module='ns.network', parent=root_module['ns3::Object'])
    ## connection-manager.h (module 'wimax'): ns3::ConnectionManager [class]
    module.add_class('ConnectionManager', parent=root_module['ns3::Object'])
    ## random-variable-stream.h (module 'core'): ns3::ConstantRandomVariable [class]
    module.add_class('ConstantRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## dl-mac-messages.h (module 'wimax'): ns3::Dcd [class]
    module.add_class('Dcd', parent=root_module['ns3::Header'])
    ## random-variable-stream.h (module 'core'): ns3::DeterministicRandomVariable [class]
    module.add_class('DeterministicRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## dl-mac-messages.h (module 'wimax'): ns3::DlMap [class]
    module.add_class('DlMap', parent=root_module['ns3::Header'])
    ## mac-messages.h (module 'wimax'): ns3::DsaAck [class]
    module.add_class('DsaAck', parent=root_module['ns3::Header'])
    ## mac-messages.h (module 'wimax'): ns3::DsaReq [class]
    module.add_class('DsaReq', parent=root_module['ns3::Header'])
    ## mac-messages.h (module 'wimax'): ns3::DsaRsp [class]
    module.add_class('DsaRsp', parent=root_module['ns3::Header'])
    ## random-variable-stream.h (module 'core'): ns3::EmpiricalRandomVariable [class]
    module.add_class('EmpiricalRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## attribute.h (module 'core'): ns3::EmptyAttributeAccessor [class]
    module.add_class('EmptyAttributeAccessor', import_from_module='ns.core', parent=root_module['ns3::AttributeAccessor'])
    ## attribute.h (module 'core'): ns3::EmptyAttributeChecker [class]
    module.add_class('EmptyAttributeChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## attribute.h (module 'core'): ns3::EmptyAttributeValue [class]
    module.add_class('EmptyAttributeValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## random-variable-stream.h (module 'core'): ns3::ErlangRandomVariable [class]
    module.add_class('ErlangRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## event-impl.h (module 'core'): ns3::EventImpl [class]
    module.add_class('EventImpl', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl> >'])
    ## random-variable-stream.h (module 'core'): ns3::ExponentialRandomVariable [class]
    module.add_class('ExponentialRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## propagation-loss-model.h (module 'propagation'): ns3::FixedRssLossModel [class]
    module.add_class('FixedRssLossModel', import_from_module='ns.propagation', parent=root_module['ns3::PropagationLossModel'])
    ## wimax-mac-header.h (module 'wimax'): ns3::FragmentationSubheader [class]
    module.add_class('FragmentationSubheader', parent=root_module['ns3::Header'])
    ## propagation-loss-model.h (module 'propagation'): ns3::FriisPropagationLossModel [class]
    module.add_class('FriisPropagationLossModel', import_from_module='ns.propagation', parent=root_module['ns3::PropagationLossModel'])
    ## random-variable-stream.h (module 'core'): ns3::GammaRandomVariable [class]
    module.add_class('GammaRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## wimax-mac-header.h (module 'wimax'): ns3::GenericMacHeader [class]
    module.add_class('GenericMacHeader', parent=root_module['ns3::Header'])
    ## wimax-mac-header.h (module 'wimax'): ns3::GrantManagementSubheader [class]
    module.add_class('GrantManagementSubheader', parent=root_module['ns3::Header'])
    ## ipcs-classifier.h (module 'wimax'): ns3::IpcsClassifier [class]
    module.add_class('IpcsClassifier', parent=root_module['ns3::Object'])
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
    ## propagation-loss-model.h (module 'propagation'): ns3::LogDistancePropagationLossModel [class]
    module.add_class('LogDistancePropagationLossModel', import_from_module='ns.propagation', parent=root_module['ns3::PropagationLossModel'])
    ## random-variable-stream.h (module 'core'): ns3::LogNormalRandomVariable [class]
    module.add_class('LogNormalRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## mac48-address.h (module 'network'): ns3::Mac48AddressChecker [class]
    module.add_class('Mac48AddressChecker', import_from_module='ns.network', parent=root_module['ns3::AttributeChecker'])
    ## mac48-address.h (module 'network'): ns3::Mac48AddressValue [class]
    module.add_class('Mac48AddressValue', import_from_module='ns.network', parent=root_module['ns3::AttributeValue'])
    ## propagation-loss-model.h (module 'propagation'): ns3::MatrixPropagationLossModel [class]
    module.add_class('MatrixPropagationLossModel', import_from_module='ns.propagation', parent=root_module['ns3::PropagationLossModel'])
    ## propagation-loss-model.h (module 'propagation'): ns3::NakagamiPropagationLossModel [class]
    module.add_class('NakagamiPropagationLossModel', import_from_module='ns.propagation', parent=root_module['ns3::PropagationLossModel'])
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
    ## output-stream-wrapper.h (module 'network'): ns3::OutputStreamWrapper [class]
    module.add_class('OutputStreamWrapper', import_from_module='ns.network', parent=root_module['ns3::SimpleRefCount< ns3::OutputStreamWrapper, ns3::empty, ns3::DefaultDeleter<ns3::OutputStreamWrapper> >'])
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
    ## simple-ofdm-wimax-phy.h (module 'wimax'): ns3::SimpleOfdmWimaxPhy [class]
    module.add_class('SimpleOfdmWimaxPhy', parent=root_module['ns3::WimaxPhy'])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): ns3::SimpleOfdmWimaxPhy::FrameDurationCode [enumeration]
    module.add_enum('FrameDurationCode', ['FRAME_DURATION_2_POINT_5_MS', 'FRAME_DURATION_4_MS', 'FRAME_DURATION_5_MS', 'FRAME_DURATION_8_MS', 'FRAME_DURATION_10_MS', 'FRAME_DURATION_12_POINT_5_MS', 'FRAME_DURATION_20_MS'], outer_class=root_module['ns3::SimpleOfdmWimaxPhy'])
    ## nstime.h (module 'core'): ns3::TimeValue [class]
    module.add_class('TimeValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## type-id.h (module 'core'): ns3::TypeIdChecker [class]
    module.add_class('TypeIdChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## type-id.h (module 'core'): ns3::TypeIdValue [class]
    module.add_class('TypeIdValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## uinteger.h (module 'core'): ns3::UintegerValue [class]
    module.add_class('UintegerValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## wimax-channel.h (module 'wimax'): ns3::WimaxChannel [class]
    module.add_class('WimaxChannel', parent=root_module['ns3::Channel'])
    ## wimax-net-device.h (module 'wimax'): ns3::WimaxNetDevice [class]
    module.add_class('WimaxNetDevice', parent=root_module['ns3::NetDevice'])
    ## wimax-net-device.h (module 'wimax'): ns3::WimaxNetDevice::Direction [enumeration]
    module.add_enum('Direction', ['DIRECTION_DOWNLINK', 'DIRECTION_UPLINK'], outer_class=root_module['ns3::WimaxNetDevice'])
    ## wimax-net-device.h (module 'wimax'): ns3::WimaxNetDevice::RangingStatus [enumeration]
    module.add_enum('RangingStatus', ['RANGING_STATUS_EXPIRED', 'RANGING_STATUS_CONTINUE', 'RANGING_STATUS_ABORT', 'RANGING_STATUS_SUCCESS'], outer_class=root_module['ns3::WimaxNetDevice'])
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, ns3::Mac48Address const & )', 'ns3::WimaxNetDevice::TxRxTracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, ns3::Mac48Address const & )*', 'ns3::WimaxNetDevice::TxRxTracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::Packet const >, ns3::Mac48Address const & )&', 'ns3::WimaxNetDevice::TxRxTracedCallback&')
    ## address.h (module 'network'): ns3::AddressChecker [class]
    module.add_class('AddressChecker', import_from_module='ns.network', parent=root_module['ns3::AttributeChecker'])
    ## address.h (module 'network'): ns3::AddressValue [class]
    module.add_class('AddressValue', import_from_module='ns.network', parent=root_module['ns3::AttributeValue'])
    ## bs-net-device.h (module 'wimax'): ns3::BaseStationNetDevice [class]
    module.add_class('BaseStationNetDevice', parent=root_module['ns3::WimaxNetDevice'])
    ## bs-net-device.h (module 'wimax'): ns3::BaseStationNetDevice::State [enumeration]
    module.add_enum('State', ['BS_STATE_DL_SUB_FRAME', 'BS_STATE_UL_SUB_FRAME', 'BS_STATE_TTG', 'BS_STATE_RTG'], outer_class=root_module['ns3::BaseStationNetDevice'])
    ## bs-net-device.h (module 'wimax'): ns3::BaseStationNetDevice::MacPreamble [enumeration]
    module.add_enum('MacPreamble', ['SHORT_PREAMBLE', 'LONG_PREAMBLE'], outer_class=root_module['ns3::BaseStationNetDevice'])
    ## callback.h (module 'core'): ns3::CallbackImpl<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['ns3::ObjectBase *', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'bool', 'unsigned long long', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Mac48Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<const ns3::Packet>', 'const ns3::Mac48Address &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, const ns3::Cid &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<const ns3::Packet>', 'ns3::Mac48Address', 'const ns3::Cid &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, ns3::Cid, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<const ns3::Packet>', 'ns3::Mac48Address', 'ns3::Cid', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<const ns3::Packet>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<const ns3::PacketBurst>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<ns3::NetDevice>', 'ns3::Ptr<const ns3::Packet>', 'unsigned short', 'const ns3::Address &', 'const ns3::Address &', 'ns3::NetDevice::PacketType', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<ns3::NetDevice>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<ns3::PacketBurst>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## simple-ofdm-wimax-channel.h (module 'wimax'): ns3::SimpleOfdmWimaxChannel [class]
    module.add_class('SimpleOfdmWimaxChannel', parent=root_module['ns3::WimaxChannel'])
    ## simple-ofdm-wimax-channel.h (module 'wimax'): ns3::SimpleOfdmWimaxChannel::PropModel [enumeration]
    module.add_enum('PropModel', ['RANDOM_PROPAGATION', 'FRIIS_PROPAGATION', 'LOG_DISTANCE_PROPAGATION', 'COST231_PROPAGATION'], outer_class=root_module['ns3::SimpleOfdmWimaxChannel'])
    ## ss-net-device.h (module 'wimax'): ns3::SubscriberStationNetDevice [class]
    module.add_class('SubscriberStationNetDevice', parent=root_module['ns3::WimaxNetDevice'])
    ## ss-net-device.h (module 'wimax'): ns3::SubscriberStationNetDevice::State [enumeration]
    module.add_enum('State', ['SS_STATE_IDLE', 'SS_STATE_SCANNING', 'SS_STATE_SYNCHRONIZING', 'SS_STATE_ACQUIRING_PARAMETERS', 'SS_STATE_WAITING_REG_RANG_INTRVL', 'SS_STATE_WAITING_INV_RANG_INTRVL', 'SS_STATE_WAITING_RNG_RSP', 'SS_STATE_ADJUSTING_PARAMETERS', 'SS_STATE_REGISTERED', 'SS_STATE_TRANSMITTING', 'SS_STATE_STOPPED'], outer_class=root_module['ns3::SubscriberStationNetDevice'])
    ## ss-net-device.h (module 'wimax'): ns3::SubscriberStationNetDevice::EventType [enumeration]
    module.add_enum('EventType', ['EVENT_NONE', 'EVENT_WAIT_FOR_RNG_RSP', 'EVENT_DL_MAP_SYNC_TIMEOUT', 'EVENT_LOST_DL_MAP', 'EVENT_LOST_UL_MAP', 'EVENT_DCD_WAIT_TIMEOUT', 'EVENT_UCD_WAIT_TIMEOUT', 'EVENT_RANG_OPP_WAIT_TIMEOUT'], outer_class=root_module['ns3::SubscriberStationNetDevice'])
    typehandlers.add_type_alias('ns3::Callback< void, std::basic_string< char >, ns3::Ptr< ns3::Packet const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::SubscriberStationNetDevice::AsciiTraceCallback')
    typehandlers.add_type_alias('ns3::Callback< void, std::basic_string< char >, ns3::Ptr< ns3::Packet const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::SubscriberStationNetDevice::AsciiTraceCallback*')
    typehandlers.add_type_alias('ns3::Callback< void, std::basic_string< char >, ns3::Ptr< ns3::Packet const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::SubscriberStationNetDevice::AsciiTraceCallback&')
    module.add_container('std::map< std::string, ns3::LogComponent * >', ('std::basic_string< char, std::char_traits< char >, std::allocator< char > >', 'ns3::LogComponent *'), container_type='map')
    module.add_container('std::vector< ns3::ServiceFlow * >', 'ns3::ServiceFlow *', container_type='vector')
    module.add_container('std::vector< bool >', 'bool', container_type='vector')
    module.add_container('ns3::bvec', 'bool', container_type='vector')
    module.add_container('std::vector< ns3::DlFramePrefixIe >', 'ns3::DlFramePrefixIe', container_type='vector')
    module.add_container('std::list< ns3::Ptr< ns3::Packet > >', 'ns3::Ptr< ns3::Packet >', container_type='list')
    module.add_container('std::vector< ns3::SSRecord * >', 'ns3::SSRecord *', container_type='vector')
    module.add_container('std::vector< ns3::OfdmUlBurstProfile >', 'ns3::OfdmUlBurstProfile', container_type='vector')
    module.add_container('std::list< ns3::OfdmUlMapIe >', 'ns3::OfdmUlMapIe', container_type='list')
    module.add_container('std::list< ns3::Ptr< ns3::UlJob > >', 'ns3::Ptr< ns3::UlJob >', container_type='list')
    module.add_container('std::list< ns3::Ptr< ns3::Packet const > >', 'ns3::Ptr< ns3::Packet const >', container_type='list')
    module.add_container('std::deque< ns3::WimaxMacQueue::QueueElement >', 'ns3::WimaxMacQueue::QueueElement', container_type='dequeue')
    module.add_container('std::list< std::pair< ns3::OfdmDlMapIe *, ns3::Ptr< ns3::PacketBurst > > >', 'std::pair< ns3::OfdmDlMapIe *, ns3::Ptr< ns3::PacketBurst > >', container_type='list')
    module.add_container('std::vector< ns3::Ptr< ns3::WimaxConnection > >', 'ns3::Ptr< ns3::WimaxConnection >', container_type='vector')
    module.add_container('std::vector< ns3::OfdmDlBurstProfile >', 'ns3::OfdmDlBurstProfile', container_type='vector')
    module.add_container('std::list< ns3::OfdmDlMapIe >', 'ns3::OfdmDlMapIe', container_type='list')
    typehandlers.add_type_alias('std::vector< bool >', 'ns3::bvec')
    typehandlers.add_type_alias('std::vector< bool >*', 'ns3::bvec*')
    typehandlers.add_type_alias('std::vector< bool >&', 'ns3::bvec&')
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

def register_types_ns3_internal(module):
    root_module = module.get_root()
    

def register_methods(root_module):
    register_Ns3Address_methods(root_module, root_module['ns3::Address'])
    register_Ns3AsciiTraceHelper_methods(root_module, root_module['ns3::AsciiTraceHelper'])
    register_Ns3AsciiTraceHelperForDevice_methods(root_module, root_module['ns3::AsciiTraceHelperForDevice'])
    register_Ns3AttributeConstructionList_methods(root_module, root_module['ns3::AttributeConstructionList'])
    register_Ns3AttributeConstructionListItem_methods(root_module, root_module['ns3::AttributeConstructionList::Item'])
    register_Ns3Buffer_methods(root_module, root_module['ns3::Buffer'])
    register_Ns3BufferIterator_methods(root_module, root_module['ns3::Buffer::Iterator'])
    register_Ns3ByteTagIterator_methods(root_module, root_module['ns3::ByteTagIterator'])
    register_Ns3ByteTagIteratorItem_methods(root_module, root_module['ns3::ByteTagIterator::Item'])
    register_Ns3ByteTagList_methods(root_module, root_module['ns3::ByteTagList'])
    register_Ns3ByteTagListIterator_methods(root_module, root_module['ns3::ByteTagList::Iterator'])
    register_Ns3ByteTagListIteratorItem_methods(root_module, root_module['ns3::ByteTagList::Iterator::Item'])
    register_Ns3CallbackBase_methods(root_module, root_module['ns3::CallbackBase'])
    register_Ns3Cid_methods(root_module, root_module['ns3::Cid'])
    register_Ns3CidFactory_methods(root_module, root_module['ns3::CidFactory'])
    register_Ns3CsParameters_methods(root_module, root_module['ns3::CsParameters'])
    register_Ns3DcdChannelEncodings_methods(root_module, root_module['ns3::DcdChannelEncodings'])
    register_Ns3DefaultDeleter__Ns3AttributeAccessor_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeAccessor >'])
    register_Ns3DefaultDeleter__Ns3AttributeChecker_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeChecker >'])
    register_Ns3DefaultDeleter__Ns3AttributeValue_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeValue >'])
    register_Ns3DefaultDeleter__Ns3CallbackImplBase_methods(root_module, root_module['ns3::DefaultDeleter< ns3::CallbackImplBase >'])
    register_Ns3DefaultDeleter__Ns3EventImpl_methods(root_module, root_module['ns3::DefaultDeleter< ns3::EventImpl >'])
    register_Ns3DefaultDeleter__Ns3HashImplementation_methods(root_module, root_module['ns3::DefaultDeleter< ns3::Hash::Implementation >'])
    register_Ns3DefaultDeleter__Ns3NixVector_methods(root_module, root_module['ns3::DefaultDeleter< ns3::NixVector >'])
    register_Ns3DefaultDeleter__Ns3OutputStreamWrapper_methods(root_module, root_module['ns3::DefaultDeleter< ns3::OutputStreamWrapper >'])
    register_Ns3DefaultDeleter__Ns3Packet_methods(root_module, root_module['ns3::DefaultDeleter< ns3::Packet >'])
    register_Ns3DefaultDeleter__Ns3TraceSourceAccessor_methods(root_module, root_module['ns3::DefaultDeleter< ns3::TraceSourceAccessor >'])
    register_Ns3DlFramePrefixIe_methods(root_module, root_module['ns3::DlFramePrefixIe'])
    register_Ns3EventId_methods(root_module, root_module['ns3::EventId'])
    register_Ns3Hasher_methods(root_module, root_module['ns3::Hasher'])
    register_Ns3IpcsClassifierRecord_methods(root_module, root_module['ns3::IpcsClassifierRecord'])
    register_Ns3Ipv4Address_methods(root_module, root_module['ns3::Ipv4Address'])
    register_Ns3Ipv4AddressHash_methods(root_module, root_module['ns3::Ipv4AddressHash'])
    register_Ns3Ipv4Mask_methods(root_module, root_module['ns3::Ipv4Mask'])
    register_Ns3Ipv6Address_methods(root_module, root_module['ns3::Ipv6Address'])
    register_Ns3Ipv6AddressHash_methods(root_module, root_module['ns3::Ipv6AddressHash'])
    register_Ns3Ipv6Prefix_methods(root_module, root_module['ns3::Ipv6Prefix'])
    register_Ns3LogComponent_methods(root_module, root_module['ns3::LogComponent'])
    register_Ns3Mac48Address_methods(root_module, root_module['ns3::Mac48Address'])
    register_Ns3Mac8Address_methods(root_module, root_module['ns3::Mac8Address'])
    register_Ns3NetDeviceContainer_methods(root_module, root_module['ns3::NetDeviceContainer'])
    register_Ns3NodeContainer_methods(root_module, root_module['ns3::NodeContainer'])
    register_Ns3ObjectBase_methods(root_module, root_module['ns3::ObjectBase'])
    register_Ns3ObjectDeleter_methods(root_module, root_module['ns3::ObjectDeleter'])
    register_Ns3ObjectFactory_methods(root_module, root_module['ns3::ObjectFactory'])
    register_Ns3OfdmDcdChannelEncodings_methods(root_module, root_module['ns3::OfdmDcdChannelEncodings'])
    register_Ns3OfdmDlBurstProfile_methods(root_module, root_module['ns3::OfdmDlBurstProfile'])
    register_Ns3OfdmDlMapIe_methods(root_module, root_module['ns3::OfdmDlMapIe'])
    register_Ns3OfdmUlBurstProfile_methods(root_module, root_module['ns3::OfdmUlBurstProfile'])
    register_Ns3OfdmUlMapIe_methods(root_module, root_module['ns3::OfdmUlMapIe'])
    register_Ns3PacketMetadata_methods(root_module, root_module['ns3::PacketMetadata'])
    register_Ns3PacketMetadataItem_methods(root_module, root_module['ns3::PacketMetadata::Item'])
    register_Ns3PacketMetadataItemIterator_methods(root_module, root_module['ns3::PacketMetadata::ItemIterator'])
    register_Ns3PacketTagIterator_methods(root_module, root_module['ns3::PacketTagIterator'])
    register_Ns3PacketTagIteratorItem_methods(root_module, root_module['ns3::PacketTagIterator::Item'])
    register_Ns3PacketTagList_methods(root_module, root_module['ns3::PacketTagList'])
    register_Ns3PacketTagListTagData_methods(root_module, root_module['ns3::PacketTagList::TagData'])
    register_Ns3ParameterLogger_methods(root_module, root_module['ns3::ParameterLogger'])
    register_Ns3PcapFile_methods(root_module, root_module['ns3::PcapFile'])
    register_Ns3PcapHelper_methods(root_module, root_module['ns3::PcapHelper'])
    register_Ns3PcapHelperForDevice_methods(root_module, root_module['ns3::PcapHelperForDevice'])
    register_Ns3SNRToBlockErrorRateManager_methods(root_module, root_module['ns3::SNRToBlockErrorRateManager'])
    register_Ns3SNRToBlockErrorRateRecord_methods(root_module, root_module['ns3::SNRToBlockErrorRateRecord'])
    register_Ns3SSRecord_methods(root_module, root_module['ns3::SSRecord'])
    register_Ns3SendParams_methods(root_module, root_module['ns3::SendParams'])
    register_Ns3ServiceFlow_methods(root_module, root_module['ns3::ServiceFlow'])
    register_Ns3ServiceFlowRecord_methods(root_module, root_module['ns3::ServiceFlowRecord'])
    register_Ns3SimpleRefCount__Ns3Object_Ns3ObjectBase_Ns3ObjectDeleter_methods(root_module, root_module['ns3::SimpleRefCount< ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter >'])
    register_Ns3Simulator_methods(root_module, root_module['ns3::Simulator'])
    register_Ns3SortProcess_methods(root_module, root_module['ns3::SortProcess'])
    register_Ns3SortProcessPtr_methods(root_module, root_module['ns3::SortProcessPtr'])
    register_Ns3Tag_methods(root_module, root_module['ns3::Tag'])
    register_Ns3TagBuffer_methods(root_module, root_module['ns3::TagBuffer'])
    register_Ns3Time_methods(root_module, root_module['ns3::Time'])
    register_Ns3TimeWithUnit_methods(root_module, root_module['ns3::TimeWithUnit'])
    register_Ns3TlvValue_methods(root_module, root_module['ns3::TlvValue'])
    register_Ns3TosTlvValue_methods(root_module, root_module['ns3::TosTlvValue'])
    register_Ns3TypeId_methods(root_module, root_module['ns3::TypeId'])
    register_Ns3TypeIdAttributeInformation_methods(root_module, root_module['ns3::TypeId::AttributeInformation'])
    register_Ns3TypeIdTraceSourceInformation_methods(root_module, root_module['ns3::TypeId::TraceSourceInformation'])
    register_Ns3U16TlvValue_methods(root_module, root_module['ns3::U16TlvValue'])
    register_Ns3U32TlvValue_methods(root_module, root_module['ns3::U32TlvValue'])
    register_Ns3U8TlvValue_methods(root_module, root_module['ns3::U8TlvValue'])
    register_Ns3UcdChannelEncodings_methods(root_module, root_module['ns3::UcdChannelEncodings'])
    register_Ns3VectorTlvValue_methods(root_module, root_module['ns3::VectorTlvValue'])
    register_Ns3WimaxHelper_methods(root_module, root_module['ns3::WimaxHelper'])
    register_Ns3Empty_methods(root_module, root_module['ns3::empty'])
    register_Ns3Int64x64_t_methods(root_module, root_module['ns3::int64x64_t'])
    register_Ns3SimpleOfdmSendParam_methods(root_module, root_module['ns3::simpleOfdmSendParam'])
    register_Ns3Chunk_methods(root_module, root_module['ns3::Chunk'])
    register_Ns3ClassificationRuleVectorTlvValue_methods(root_module, root_module['ns3::ClassificationRuleVectorTlvValue'])
    register_Ns3CsParamVectorTlvValue_methods(root_module, root_module['ns3::CsParamVectorTlvValue'])
    register_Ns3Header_methods(root_module, root_module['ns3::Header'])
    register_Ns3Ipv4AddressTlvValue_methods(root_module, root_module['ns3::Ipv4AddressTlvValue'])
    register_Ns3Ipv4AddressTlvValueIpv4Addr_methods(root_module, root_module['ns3::Ipv4AddressTlvValue::ipv4Addr'])
    register_Ns3MacHeaderType_methods(root_module, root_module['ns3::MacHeaderType'])
    register_Ns3ManagementMessageType_methods(root_module, root_module['ns3::ManagementMessageType'])
    register_Ns3Object_methods(root_module, root_module['ns3::Object'])
    register_Ns3ObjectAggregateIterator_methods(root_module, root_module['ns3::Object::AggregateIterator'])
    register_Ns3OfdmDownlinkFramePrefix_methods(root_module, root_module['ns3::OfdmDownlinkFramePrefix'])
    register_Ns3OfdmSendParams_methods(root_module, root_module['ns3::OfdmSendParams'])
    register_Ns3OfdmUcdChannelEncodings_methods(root_module, root_module['ns3::OfdmUcdChannelEncodings'])
    register_Ns3PacketBurst_methods(root_module, root_module['ns3::PacketBurst'])
    register_Ns3PcapFileWrapper_methods(root_module, root_module['ns3::PcapFileWrapper'])
    register_Ns3PortRangeTlvValue_methods(root_module, root_module['ns3::PortRangeTlvValue'])
    register_Ns3PortRangeTlvValuePortRange_methods(root_module, root_module['ns3::PortRangeTlvValue::PortRange'])
    register_Ns3PriorityUlJob_methods(root_module, root_module['ns3::PriorityUlJob'])
    register_Ns3PropagationLossModel_methods(root_module, root_module['ns3::PropagationLossModel'])
    register_Ns3ProtocolTlvValue_methods(root_module, root_module['ns3::ProtocolTlvValue'])
    register_Ns3RandomPropagationLossModel_methods(root_module, root_module['ns3::RandomPropagationLossModel'])
    register_Ns3RandomVariableStream_methods(root_module, root_module['ns3::RandomVariableStream'])
    register_Ns3RangePropagationLossModel_methods(root_module, root_module['ns3::RangePropagationLossModel'])
    register_Ns3RngReq_methods(root_module, root_module['ns3::RngReq'])
    register_Ns3RngRsp_methods(root_module, root_module['ns3::RngRsp'])
    register_Ns3SSManager_methods(root_module, root_module['ns3::SSManager'])
    register_Ns3SequentialRandomVariable_methods(root_module, root_module['ns3::SequentialRandomVariable'])
    register_Ns3ServiceFlowManager_methods(root_module, root_module['ns3::ServiceFlowManager'])
    register_Ns3SfVectorTlvValue_methods(root_module, root_module['ns3::SfVectorTlvValue'])
    register_Ns3SimpleRefCount__Ns3AttributeAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeAccessor__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeAccessor, ns3::empty, ns3::DefaultDeleter<ns3::AttributeAccessor> >'])
    register_Ns3SimpleRefCount__Ns3AttributeChecker_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeChecker__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeChecker, ns3::empty, ns3::DefaultDeleter<ns3::AttributeChecker> >'])
    register_Ns3SimpleRefCount__Ns3AttributeValue_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeValue__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeValue, ns3::empty, ns3::DefaultDeleter<ns3::AttributeValue> >'])
    register_Ns3SimpleRefCount__Ns3CallbackImplBase_Ns3Empty_Ns3DefaultDeleter__lt__ns3CallbackImplBase__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter<ns3::CallbackImplBase> >'])
    register_Ns3SimpleRefCount__Ns3EventImpl_Ns3Empty_Ns3DefaultDeleter__lt__ns3EventImpl__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl> >'])
    register_Ns3SimpleRefCount__Ns3HashImplementation_Ns3Empty_Ns3DefaultDeleter__lt__ns3HashImplementation__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::Hash::Implementation, ns3::empty, ns3::DefaultDeleter<ns3::Hash::Implementation> >'])
    register_Ns3SimpleRefCount__Ns3NixVector_Ns3Empty_Ns3DefaultDeleter__lt__ns3NixVector__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector> >'])
    register_Ns3SimpleRefCount__Ns3OutputStreamWrapper_Ns3Empty_Ns3DefaultDeleter__lt__ns3OutputStreamWrapper__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::OutputStreamWrapper, ns3::empty, ns3::DefaultDeleter<ns3::OutputStreamWrapper> >'])
    register_Ns3SimpleRefCount__Ns3Packet_Ns3Empty_Ns3DefaultDeleter__lt__ns3Packet__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet> >'])
    register_Ns3SimpleRefCount__Ns3TraceSourceAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3TraceSourceAccessor__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor> >'])
    register_Ns3SsServiceFlowManager_methods(root_module, root_module['ns3::SsServiceFlowManager'])
    register_Ns3ThreeLogDistancePropagationLossModel_methods(root_module, root_module['ns3::ThreeLogDistancePropagationLossModel'])
    register_Ns3Tlv_methods(root_module, root_module['ns3::Tlv'])
    register_Ns3TraceSourceAccessor_methods(root_module, root_module['ns3::TraceSourceAccessor'])
    register_Ns3Trailer_methods(root_module, root_module['ns3::Trailer'])
    register_Ns3TriangularRandomVariable_methods(root_module, root_module['ns3::TriangularRandomVariable'])
    register_Ns3TwoRayGroundPropagationLossModel_methods(root_module, root_module['ns3::TwoRayGroundPropagationLossModel'])
    register_Ns3Ucd_methods(root_module, root_module['ns3::Ucd'])
    register_Ns3UlJob_methods(root_module, root_module['ns3::UlJob'])
    register_Ns3UlMap_methods(root_module, root_module['ns3::UlMap'])
    register_Ns3UniformRandomVariable_methods(root_module, root_module['ns3::UniformRandomVariable'])
    register_Ns3UplinkScheduler_methods(root_module, root_module['ns3::UplinkScheduler'])
    register_Ns3UplinkSchedulerMBQoS_methods(root_module, root_module['ns3::UplinkSchedulerMBQoS'])
    register_Ns3UplinkSchedulerRtps_methods(root_module, root_module['ns3::UplinkSchedulerRtps'])
    register_Ns3UplinkSchedulerSimple_methods(root_module, root_module['ns3::UplinkSchedulerSimple'])
    register_Ns3WeibullRandomVariable_methods(root_module, root_module['ns3::WeibullRandomVariable'])
    register_Ns3WimaxConnection_methods(root_module, root_module['ns3::WimaxConnection'])
    register_Ns3WimaxMacQueue_methods(root_module, root_module['ns3::WimaxMacQueue'])
    register_Ns3WimaxMacQueueQueueElement_methods(root_module, root_module['ns3::WimaxMacQueue::QueueElement'])
    register_Ns3WimaxMacToMacHeader_methods(root_module, root_module['ns3::WimaxMacToMacHeader'])
    register_Ns3WimaxPhy_methods(root_module, root_module['ns3::WimaxPhy'])
    register_Ns3ZetaRandomVariable_methods(root_module, root_module['ns3::ZetaRandomVariable'])
    register_Ns3ZipfRandomVariable_methods(root_module, root_module['ns3::ZipfRandomVariable'])
    register_Ns3AttributeAccessor_methods(root_module, root_module['ns3::AttributeAccessor'])
    register_Ns3AttributeChecker_methods(root_module, root_module['ns3::AttributeChecker'])
    register_Ns3AttributeValue_methods(root_module, root_module['ns3::AttributeValue'])
    register_Ns3BSScheduler_methods(root_module, root_module['ns3::BSScheduler'])
    register_Ns3BSSchedulerRtps_methods(root_module, root_module['ns3::BSSchedulerRtps'])
    register_Ns3BSSchedulerSimple_methods(root_module, root_module['ns3::BSSchedulerSimple'])
    register_Ns3BandwidthRequestHeader_methods(root_module, root_module['ns3::BandwidthRequestHeader'])
    register_Ns3BsServiceFlowManager_methods(root_module, root_module['ns3::BsServiceFlowManager'])
    register_Ns3CallbackChecker_methods(root_module, root_module['ns3::CallbackChecker'])
    register_Ns3CallbackImplBase_methods(root_module, root_module['ns3::CallbackImplBase'])
    register_Ns3CallbackValue_methods(root_module, root_module['ns3::CallbackValue'])
    register_Ns3Channel_methods(root_module, root_module['ns3::Channel'])
    register_Ns3ConnectionManager_methods(root_module, root_module['ns3::ConnectionManager'])
    register_Ns3ConstantRandomVariable_methods(root_module, root_module['ns3::ConstantRandomVariable'])
    register_Ns3Dcd_methods(root_module, root_module['ns3::Dcd'])
    register_Ns3DeterministicRandomVariable_methods(root_module, root_module['ns3::DeterministicRandomVariable'])
    register_Ns3DlMap_methods(root_module, root_module['ns3::DlMap'])
    register_Ns3DsaAck_methods(root_module, root_module['ns3::DsaAck'])
    register_Ns3DsaReq_methods(root_module, root_module['ns3::DsaReq'])
    register_Ns3DsaRsp_methods(root_module, root_module['ns3::DsaRsp'])
    register_Ns3EmpiricalRandomVariable_methods(root_module, root_module['ns3::EmpiricalRandomVariable'])
    register_Ns3EmptyAttributeAccessor_methods(root_module, root_module['ns3::EmptyAttributeAccessor'])
    register_Ns3EmptyAttributeChecker_methods(root_module, root_module['ns3::EmptyAttributeChecker'])
    register_Ns3EmptyAttributeValue_methods(root_module, root_module['ns3::EmptyAttributeValue'])
    register_Ns3ErlangRandomVariable_methods(root_module, root_module['ns3::ErlangRandomVariable'])
    register_Ns3EventImpl_methods(root_module, root_module['ns3::EventImpl'])
    register_Ns3ExponentialRandomVariable_methods(root_module, root_module['ns3::ExponentialRandomVariable'])
    register_Ns3FixedRssLossModel_methods(root_module, root_module['ns3::FixedRssLossModel'])
    register_Ns3FragmentationSubheader_methods(root_module, root_module['ns3::FragmentationSubheader'])
    register_Ns3FriisPropagationLossModel_methods(root_module, root_module['ns3::FriisPropagationLossModel'])
    register_Ns3GammaRandomVariable_methods(root_module, root_module['ns3::GammaRandomVariable'])
    register_Ns3GenericMacHeader_methods(root_module, root_module['ns3::GenericMacHeader'])
    register_Ns3GrantManagementSubheader_methods(root_module, root_module['ns3::GrantManagementSubheader'])
    register_Ns3IpcsClassifier_methods(root_module, root_module['ns3::IpcsClassifier'])
    register_Ns3Ipv4AddressChecker_methods(root_module, root_module['ns3::Ipv4AddressChecker'])
    register_Ns3Ipv4AddressValue_methods(root_module, root_module['ns3::Ipv4AddressValue'])
    register_Ns3Ipv4MaskChecker_methods(root_module, root_module['ns3::Ipv4MaskChecker'])
    register_Ns3Ipv4MaskValue_methods(root_module, root_module['ns3::Ipv4MaskValue'])
    register_Ns3Ipv6AddressChecker_methods(root_module, root_module['ns3::Ipv6AddressChecker'])
    register_Ns3Ipv6AddressValue_methods(root_module, root_module['ns3::Ipv6AddressValue'])
    register_Ns3Ipv6PrefixChecker_methods(root_module, root_module['ns3::Ipv6PrefixChecker'])
    register_Ns3Ipv6PrefixValue_methods(root_module, root_module['ns3::Ipv6PrefixValue'])
    register_Ns3LogDistancePropagationLossModel_methods(root_module, root_module['ns3::LogDistancePropagationLossModel'])
    register_Ns3LogNormalRandomVariable_methods(root_module, root_module['ns3::LogNormalRandomVariable'])
    register_Ns3Mac48AddressChecker_methods(root_module, root_module['ns3::Mac48AddressChecker'])
    register_Ns3Mac48AddressValue_methods(root_module, root_module['ns3::Mac48AddressValue'])
    register_Ns3MatrixPropagationLossModel_methods(root_module, root_module['ns3::MatrixPropagationLossModel'])
    register_Ns3NakagamiPropagationLossModel_methods(root_module, root_module['ns3::NakagamiPropagationLossModel'])
    register_Ns3NetDevice_methods(root_module, root_module['ns3::NetDevice'])
    register_Ns3NixVector_methods(root_module, root_module['ns3::NixVector'])
    register_Ns3Node_methods(root_module, root_module['ns3::Node'])
    register_Ns3NormalRandomVariable_methods(root_module, root_module['ns3::NormalRandomVariable'])
    register_Ns3ObjectFactoryChecker_methods(root_module, root_module['ns3::ObjectFactoryChecker'])
    register_Ns3ObjectFactoryValue_methods(root_module, root_module['ns3::ObjectFactoryValue'])
    register_Ns3OutputStreamWrapper_methods(root_module, root_module['ns3::OutputStreamWrapper'])
    register_Ns3Packet_methods(root_module, root_module['ns3::Packet'])
    register_Ns3ParetoRandomVariable_methods(root_module, root_module['ns3::ParetoRandomVariable'])
    register_Ns3SimpleOfdmWimaxPhy_methods(root_module, root_module['ns3::SimpleOfdmWimaxPhy'])
    register_Ns3TimeValue_methods(root_module, root_module['ns3::TimeValue'])
    register_Ns3TypeIdChecker_methods(root_module, root_module['ns3::TypeIdChecker'])
    register_Ns3TypeIdValue_methods(root_module, root_module['ns3::TypeIdValue'])
    register_Ns3UintegerValue_methods(root_module, root_module['ns3::UintegerValue'])
    register_Ns3WimaxChannel_methods(root_module, root_module['ns3::WimaxChannel'])
    register_Ns3WimaxNetDevice_methods(root_module, root_module['ns3::WimaxNetDevice'])
    register_Ns3AddressChecker_methods(root_module, root_module['ns3::AddressChecker'])
    register_Ns3AddressValue_methods(root_module, root_module['ns3::AddressValue'])
    register_Ns3BaseStationNetDevice_methods(root_module, root_module['ns3::BaseStationNetDevice'])
    register_Ns3CallbackImpl__Ns3ObjectBase___star___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Bool_Unsigned_long_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Const_ns3Mac48Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<const ns3::Packet>, const ns3::Mac48Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Ns3Mac48Address_Const_ns3Cid___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, const ns3::Cid &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Ns3Mac48Address_Ns3Cid_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, ns3::Cid, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3PacketBurst__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<const ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Ptr__lt__const_ns3Packet__gt___Unsigned_short_Const_ns3Address___amp___Const_ns3Address___amp___Ns3NetDevicePacketType_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3PacketBurst__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3SimpleOfdmWimaxChannel_methods(root_module, root_module['ns3::SimpleOfdmWimaxChannel'])
    register_Ns3SubscriberStationNetDevice_methods(root_module, root_module['ns3::SubscriberStationNetDevice'])
    register_Ns3HashImplementation_methods(root_module, root_module['ns3::Hash::Implementation'])
    register_Ns3HashFunctionFnv1a_methods(root_module, root_module['ns3::Hash::Function::Fnv1a'])
    register_Ns3HashFunctionHash32_methods(root_module, root_module['ns3::Hash::Function::Hash32'])
    register_Ns3HashFunctionHash64_methods(root_module, root_module['ns3::Hash::Function::Hash64'])
    register_Ns3HashFunctionMurmur3_methods(root_module, root_module['ns3::Hash::Function::Murmur3'])
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

def register_Ns3AsciiTraceHelper_methods(root_module, cls):
    ## trace-helper.h (module 'network'): ns3::AsciiTraceHelper::AsciiTraceHelper(ns3::AsciiTraceHelper const & arg0) [constructor]
    cls.add_constructor([param('ns3::AsciiTraceHelper const &', 'arg0')])
    ## trace-helper.h (module 'network'): ns3::AsciiTraceHelper::AsciiTraceHelper() [constructor]
    cls.add_constructor([])
    ## trace-helper.h (module 'network'): ns3::Ptr<ns3::OutputStreamWrapper> ns3::AsciiTraceHelper::CreateFileStream(std::string filename, std::ios_base::openmode filemode=std::ios_base::out) [member function]
    cls.add_method('CreateFileStream', 
                   'ns3::Ptr< ns3::OutputStreamWrapper >', 
                   [param('std::string', 'filename'), param('std::ios_base::openmode', 'filemode', default_value='std::ios_base::out')])
    ## trace-helper.h (module 'network'): static void ns3::AsciiTraceHelper::DefaultDequeueSinkWithContext(ns3::Ptr<ns3::OutputStreamWrapper> file, std::string context, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('DefaultDequeueSinkWithContext', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'file'), param('std::string', 'context'), param('ns3::Ptr< ns3::Packet const >', 'p')], 
                   is_static=True)
    ## trace-helper.h (module 'network'): static void ns3::AsciiTraceHelper::DefaultDequeueSinkWithoutContext(ns3::Ptr<ns3::OutputStreamWrapper> file, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('DefaultDequeueSinkWithoutContext', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'file'), param('ns3::Ptr< ns3::Packet const >', 'p')], 
                   is_static=True)
    ## trace-helper.h (module 'network'): static void ns3::AsciiTraceHelper::DefaultDropSinkWithContext(ns3::Ptr<ns3::OutputStreamWrapper> file, std::string context, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('DefaultDropSinkWithContext', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'file'), param('std::string', 'context'), param('ns3::Ptr< ns3::Packet const >', 'p')], 
                   is_static=True)
    ## trace-helper.h (module 'network'): static void ns3::AsciiTraceHelper::DefaultDropSinkWithoutContext(ns3::Ptr<ns3::OutputStreamWrapper> file, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('DefaultDropSinkWithoutContext', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'file'), param('ns3::Ptr< ns3::Packet const >', 'p')], 
                   is_static=True)
    ## trace-helper.h (module 'network'): static void ns3::AsciiTraceHelper::DefaultEnqueueSinkWithContext(ns3::Ptr<ns3::OutputStreamWrapper> file, std::string context, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('DefaultEnqueueSinkWithContext', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'file'), param('std::string', 'context'), param('ns3::Ptr< ns3::Packet const >', 'p')], 
                   is_static=True)
    ## trace-helper.h (module 'network'): static void ns3::AsciiTraceHelper::DefaultEnqueueSinkWithoutContext(ns3::Ptr<ns3::OutputStreamWrapper> file, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('DefaultEnqueueSinkWithoutContext', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'file'), param('ns3::Ptr< ns3::Packet const >', 'p')], 
                   is_static=True)
    ## trace-helper.h (module 'network'): static void ns3::AsciiTraceHelper::DefaultReceiveSinkWithContext(ns3::Ptr<ns3::OutputStreamWrapper> file, std::string context, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('DefaultReceiveSinkWithContext', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'file'), param('std::string', 'context'), param('ns3::Ptr< ns3::Packet const >', 'p')], 
                   is_static=True)
    ## trace-helper.h (module 'network'): static void ns3::AsciiTraceHelper::DefaultReceiveSinkWithoutContext(ns3::Ptr<ns3::OutputStreamWrapper> file, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('DefaultReceiveSinkWithoutContext', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'file'), param('ns3::Ptr< ns3::Packet const >', 'p')], 
                   is_static=True)
    ## trace-helper.h (module 'network'): std::string ns3::AsciiTraceHelper::GetFilenameFromDevice(std::string prefix, ns3::Ptr<ns3::NetDevice> device, bool useObjectNames=true) [member function]
    cls.add_method('GetFilenameFromDevice', 
                   'std::string', 
                   [param('std::string', 'prefix'), param('ns3::Ptr< ns3::NetDevice >', 'device'), param('bool', 'useObjectNames', default_value='true')])
    ## trace-helper.h (module 'network'): std::string ns3::AsciiTraceHelper::GetFilenameFromInterfacePair(std::string prefix, ns3::Ptr<ns3::Object> object, uint32_t interface, bool useObjectNames=true) [member function]
    cls.add_method('GetFilenameFromInterfacePair', 
                   'std::string', 
                   [param('std::string', 'prefix'), param('ns3::Ptr< ns3::Object >', 'object'), param('uint32_t', 'interface'), param('bool', 'useObjectNames', default_value='true')])
    return

def register_Ns3AsciiTraceHelperForDevice_methods(root_module, cls):
    ## trace-helper.h (module 'network'): ns3::AsciiTraceHelperForDevice::AsciiTraceHelperForDevice(ns3::AsciiTraceHelperForDevice const & arg0) [constructor]
    cls.add_constructor([param('ns3::AsciiTraceHelperForDevice const &', 'arg0')])
    ## trace-helper.h (module 'network'): ns3::AsciiTraceHelperForDevice::AsciiTraceHelperForDevice() [constructor]
    cls.add_constructor([])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAscii(std::string prefix, ns3::Ptr<ns3::NetDevice> nd, bool explicitFilename=false) [member function]
    cls.add_method('EnableAscii', 
                   'void', 
                   [param('std::string', 'prefix'), param('ns3::Ptr< ns3::NetDevice >', 'nd'), param('bool', 'explicitFilename', default_value='false')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAscii(ns3::Ptr<ns3::OutputStreamWrapper> stream, ns3::Ptr<ns3::NetDevice> nd) [member function]
    cls.add_method('EnableAscii', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'stream'), param('ns3::Ptr< ns3::NetDevice >', 'nd')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAscii(std::string prefix, std::string ndName, bool explicitFilename=false) [member function]
    cls.add_method('EnableAscii', 
                   'void', 
                   [param('std::string', 'prefix'), param('std::string', 'ndName'), param('bool', 'explicitFilename', default_value='false')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAscii(ns3::Ptr<ns3::OutputStreamWrapper> stream, std::string ndName) [member function]
    cls.add_method('EnableAscii', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'stream'), param('std::string', 'ndName')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAscii(std::string prefix, ns3::NetDeviceContainer d) [member function]
    cls.add_method('EnableAscii', 
                   'void', 
                   [param('std::string', 'prefix'), param('ns3::NetDeviceContainer', 'd')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAscii(ns3::Ptr<ns3::OutputStreamWrapper> stream, ns3::NetDeviceContainer d) [member function]
    cls.add_method('EnableAscii', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'stream'), param('ns3::NetDeviceContainer', 'd')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAscii(std::string prefix, ns3::NodeContainer n) [member function]
    cls.add_method('EnableAscii', 
                   'void', 
                   [param('std::string', 'prefix'), param('ns3::NodeContainer', 'n')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAscii(ns3::Ptr<ns3::OutputStreamWrapper> stream, ns3::NodeContainer n) [member function]
    cls.add_method('EnableAscii', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'stream'), param('ns3::NodeContainer', 'n')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAscii(std::string prefix, uint32_t nodeid, uint32_t deviceid, bool explicitFilename) [member function]
    cls.add_method('EnableAscii', 
                   'void', 
                   [param('std::string', 'prefix'), param('uint32_t', 'nodeid'), param('uint32_t', 'deviceid'), param('bool', 'explicitFilename')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAscii(ns3::Ptr<ns3::OutputStreamWrapper> stream, uint32_t nodeid, uint32_t deviceid) [member function]
    cls.add_method('EnableAscii', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'stream'), param('uint32_t', 'nodeid'), param('uint32_t', 'deviceid')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAsciiAll(std::string prefix) [member function]
    cls.add_method('EnableAsciiAll', 
                   'void', 
                   [param('std::string', 'prefix')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAsciiAll(ns3::Ptr<ns3::OutputStreamWrapper> stream) [member function]
    cls.add_method('EnableAsciiAll', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'stream')])
    ## trace-helper.h (module 'network'): void ns3::AsciiTraceHelperForDevice::EnableAsciiInternal(ns3::Ptr<ns3::OutputStreamWrapper> stream, std::string prefix, ns3::Ptr<ns3::NetDevice> nd, bool explicitFilename) [member function]
    cls.add_method('EnableAsciiInternal', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'stream'), param('std::string', 'prefix'), param('ns3::Ptr< ns3::NetDevice >', 'nd'), param('bool', 'explicitFilename')], 
                   is_pure_virtual=True, is_virtual=True)
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

def register_Ns3Cid_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_output_stream_operator()
    ## cid.h (module 'wimax'): ns3::Cid::Cid(ns3::Cid const & arg0) [constructor]
    cls.add_constructor([param('ns3::Cid const &', 'arg0')])
    ## cid.h (module 'wimax'): ns3::Cid::Cid() [constructor]
    cls.add_constructor([])
    ## cid.h (module 'wimax'): ns3::Cid::Cid(uint16_t cid) [constructor]
    cls.add_constructor([param('uint16_t', 'cid')])
    ## cid.h (module 'wimax'): static ns3::Cid ns3::Cid::Broadcast() [member function]
    cls.add_method('Broadcast', 
                   'ns3::Cid', 
                   [], 
                   is_static=True)
    ## cid.h (module 'wimax'): uint16_t ns3::Cid::GetIdentifier() const [member function]
    cls.add_method('GetIdentifier', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## cid.h (module 'wimax'): static ns3::Cid ns3::Cid::InitialRanging() [member function]
    cls.add_method('InitialRanging', 
                   'ns3::Cid', 
                   [], 
                   is_static=True)
    ## cid.h (module 'wimax'): bool ns3::Cid::IsBroadcast() const [member function]
    cls.add_method('IsBroadcast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## cid.h (module 'wimax'): bool ns3::Cid::IsInitialRanging() const [member function]
    cls.add_method('IsInitialRanging', 
                   'bool', 
                   [], 
                   is_const=True)
    ## cid.h (module 'wimax'): bool ns3::Cid::IsMulticast() const [member function]
    cls.add_method('IsMulticast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## cid.h (module 'wimax'): bool ns3::Cid::IsPadding() const [member function]
    cls.add_method('IsPadding', 
                   'bool', 
                   [], 
                   is_const=True)
    ## cid.h (module 'wimax'): static ns3::Cid ns3::Cid::Padding() [member function]
    cls.add_method('Padding', 
                   'ns3::Cid', 
                   [], 
                   is_static=True)
    return

def register_Ns3CidFactory_methods(root_module, cls):
    ## cid-factory.h (module 'wimax'): ns3::CidFactory::CidFactory(ns3::CidFactory const & arg0) [constructor]
    cls.add_constructor([param('ns3::CidFactory const &', 'arg0')])
    ## cid-factory.h (module 'wimax'): ns3::CidFactory::CidFactory() [constructor]
    cls.add_constructor([])
    ## cid-factory.h (module 'wimax'): ns3::Cid ns3::CidFactory::Allocate(ns3::Cid::Type type) [member function]
    cls.add_method('Allocate', 
                   'ns3::Cid', 
                   [param('ns3::Cid::Type', 'type')])
    ## cid-factory.h (module 'wimax'): ns3::Cid ns3::CidFactory::AllocateBasic() [member function]
    cls.add_method('AllocateBasic', 
                   'ns3::Cid', 
                   [])
    ## cid-factory.h (module 'wimax'): ns3::Cid ns3::CidFactory::AllocateMulticast() [member function]
    cls.add_method('AllocateMulticast', 
                   'ns3::Cid', 
                   [])
    ## cid-factory.h (module 'wimax'): ns3::Cid ns3::CidFactory::AllocatePrimary() [member function]
    cls.add_method('AllocatePrimary', 
                   'ns3::Cid', 
                   [])
    ## cid-factory.h (module 'wimax'): ns3::Cid ns3::CidFactory::AllocateTransportOrSecondary() [member function]
    cls.add_method('AllocateTransportOrSecondary', 
                   'ns3::Cid', 
                   [])
    ## cid-factory.h (module 'wimax'): void ns3::CidFactory::FreeCid(ns3::Cid cid) [member function]
    cls.add_method('FreeCid', 
                   'void', 
                   [param('ns3::Cid', 'cid')])
    ## cid-factory.h (module 'wimax'): bool ns3::CidFactory::IsBasic(ns3::Cid cid) const [member function]
    cls.add_method('IsBasic', 
                   'bool', 
                   [param('ns3::Cid', 'cid')], 
                   is_const=True)
    ## cid-factory.h (module 'wimax'): bool ns3::CidFactory::IsPrimary(ns3::Cid cid) const [member function]
    cls.add_method('IsPrimary', 
                   'bool', 
                   [param('ns3::Cid', 'cid')], 
                   is_const=True)
    ## cid-factory.h (module 'wimax'): bool ns3::CidFactory::IsTransport(ns3::Cid cid) const [member function]
    cls.add_method('IsTransport', 
                   'bool', 
                   [param('ns3::Cid', 'cid')], 
                   is_const=True)
    return

def register_Ns3CsParameters_methods(root_module, cls):
    ## cs-parameters.h (module 'wimax'): ns3::CsParameters::CsParameters(ns3::CsParameters const & arg0) [constructor]
    cls.add_constructor([param('ns3::CsParameters const &', 'arg0')])
    ## cs-parameters.h (module 'wimax'): ns3::CsParameters::CsParameters() [constructor]
    cls.add_constructor([])
    ## cs-parameters.h (module 'wimax'): ns3::CsParameters::CsParameters(ns3::Tlv tlv) [constructor]
    cls.add_constructor([param('ns3::Tlv', 'tlv')])
    ## cs-parameters.h (module 'wimax'): ns3::CsParameters::CsParameters(ns3::CsParameters::Action classifierDscAction, ns3::IpcsClassifierRecord classifier) [constructor]
    cls.add_constructor([param('ns3::CsParameters::Action', 'classifierDscAction'), param('ns3::IpcsClassifierRecord', 'classifier')])
    ## cs-parameters.h (module 'wimax'): ns3::CsParameters::Action ns3::CsParameters::GetClassifierDscAction() const [member function]
    cls.add_method('GetClassifierDscAction', 
                   'ns3::CsParameters::Action', 
                   [], 
                   is_const=True)
    ## cs-parameters.h (module 'wimax'): ns3::IpcsClassifierRecord ns3::CsParameters::GetPacketClassifierRule() const [member function]
    cls.add_method('GetPacketClassifierRule', 
                   'ns3::IpcsClassifierRecord', 
                   [], 
                   is_const=True)
    ## cs-parameters.h (module 'wimax'): void ns3::CsParameters::SetClassifierDscAction(ns3::CsParameters::Action action) [member function]
    cls.add_method('SetClassifierDscAction', 
                   'void', 
                   [param('ns3::CsParameters::Action', 'action')])
    ## cs-parameters.h (module 'wimax'): void ns3::CsParameters::SetPacketClassifierRule(ns3::IpcsClassifierRecord packetClassifierRule) [member function]
    cls.add_method('SetPacketClassifierRule', 
                   'void', 
                   [param('ns3::IpcsClassifierRecord', 'packetClassifierRule')])
    ## cs-parameters.h (module 'wimax'): ns3::Tlv ns3::CsParameters::ToTlv() const [member function]
    cls.add_method('ToTlv', 
                   'ns3::Tlv', 
                   [], 
                   is_const=True)
    return

def register_Ns3DcdChannelEncodings_methods(root_module, cls):
    ## dl-mac-messages.h (module 'wimax'): ns3::DcdChannelEncodings::DcdChannelEncodings(ns3::DcdChannelEncodings const & arg0) [constructor]
    cls.add_constructor([param('ns3::DcdChannelEncodings const &', 'arg0')])
    ## dl-mac-messages.h (module 'wimax'): ns3::DcdChannelEncodings::DcdChannelEncodings() [constructor]
    cls.add_constructor([])
    ## dl-mac-messages.h (module 'wimax'): uint16_t ns3::DcdChannelEncodings::GetBsEirp() const [member function]
    cls.add_method('GetBsEirp', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint16_t ns3::DcdChannelEncodings::GetEirxPIrMax() const [member function]
    cls.add_method('GetEirxPIrMax', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint32_t ns3::DcdChannelEncodings::GetFrequency() const [member function]
    cls.add_method('GetFrequency', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint16_t ns3::DcdChannelEncodings::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::DcdChannelEncodings::Read(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Read', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::DcdChannelEncodings::SetBsEirp(uint16_t bs_eirp) [member function]
    cls.add_method('SetBsEirp', 
                   'void', 
                   [param('uint16_t', 'bs_eirp')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::DcdChannelEncodings::SetEirxPIrMax(uint16_t rss_ir_max) [member function]
    cls.add_method('SetEirxPIrMax', 
                   'void', 
                   [param('uint16_t', 'rss_ir_max')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::DcdChannelEncodings::SetFrequency(uint32_t frequency) [member function]
    cls.add_method('SetFrequency', 
                   'void', 
                   [param('uint32_t', 'frequency')])
    ## dl-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::DcdChannelEncodings::Write(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Write', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::DcdChannelEncodings::DoRead(ns3::Buffer::Iterator start) [member function]
    cls.add_method('DoRead', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## dl-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::DcdChannelEncodings::DoWrite(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('DoWrite', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
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

def register_Ns3DefaultDeleter__Ns3OutputStreamWrapper_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::OutputStreamWrapper>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::OutputStreamWrapper>::DefaultDeleter(ns3::DefaultDeleter<ns3::OutputStreamWrapper> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::OutputStreamWrapper > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::OutputStreamWrapper>::Delete(ns3::OutputStreamWrapper * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::OutputStreamWrapper *', 'object')], 
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

def register_Ns3DlFramePrefixIe_methods(root_module, cls):
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): ns3::DlFramePrefixIe::DlFramePrefixIe(ns3::DlFramePrefixIe const & arg0) [constructor]
    cls.add_constructor([param('ns3::DlFramePrefixIe const &', 'arg0')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): ns3::DlFramePrefixIe::DlFramePrefixIe() [constructor]
    cls.add_constructor([])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint8_t ns3::DlFramePrefixIe::GetDiuc() const [member function]
    cls.add_method('GetDiuc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint16_t ns3::DlFramePrefixIe::GetLength() const [member function]
    cls.add_method('GetLength', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint8_t ns3::DlFramePrefixIe::GetPreamblePresent() const [member function]
    cls.add_method('GetPreamblePresent', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint8_t ns3::DlFramePrefixIe::GetRateId() const [member function]
    cls.add_method('GetRateId', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint16_t ns3::DlFramePrefixIe::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint16_t ns3::DlFramePrefixIe::GetStartTime() const [member function]
    cls.add_method('GetStartTime', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): ns3::Buffer::Iterator ns3::DlFramePrefixIe::Read(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Read', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::DlFramePrefixIe::SetDiuc(uint8_t diuc) [member function]
    cls.add_method('SetDiuc', 
                   'void', 
                   [param('uint8_t', 'diuc')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::DlFramePrefixIe::SetLength(uint16_t length) [member function]
    cls.add_method('SetLength', 
                   'void', 
                   [param('uint16_t', 'length')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::DlFramePrefixIe::SetPreamblePresent(uint8_t preamblePresent) [member function]
    cls.add_method('SetPreamblePresent', 
                   'void', 
                   [param('uint8_t', 'preamblePresent')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::DlFramePrefixIe::SetRateId(uint8_t rateId) [member function]
    cls.add_method('SetRateId', 
                   'void', 
                   [param('uint8_t', 'rateId')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::DlFramePrefixIe::SetStartTime(uint16_t startTime) [member function]
    cls.add_method('SetStartTime', 
                   'void', 
                   [param('uint16_t', 'startTime')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): ns3::Buffer::Iterator ns3::DlFramePrefixIe::Write(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Write', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True)
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

def register_Ns3IpcsClassifierRecord_methods(root_module, cls):
    ## ipcs-classifier-record.h (module 'wimax'): ns3::IpcsClassifierRecord::IpcsClassifierRecord(ns3::IpcsClassifierRecord const & arg0) [constructor]
    cls.add_constructor([param('ns3::IpcsClassifierRecord const &', 'arg0')])
    ## ipcs-classifier-record.h (module 'wimax'): ns3::IpcsClassifierRecord::IpcsClassifierRecord() [constructor]
    cls.add_constructor([])
    ## ipcs-classifier-record.h (module 'wimax'): ns3::IpcsClassifierRecord::IpcsClassifierRecord(ns3::Ipv4Address srcAddress, ns3::Ipv4Mask srcMask, ns3::Ipv4Address dstAddress, ns3::Ipv4Mask dstMask, uint16_t srcPortLow, uint16_t srcPortHigh, uint16_t dstPortLow, uint16_t dstPortHigh, uint8_t protocol, uint8_t priority) [constructor]
    cls.add_constructor([param('ns3::Ipv4Address', 'srcAddress'), param('ns3::Ipv4Mask', 'srcMask'), param('ns3::Ipv4Address', 'dstAddress'), param('ns3::Ipv4Mask', 'dstMask'), param('uint16_t', 'srcPortLow'), param('uint16_t', 'srcPortHigh'), param('uint16_t', 'dstPortLow'), param('uint16_t', 'dstPortHigh'), param('uint8_t', 'protocol'), param('uint8_t', 'priority')])
    ## ipcs-classifier-record.h (module 'wimax'): ns3::IpcsClassifierRecord::IpcsClassifierRecord(ns3::Tlv tlv) [constructor]
    cls.add_constructor([param('ns3::Tlv', 'tlv')])
    ## ipcs-classifier-record.h (module 'wimax'): void ns3::IpcsClassifierRecord::AddDstAddr(ns3::Ipv4Address dstAddress, ns3::Ipv4Mask dstMask) [member function]
    cls.add_method('AddDstAddr', 
                   'void', 
                   [param('ns3::Ipv4Address', 'dstAddress'), param('ns3::Ipv4Mask', 'dstMask')])
    ## ipcs-classifier-record.h (module 'wimax'): void ns3::IpcsClassifierRecord::AddDstPortRange(uint16_t dstPortLow, uint16_t dstPortHigh) [member function]
    cls.add_method('AddDstPortRange', 
                   'void', 
                   [param('uint16_t', 'dstPortLow'), param('uint16_t', 'dstPortHigh')])
    ## ipcs-classifier-record.h (module 'wimax'): void ns3::IpcsClassifierRecord::AddProtocol(uint8_t proto) [member function]
    cls.add_method('AddProtocol', 
                   'void', 
                   [param('uint8_t', 'proto')])
    ## ipcs-classifier-record.h (module 'wimax'): void ns3::IpcsClassifierRecord::AddSrcAddr(ns3::Ipv4Address srcAddress, ns3::Ipv4Mask srcMask) [member function]
    cls.add_method('AddSrcAddr', 
                   'void', 
                   [param('ns3::Ipv4Address', 'srcAddress'), param('ns3::Ipv4Mask', 'srcMask')])
    ## ipcs-classifier-record.h (module 'wimax'): void ns3::IpcsClassifierRecord::AddSrcPortRange(uint16_t srcPortLow, uint16_t srcPortHigh) [member function]
    cls.add_method('AddSrcPortRange', 
                   'void', 
                   [param('uint16_t', 'srcPortLow'), param('uint16_t', 'srcPortHigh')])
    ## ipcs-classifier-record.h (module 'wimax'): bool ns3::IpcsClassifierRecord::CheckMatch(ns3::Ipv4Address srcAddress, ns3::Ipv4Address dstAddress, uint16_t srcPort, uint16_t dstPort, uint8_t proto) const [member function]
    cls.add_method('CheckMatch', 
                   'bool', 
                   [param('ns3::Ipv4Address', 'srcAddress'), param('ns3::Ipv4Address', 'dstAddress'), param('uint16_t', 'srcPort'), param('uint16_t', 'dstPort'), param('uint8_t', 'proto')], 
                   is_const=True)
    ## ipcs-classifier-record.h (module 'wimax'): uint16_t ns3::IpcsClassifierRecord::GetCid() const [member function]
    cls.add_method('GetCid', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ipcs-classifier-record.h (module 'wimax'): uint16_t ns3::IpcsClassifierRecord::GetIndex() const [member function]
    cls.add_method('GetIndex', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ipcs-classifier-record.h (module 'wimax'): uint8_t ns3::IpcsClassifierRecord::GetPriority() const [member function]
    cls.add_method('GetPriority', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ipcs-classifier-record.h (module 'wimax'): void ns3::IpcsClassifierRecord::SetCid(uint16_t cid) [member function]
    cls.add_method('SetCid', 
                   'void', 
                   [param('uint16_t', 'cid')])
    ## ipcs-classifier-record.h (module 'wimax'): void ns3::IpcsClassifierRecord::SetIndex(uint16_t index) [member function]
    cls.add_method('SetIndex', 
                   'void', 
                   [param('uint16_t', 'index')])
    ## ipcs-classifier-record.h (module 'wimax'): void ns3::IpcsClassifierRecord::SetPriority(uint8_t prio) [member function]
    cls.add_method('SetPriority', 
                   'void', 
                   [param('uint8_t', 'prio')])
    ## ipcs-classifier-record.h (module 'wimax'): ns3::Tlv ns3::IpcsClassifierRecord::ToTlv() const [member function]
    cls.add_method('ToTlv', 
                   'ns3::Tlv', 
                   [], 
                   is_const=True)
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

def register_Ns3NetDeviceContainer_methods(root_module, cls):
    ## net-device-container.h (module 'network'): ns3::NetDeviceContainer::NetDeviceContainer(ns3::NetDeviceContainer const & arg0) [constructor]
    cls.add_constructor([param('ns3::NetDeviceContainer const &', 'arg0')])
    ## net-device-container.h (module 'network'): ns3::NetDeviceContainer::NetDeviceContainer() [constructor]
    cls.add_constructor([])
    ## net-device-container.h (module 'network'): ns3::NetDeviceContainer::NetDeviceContainer(ns3::Ptr<ns3::NetDevice> dev) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::NetDevice >', 'dev')])
    ## net-device-container.h (module 'network'): ns3::NetDeviceContainer::NetDeviceContainer(std::string devName) [constructor]
    cls.add_constructor([param('std::string', 'devName')])
    ## net-device-container.h (module 'network'): ns3::NetDeviceContainer::NetDeviceContainer(ns3::NetDeviceContainer const & a, ns3::NetDeviceContainer const & b) [constructor]
    cls.add_constructor([param('ns3::NetDeviceContainer const &', 'a'), param('ns3::NetDeviceContainer const &', 'b')])
    ## net-device-container.h (module 'network'): void ns3::NetDeviceContainer::Add(ns3::NetDeviceContainer other) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::NetDeviceContainer', 'other')])
    ## net-device-container.h (module 'network'): void ns3::NetDeviceContainer::Add(ns3::Ptr<ns3::NetDevice> device) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::Ptr< ns3::NetDevice >', 'device')])
    ## net-device-container.h (module 'network'): void ns3::NetDeviceContainer::Add(std::string deviceName) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('std::string', 'deviceName')])
    ## net-device-container.h (module 'network'): ns3::NetDeviceContainer::Iterator ns3::NetDeviceContainer::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::NetDeviceContainer::Iterator', 
                   [], 
                   is_const=True)
    ## net-device-container.h (module 'network'): ns3::NetDeviceContainer::Iterator ns3::NetDeviceContainer::End() const [member function]
    cls.add_method('End', 
                   'ns3::NetDeviceContainer::Iterator', 
                   [], 
                   is_const=True)
    ## net-device-container.h (module 'network'): ns3::Ptr<ns3::NetDevice> ns3::NetDeviceContainer::Get(uint32_t i) const [member function]
    cls.add_method('Get', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [param('uint32_t', 'i')], 
                   is_const=True)
    ## net-device-container.h (module 'network'): uint32_t ns3::NetDeviceContainer::GetN() const [member function]
    cls.add_method('GetN', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    return

def register_Ns3NodeContainer_methods(root_module, cls):
    ## node-container.h (module 'network'): ns3::NodeContainer::NodeContainer(ns3::NodeContainer const & arg0) [constructor]
    cls.add_constructor([param('ns3::NodeContainer const &', 'arg0')])
    ## node-container.h (module 'network'): ns3::NodeContainer::NodeContainer() [constructor]
    cls.add_constructor([])
    ## node-container.h (module 'network'): ns3::NodeContainer::NodeContainer(ns3::Ptr<ns3::Node> node) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::Node >', 'node')])
    ## node-container.h (module 'network'): ns3::NodeContainer::NodeContainer(std::string nodeName) [constructor]
    cls.add_constructor([param('std::string', 'nodeName')])
    ## node-container.h (module 'network'): ns3::NodeContainer::NodeContainer(uint32_t n, uint32_t systemId=0) [constructor]
    cls.add_constructor([param('uint32_t', 'n'), param('uint32_t', 'systemId', default_value='0')])
    ## node-container.h (module 'network'): ns3::NodeContainer::NodeContainer(ns3::NodeContainer const & a, ns3::NodeContainer const & b) [constructor]
    cls.add_constructor([param('ns3::NodeContainer const &', 'a'), param('ns3::NodeContainer const &', 'b')])
    ## node-container.h (module 'network'): ns3::NodeContainer::NodeContainer(ns3::NodeContainer const & a, ns3::NodeContainer const & b, ns3::NodeContainer const & c) [constructor]
    cls.add_constructor([param('ns3::NodeContainer const &', 'a'), param('ns3::NodeContainer const &', 'b'), param('ns3::NodeContainer const &', 'c')])
    ## node-container.h (module 'network'): ns3::NodeContainer::NodeContainer(ns3::NodeContainer const & a, ns3::NodeContainer const & b, ns3::NodeContainer const & c, ns3::NodeContainer const & d) [constructor]
    cls.add_constructor([param('ns3::NodeContainer const &', 'a'), param('ns3::NodeContainer const &', 'b'), param('ns3::NodeContainer const &', 'c'), param('ns3::NodeContainer const &', 'd')])
    ## node-container.h (module 'network'): ns3::NodeContainer::NodeContainer(ns3::NodeContainer const & a, ns3::NodeContainer const & b, ns3::NodeContainer const & c, ns3::NodeContainer const & d, ns3::NodeContainer const & e) [constructor]
    cls.add_constructor([param('ns3::NodeContainer const &', 'a'), param('ns3::NodeContainer const &', 'b'), param('ns3::NodeContainer const &', 'c'), param('ns3::NodeContainer const &', 'd'), param('ns3::NodeContainer const &', 'e')])
    ## node-container.h (module 'network'): void ns3::NodeContainer::Add(ns3::NodeContainer other) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::NodeContainer', 'other')])
    ## node-container.h (module 'network'): void ns3::NodeContainer::Add(ns3::Ptr<ns3::Node> node) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::Ptr< ns3::Node >', 'node')])
    ## node-container.h (module 'network'): void ns3::NodeContainer::Add(std::string nodeName) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('std::string', 'nodeName')])
    ## node-container.h (module 'network'): ns3::NodeContainer::Iterator ns3::NodeContainer::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::NodeContainer::Iterator', 
                   [], 
                   is_const=True)
    ## node-container.h (module 'network'): bool ns3::NodeContainer::Contains(uint32_t id) const [member function]
    cls.add_method('Contains', 
                   'bool', 
                   [param('uint32_t', 'id')], 
                   is_const=True)
    ## node-container.h (module 'network'): void ns3::NodeContainer::Create(uint32_t n) [member function]
    cls.add_method('Create', 
                   'void', 
                   [param('uint32_t', 'n')])
    ## node-container.h (module 'network'): void ns3::NodeContainer::Create(uint32_t n, uint32_t systemId) [member function]
    cls.add_method('Create', 
                   'void', 
                   [param('uint32_t', 'n'), param('uint32_t', 'systemId')])
    ## node-container.h (module 'network'): ns3::NodeContainer::Iterator ns3::NodeContainer::End() const [member function]
    cls.add_method('End', 
                   'ns3::NodeContainer::Iterator', 
                   [], 
                   is_const=True)
    ## node-container.h (module 'network'): ns3::Ptr<ns3::Node> ns3::NodeContainer::Get(uint32_t i) const [member function]
    cls.add_method('Get', 
                   'ns3::Ptr< ns3::Node >', 
                   [param('uint32_t', 'i')], 
                   is_const=True)
    ## node-container.h (module 'network'): static ns3::NodeContainer ns3::NodeContainer::GetGlobal() [member function]
    cls.add_method('GetGlobal', 
                   'ns3::NodeContainer', 
                   [], 
                   is_static=True)
    ## node-container.h (module 'network'): uint32_t ns3::NodeContainer::GetN() const [member function]
    cls.add_method('GetN', 
                   'uint32_t', 
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

def register_Ns3OfdmDcdChannelEncodings_methods(root_module, cls):
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDcdChannelEncodings::OfdmDcdChannelEncodings(ns3::OfdmDcdChannelEncodings const & arg0) [constructor]
    cls.add_constructor([param('ns3::OfdmDcdChannelEncodings const &', 'arg0')])
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDcdChannelEncodings::OfdmDcdChannelEncodings() [constructor]
    cls.add_constructor([])
    ## dl-mac-messages.h (module 'wimax'): ns3::Mac48Address ns3::OfdmDcdChannelEncodings::GetBaseStationId() const [member function]
    cls.add_method('GetBaseStationId', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmDcdChannelEncodings::GetChannelNr() const [member function]
    cls.add_method('GetChannelNr', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmDcdChannelEncodings::GetFrameDurationCode() const [member function]
    cls.add_method('GetFrameDurationCode', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint32_t ns3::OfdmDcdChannelEncodings::GetFrameNumber() const [member function]
    cls.add_method('GetFrameNumber', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmDcdChannelEncodings::GetRtg() const [member function]
    cls.add_method('GetRtg', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint16_t ns3::OfdmDcdChannelEncodings::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmDcdChannelEncodings::GetTtg() const [member function]
    cls.add_method('GetTtg', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDcdChannelEncodings::SetBaseStationId(ns3::Mac48Address baseStationId) [member function]
    cls.add_method('SetBaseStationId', 
                   'void', 
                   [param('ns3::Mac48Address', 'baseStationId')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDcdChannelEncodings::SetChannelNr(uint8_t channelNr) [member function]
    cls.add_method('SetChannelNr', 
                   'void', 
                   [param('uint8_t', 'channelNr')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDcdChannelEncodings::SetFrameDurationCode(uint8_t frameDurationCode) [member function]
    cls.add_method('SetFrameDurationCode', 
                   'void', 
                   [param('uint8_t', 'frameDurationCode')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDcdChannelEncodings::SetFrameNumber(uint32_t frameNumber) [member function]
    cls.add_method('SetFrameNumber', 
                   'void', 
                   [param('uint32_t', 'frameNumber')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDcdChannelEncodings::SetRtg(uint8_t rtg) [member function]
    cls.add_method('SetRtg', 
                   'void', 
                   [param('uint8_t', 'rtg')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDcdChannelEncodings::SetTtg(uint8_t ttg) [member function]
    cls.add_method('SetTtg', 
                   'void', 
                   [param('uint8_t', 'ttg')])
    ## dl-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmDcdChannelEncodings::DoRead(ns3::Buffer::Iterator start) [member function]
    cls.add_method('DoRead', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True, visibility='private')
    ## dl-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmDcdChannelEncodings::DoWrite(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('DoWrite', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True, visibility='private')
    return

def register_Ns3OfdmDlBurstProfile_methods(root_module, cls):
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDlBurstProfile::OfdmDlBurstProfile(ns3::OfdmDlBurstProfile const & arg0) [constructor]
    cls.add_constructor([param('ns3::OfdmDlBurstProfile const &', 'arg0')])
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDlBurstProfile::OfdmDlBurstProfile() [constructor]
    cls.add_constructor([])
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmDlBurstProfile::GetDiuc() const [member function]
    cls.add_method('GetDiuc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmDlBurstProfile::GetFecCodeType() const [member function]
    cls.add_method('GetFecCodeType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmDlBurstProfile::GetLength() const [member function]
    cls.add_method('GetLength', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint16_t ns3::OfdmDlBurstProfile::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmDlBurstProfile::GetType() const [member function]
    cls.add_method('GetType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmDlBurstProfile::Read(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Read', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDlBurstProfile::SetDiuc(uint8_t diuc) [member function]
    cls.add_method('SetDiuc', 
                   'void', 
                   [param('uint8_t', 'diuc')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDlBurstProfile::SetFecCodeType(uint8_t fecCodeType) [member function]
    cls.add_method('SetFecCodeType', 
                   'void', 
                   [param('uint8_t', 'fecCodeType')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDlBurstProfile::SetLength(uint8_t length) [member function]
    cls.add_method('SetLength', 
                   'void', 
                   [param('uint8_t', 'length')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDlBurstProfile::SetType(uint8_t type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('uint8_t', 'type')])
    ## dl-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmDlBurstProfile::Write(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Write', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True)
    return

def register_Ns3OfdmDlMapIe_methods(root_module, cls):
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDlMapIe::OfdmDlMapIe(ns3::OfdmDlMapIe const & arg0) [constructor]
    cls.add_constructor([param('ns3::OfdmDlMapIe const &', 'arg0')])
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDlMapIe::OfdmDlMapIe() [constructor]
    cls.add_constructor([])
    ## dl-mac-messages.h (module 'wimax'): ns3::Cid ns3::OfdmDlMapIe::GetCid() const [member function]
    cls.add_method('GetCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmDlMapIe::GetDiuc() const [member function]
    cls.add_method('GetDiuc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmDlMapIe::GetPreamblePresent() const [member function]
    cls.add_method('GetPreamblePresent', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint16_t ns3::OfdmDlMapIe::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint16_t ns3::OfdmDlMapIe::GetStartTime() const [member function]
    cls.add_method('GetStartTime', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmDlMapIe::Read(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Read', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDlMapIe::SetCid(ns3::Cid cid) [member function]
    cls.add_method('SetCid', 
                   'void', 
                   [param('ns3::Cid', 'cid')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDlMapIe::SetDiuc(uint8_t diuc) [member function]
    cls.add_method('SetDiuc', 
                   'void', 
                   [param('uint8_t', 'diuc')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDlMapIe::SetPreamblePresent(uint8_t preamblePresent) [member function]
    cls.add_method('SetPreamblePresent', 
                   'void', 
                   [param('uint8_t', 'preamblePresent')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::OfdmDlMapIe::SetStartTime(uint16_t startTime) [member function]
    cls.add_method('SetStartTime', 
                   'void', 
                   [param('uint16_t', 'startTime')])
    ## dl-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmDlMapIe::Write(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Write', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True)
    return

def register_Ns3OfdmUlBurstProfile_methods(root_module, cls):
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUlBurstProfile::OfdmUlBurstProfile(ns3::OfdmUlBurstProfile const & arg0) [constructor]
    cls.add_constructor([param('ns3::OfdmUlBurstProfile const &', 'arg0')])
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUlBurstProfile::OfdmUlBurstProfile() [constructor]
    cls.add_constructor([])
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmUlBurstProfile::GetFecCodeType() const [member function]
    cls.add_method('GetFecCodeType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmUlBurstProfile::GetLength() const [member function]
    cls.add_method('GetLength', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint16_t ns3::OfdmUlBurstProfile::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmUlBurstProfile::GetType() const [member function]
    cls.add_method('GetType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmUlBurstProfile::GetUiuc() const [member function]
    cls.add_method('GetUiuc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmUlBurstProfile::Read(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Read', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUlBurstProfile::SetFecCodeType(uint8_t fecCodeType) [member function]
    cls.add_method('SetFecCodeType', 
                   'void', 
                   [param('uint8_t', 'fecCodeType')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUlBurstProfile::SetLength(uint8_t length) [member function]
    cls.add_method('SetLength', 
                   'void', 
                   [param('uint8_t', 'length')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUlBurstProfile::SetType(uint8_t type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('uint8_t', 'type')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUlBurstProfile::SetUiuc(uint8_t uiuc) [member function]
    cls.add_method('SetUiuc', 
                   'void', 
                   [param('uint8_t', 'uiuc')])
    ## ul-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmUlBurstProfile::Write(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Write', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True)
    return

def register_Ns3OfdmUlMapIe_methods(root_module, cls):
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUlMapIe::OfdmUlMapIe(ns3::OfdmUlMapIe const & arg0) [constructor]
    cls.add_constructor([param('ns3::OfdmUlMapIe const &', 'arg0')])
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUlMapIe::OfdmUlMapIe() [constructor]
    cls.add_constructor([])
    ## ul-mac-messages.h (module 'wimax'): ns3::Cid ns3::OfdmUlMapIe::GetCid() const [member function]
    cls.add_method('GetCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint16_t ns3::OfdmUlMapIe::GetDuration() const [member function]
    cls.add_method('GetDuration', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmUlMapIe::GetMidambleRepetitionInterval() const [member function]
    cls.add_method('GetMidambleRepetitionInterval', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint16_t ns3::OfdmUlMapIe::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint16_t ns3::OfdmUlMapIe::GetStartTime() const [member function]
    cls.add_method('GetStartTime', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmUlMapIe::GetSubchannelIndex() const [member function]
    cls.add_method('GetSubchannelIndex', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmUlMapIe::GetUiuc() const [member function]
    cls.add_method('GetUiuc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmUlMapIe::Read(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Read', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUlMapIe::SetCid(ns3::Cid const & cid) [member function]
    cls.add_method('SetCid', 
                   'void', 
                   [param('ns3::Cid const &', 'cid')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUlMapIe::SetDuration(uint16_t duration) [member function]
    cls.add_method('SetDuration', 
                   'void', 
                   [param('uint16_t', 'duration')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUlMapIe::SetMidambleRepetitionInterval(uint8_t midambleRepetitionInterval) [member function]
    cls.add_method('SetMidambleRepetitionInterval', 
                   'void', 
                   [param('uint8_t', 'midambleRepetitionInterval')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUlMapIe::SetStartTime(uint16_t startTime) [member function]
    cls.add_method('SetStartTime', 
                   'void', 
                   [param('uint16_t', 'startTime')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUlMapIe::SetSubchannelIndex(uint8_t subchannelIndex) [member function]
    cls.add_method('SetSubchannelIndex', 
                   'void', 
                   [param('uint8_t', 'subchannelIndex')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUlMapIe::SetUiuc(uint8_t uiuc) [member function]
    cls.add_method('SetUiuc', 
                   'void', 
                   [param('uint8_t', 'uiuc')])
    ## ul-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmUlMapIe::Write(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Write', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True)
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

def register_Ns3PcapFile_methods(root_module, cls):
    ## pcap-file.h (module 'network'): ns3::PcapFile::PcapFile() [constructor]
    cls.add_constructor([])
    ## pcap-file.h (module 'network'): void ns3::PcapFile::Clear() [member function]
    cls.add_method('Clear', 
                   'void', 
                   [])
    ## pcap-file.h (module 'network'): void ns3::PcapFile::Close() [member function]
    cls.add_method('Close', 
                   'void', 
                   [])
    ## pcap-file.h (module 'network'): static bool ns3::PcapFile::Diff(std::string const & f1, std::string const & f2, uint32_t & sec, uint32_t & usec, uint32_t & packets, uint32_t snapLen=ns3::PcapFile::SNAPLEN_DEFAULT) [member function]
    cls.add_method('Diff', 
                   'bool', 
                   [param('std::string const &', 'f1'), param('std::string const &', 'f2'), param('uint32_t &', 'sec'), param('uint32_t &', 'usec'), param('uint32_t &', 'packets'), param('uint32_t', 'snapLen', default_value='ns3::PcapFile::SNAPLEN_DEFAULT')], 
                   is_static=True)
    ## pcap-file.h (module 'network'): bool ns3::PcapFile::Eof() const [member function]
    cls.add_method('Eof', 
                   'bool', 
                   [], 
                   is_const=True)
    ## pcap-file.h (module 'network'): bool ns3::PcapFile::Fail() const [member function]
    cls.add_method('Fail', 
                   'bool', 
                   [], 
                   is_const=True)
    ## pcap-file.h (module 'network'): uint32_t ns3::PcapFile::GetDataLinkType() [member function]
    cls.add_method('GetDataLinkType', 
                   'uint32_t', 
                   [])
    ## pcap-file.h (module 'network'): uint32_t ns3::PcapFile::GetMagic() [member function]
    cls.add_method('GetMagic', 
                   'uint32_t', 
                   [])
    ## pcap-file.h (module 'network'): uint32_t ns3::PcapFile::GetSigFigs() [member function]
    cls.add_method('GetSigFigs', 
                   'uint32_t', 
                   [])
    ## pcap-file.h (module 'network'): uint32_t ns3::PcapFile::GetSnapLen() [member function]
    cls.add_method('GetSnapLen', 
                   'uint32_t', 
                   [])
    ## pcap-file.h (module 'network'): bool ns3::PcapFile::GetSwapMode() [member function]
    cls.add_method('GetSwapMode', 
                   'bool', 
                   [])
    ## pcap-file.h (module 'network'): int32_t ns3::PcapFile::GetTimeZoneOffset() [member function]
    cls.add_method('GetTimeZoneOffset', 
                   'int32_t', 
                   [])
    ## pcap-file.h (module 'network'): uint16_t ns3::PcapFile::GetVersionMajor() [member function]
    cls.add_method('GetVersionMajor', 
                   'uint16_t', 
                   [])
    ## pcap-file.h (module 'network'): uint16_t ns3::PcapFile::GetVersionMinor() [member function]
    cls.add_method('GetVersionMinor', 
                   'uint16_t', 
                   [])
    ## pcap-file.h (module 'network'): void ns3::PcapFile::Init(uint32_t dataLinkType, uint32_t snapLen=ns3::PcapFile::SNAPLEN_DEFAULT, int32_t timeZoneCorrection=ns3::PcapFile::ZONE_DEFAULT, bool swapMode=false, bool nanosecMode=false) [member function]
    cls.add_method('Init', 
                   'void', 
                   [param('uint32_t', 'dataLinkType'), param('uint32_t', 'snapLen', default_value='ns3::PcapFile::SNAPLEN_DEFAULT'), param('int32_t', 'timeZoneCorrection', default_value='ns3::PcapFile::ZONE_DEFAULT'), param('bool', 'swapMode', default_value='false'), param('bool', 'nanosecMode', default_value='false')])
    ## pcap-file.h (module 'network'): bool ns3::PcapFile::IsNanoSecMode() [member function]
    cls.add_method('IsNanoSecMode', 
                   'bool', 
                   [])
    ## pcap-file.h (module 'network'): void ns3::PcapFile::Open(std::string const & filename, std::ios_base::openmode mode) [member function]
    cls.add_method('Open', 
                   'void', 
                   [param('std::string const &', 'filename'), param('std::ios_base::openmode', 'mode')])
    ## pcap-file.h (module 'network'): void ns3::PcapFile::Read(uint8_t * const data, uint32_t maxBytes, uint32_t & tsSec, uint32_t & tsUsec, uint32_t & inclLen, uint32_t & origLen, uint32_t & readLen) [member function]
    cls.add_method('Read', 
                   'void', 
                   [param('uint8_t * const', 'data'), param('uint32_t', 'maxBytes'), param('uint32_t &', 'tsSec'), param('uint32_t &', 'tsUsec'), param('uint32_t &', 'inclLen'), param('uint32_t &', 'origLen'), param('uint32_t &', 'readLen')])
    ## pcap-file.h (module 'network'): void ns3::PcapFile::Write(uint32_t tsSec, uint32_t tsUsec, uint8_t const * const data, uint32_t totalLen) [member function]
    cls.add_method('Write', 
                   'void', 
                   [param('uint32_t', 'tsSec'), param('uint32_t', 'tsUsec'), param('uint8_t const * const', 'data'), param('uint32_t', 'totalLen')])
    ## pcap-file.h (module 'network'): void ns3::PcapFile::Write(uint32_t tsSec, uint32_t tsUsec, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('Write', 
                   'void', 
                   [param('uint32_t', 'tsSec'), param('uint32_t', 'tsUsec'), param('ns3::Ptr< ns3::Packet const >', 'p')])
    ## pcap-file.h (module 'network'): void ns3::PcapFile::Write(uint32_t tsSec, uint32_t tsUsec, ns3::Header const & header, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('Write', 
                   'void', 
                   [param('uint32_t', 'tsSec'), param('uint32_t', 'tsUsec'), param('ns3::Header const &', 'header'), param('ns3::Ptr< ns3::Packet const >', 'p')])
    ## pcap-file.h (module 'network'): ns3::PcapFile::SNAPLEN_DEFAULT [variable]
    cls.add_static_attribute('SNAPLEN_DEFAULT', 'uint32_t const', is_const=True)
    ## pcap-file.h (module 'network'): ns3::PcapFile::ZONE_DEFAULT [variable]
    cls.add_static_attribute('ZONE_DEFAULT', 'int32_t const', is_const=True)
    return

def register_Ns3PcapHelper_methods(root_module, cls):
    ## trace-helper.h (module 'network'): ns3::PcapHelper::PcapHelper(ns3::PcapHelper const & arg0) [constructor]
    cls.add_constructor([param('ns3::PcapHelper const &', 'arg0')])
    ## trace-helper.h (module 'network'): ns3::PcapHelper::PcapHelper() [constructor]
    cls.add_constructor([])
    ## trace-helper.h (module 'network'): ns3::Ptr<ns3::PcapFileWrapper> ns3::PcapHelper::CreateFile(std::string filename, std::ios_base::openmode filemode, ns3::PcapHelper::DataLinkType dataLinkType, uint32_t snapLen=std::numeric_limits<unsigned int>::max(), int32_t tzCorrection=0) [member function]
    cls.add_method('CreateFile', 
                   'ns3::Ptr< ns3::PcapFileWrapper >', 
                   [param('std::string', 'filename'), param('std::ios_base::openmode', 'filemode'), param('ns3::PcapHelper::DataLinkType', 'dataLinkType'), param('uint32_t', 'snapLen', default_value='std::numeric_limits<unsigned int>::max()'), param('int32_t', 'tzCorrection', default_value='0')])
    ## trace-helper.h (module 'network'): std::string ns3::PcapHelper::GetFilenameFromDevice(std::string prefix, ns3::Ptr<ns3::NetDevice> device, bool useObjectNames=true) [member function]
    cls.add_method('GetFilenameFromDevice', 
                   'std::string', 
                   [param('std::string', 'prefix'), param('ns3::Ptr< ns3::NetDevice >', 'device'), param('bool', 'useObjectNames', default_value='true')])
    ## trace-helper.h (module 'network'): std::string ns3::PcapHelper::GetFilenameFromInterfacePair(std::string prefix, ns3::Ptr<ns3::Object> object, uint32_t interface, bool useObjectNames=true) [member function]
    cls.add_method('GetFilenameFromInterfacePair', 
                   'std::string', 
                   [param('std::string', 'prefix'), param('ns3::Ptr< ns3::Object >', 'object'), param('uint32_t', 'interface'), param('bool', 'useObjectNames', default_value='true')])
    return

def register_Ns3PcapHelperForDevice_methods(root_module, cls):
    ## trace-helper.h (module 'network'): ns3::PcapHelperForDevice::PcapHelperForDevice(ns3::PcapHelperForDevice const & arg0) [constructor]
    cls.add_constructor([param('ns3::PcapHelperForDevice const &', 'arg0')])
    ## trace-helper.h (module 'network'): ns3::PcapHelperForDevice::PcapHelperForDevice() [constructor]
    cls.add_constructor([])
    ## trace-helper.h (module 'network'): void ns3::PcapHelperForDevice::EnablePcap(std::string prefix, ns3::Ptr<ns3::NetDevice> nd, bool promiscuous=false, bool explicitFilename=false) [member function]
    cls.add_method('EnablePcap', 
                   'void', 
                   [param('std::string', 'prefix'), param('ns3::Ptr< ns3::NetDevice >', 'nd'), param('bool', 'promiscuous', default_value='false'), param('bool', 'explicitFilename', default_value='false')])
    ## trace-helper.h (module 'network'): void ns3::PcapHelperForDevice::EnablePcap(std::string prefix, std::string ndName, bool promiscuous=false, bool explicitFilename=false) [member function]
    cls.add_method('EnablePcap', 
                   'void', 
                   [param('std::string', 'prefix'), param('std::string', 'ndName'), param('bool', 'promiscuous', default_value='false'), param('bool', 'explicitFilename', default_value='false')])
    ## trace-helper.h (module 'network'): void ns3::PcapHelperForDevice::EnablePcap(std::string prefix, ns3::NetDeviceContainer d, bool promiscuous=false) [member function]
    cls.add_method('EnablePcap', 
                   'void', 
                   [param('std::string', 'prefix'), param('ns3::NetDeviceContainer', 'd'), param('bool', 'promiscuous', default_value='false')])
    ## trace-helper.h (module 'network'): void ns3::PcapHelperForDevice::EnablePcap(std::string prefix, ns3::NodeContainer n, bool promiscuous=false) [member function]
    cls.add_method('EnablePcap', 
                   'void', 
                   [param('std::string', 'prefix'), param('ns3::NodeContainer', 'n'), param('bool', 'promiscuous', default_value='false')])
    ## trace-helper.h (module 'network'): void ns3::PcapHelperForDevice::EnablePcap(std::string prefix, uint32_t nodeid, uint32_t deviceid, bool promiscuous=false) [member function]
    cls.add_method('EnablePcap', 
                   'void', 
                   [param('std::string', 'prefix'), param('uint32_t', 'nodeid'), param('uint32_t', 'deviceid'), param('bool', 'promiscuous', default_value='false')])
    ## trace-helper.h (module 'network'): void ns3::PcapHelperForDevice::EnablePcapAll(std::string prefix, bool promiscuous=false) [member function]
    cls.add_method('EnablePcapAll', 
                   'void', 
                   [param('std::string', 'prefix'), param('bool', 'promiscuous', default_value='false')])
    ## trace-helper.h (module 'network'): void ns3::PcapHelperForDevice::EnablePcapInternal(std::string prefix, ns3::Ptr<ns3::NetDevice> nd, bool promiscuous, bool explicitFilename) [member function]
    cls.add_method('EnablePcapInternal', 
                   'void', 
                   [param('std::string', 'prefix'), param('ns3::Ptr< ns3::NetDevice >', 'nd'), param('bool', 'promiscuous'), param('bool', 'explicitFilename')], 
                   is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3SNRToBlockErrorRateManager_methods(root_module, cls):
    ## snr-to-block-error-rate-manager.h (module 'wimax'): ns3::SNRToBlockErrorRateManager::SNRToBlockErrorRateManager(ns3::SNRToBlockErrorRateManager const & arg0) [constructor]
    cls.add_constructor([param('ns3::SNRToBlockErrorRateManager const &', 'arg0')])
    ## snr-to-block-error-rate-manager.h (module 'wimax'): ns3::SNRToBlockErrorRateManager::SNRToBlockErrorRateManager() [constructor]
    cls.add_constructor([])
    ## snr-to-block-error-rate-manager.h (module 'wimax'): void ns3::SNRToBlockErrorRateManager::ActivateLoss(bool loss) [member function]
    cls.add_method('ActivateLoss', 
                   'void', 
                   [param('bool', 'loss')])
    ## snr-to-block-error-rate-manager.h (module 'wimax'): double ns3::SNRToBlockErrorRateManager::GetBlockErrorRate(double SNR, uint8_t modulation) [member function]
    cls.add_method('GetBlockErrorRate', 
                   'double', 
                   [param('double', 'SNR'), param('uint8_t', 'modulation')])
    ## snr-to-block-error-rate-manager.h (module 'wimax'): ns3::SNRToBlockErrorRateRecord * ns3::SNRToBlockErrorRateManager::GetSNRToBlockErrorRateRecord(double SNR, uint8_t modulation) [member function]
    cls.add_method('GetSNRToBlockErrorRateRecord', 
                   'ns3::SNRToBlockErrorRateRecord *', 
                   [param('double', 'SNR'), param('uint8_t', 'modulation')])
    ## snr-to-block-error-rate-manager.h (module 'wimax'): std::string ns3::SNRToBlockErrorRateManager::GetTraceFilePath() [member function]
    cls.add_method('GetTraceFilePath', 
                   'std::string', 
                   [])
    ## snr-to-block-error-rate-manager.h (module 'wimax'): void ns3::SNRToBlockErrorRateManager::LoadDefaultTraces() [member function]
    cls.add_method('LoadDefaultTraces', 
                   'void', 
                   [])
    ## snr-to-block-error-rate-manager.h (module 'wimax'): void ns3::SNRToBlockErrorRateManager::LoadTraces() [member function]
    cls.add_method('LoadTraces', 
                   'void', 
                   [])
    ## snr-to-block-error-rate-manager.h (module 'wimax'): void ns3::SNRToBlockErrorRateManager::ReLoadTraces() [member function]
    cls.add_method('ReLoadTraces', 
                   'void', 
                   [])
    ## snr-to-block-error-rate-manager.h (module 'wimax'): void ns3::SNRToBlockErrorRateManager::SetTraceFilePath(char * traceFilePath) [member function]
    cls.add_method('SetTraceFilePath', 
                   'void', 
                   [param('char *', 'traceFilePath')])
    return

def register_Ns3SNRToBlockErrorRateRecord_methods(root_module, cls):
    ## snr-to-block-error-rate-record.h (module 'wimax'): ns3::SNRToBlockErrorRateRecord::SNRToBlockErrorRateRecord(ns3::SNRToBlockErrorRateRecord const & arg0) [constructor]
    cls.add_constructor([param('ns3::SNRToBlockErrorRateRecord const &', 'arg0')])
    ## snr-to-block-error-rate-record.h (module 'wimax'): ns3::SNRToBlockErrorRateRecord::SNRToBlockErrorRateRecord(double snrValue, double bitErrorRate, double BlockErrorRate, double sigma2, double I1, double I2) [constructor]
    cls.add_constructor([param('double', 'snrValue'), param('double', 'bitErrorRate'), param('double', 'BlockErrorRate'), param('double', 'sigma2'), param('double', 'I1'), param('double', 'I2')])
    ## snr-to-block-error-rate-record.h (module 'wimax'): ns3::SNRToBlockErrorRateRecord * ns3::SNRToBlockErrorRateRecord::Copy() [member function]
    cls.add_method('Copy', 
                   'ns3::SNRToBlockErrorRateRecord *', 
                   [])
    ## snr-to-block-error-rate-record.h (module 'wimax'): double ns3::SNRToBlockErrorRateRecord::GetBitErrorRate() [member function]
    cls.add_method('GetBitErrorRate', 
                   'double', 
                   [])
    ## snr-to-block-error-rate-record.h (module 'wimax'): double ns3::SNRToBlockErrorRateRecord::GetBlockErrorRate() [member function]
    cls.add_method('GetBlockErrorRate', 
                   'double', 
                   [])
    ## snr-to-block-error-rate-record.h (module 'wimax'): double ns3::SNRToBlockErrorRateRecord::GetI1() [member function]
    cls.add_method('GetI1', 
                   'double', 
                   [])
    ## snr-to-block-error-rate-record.h (module 'wimax'): double ns3::SNRToBlockErrorRateRecord::GetI2() [member function]
    cls.add_method('GetI2', 
                   'double', 
                   [])
    ## snr-to-block-error-rate-record.h (module 'wimax'): double ns3::SNRToBlockErrorRateRecord::GetSNRValue() [member function]
    cls.add_method('GetSNRValue', 
                   'double', 
                   [])
    ## snr-to-block-error-rate-record.h (module 'wimax'): double ns3::SNRToBlockErrorRateRecord::GetSigma2() [member function]
    cls.add_method('GetSigma2', 
                   'double', 
                   [])
    ## snr-to-block-error-rate-record.h (module 'wimax'): void ns3::SNRToBlockErrorRateRecord::SetBitErrorRate(double bitErrorRate) [member function]
    cls.add_method('SetBitErrorRate', 
                   'void', 
                   [param('double', 'bitErrorRate')])
    ## snr-to-block-error-rate-record.h (module 'wimax'): void ns3::SNRToBlockErrorRateRecord::SetBlockErrorRate(double blockErrorRate) [member function]
    cls.add_method('SetBlockErrorRate', 
                   'void', 
                   [param('double', 'blockErrorRate')])
    ## snr-to-block-error-rate-record.h (module 'wimax'): void ns3::SNRToBlockErrorRateRecord::SetI1(double i1) [member function]
    cls.add_method('SetI1', 
                   'void', 
                   [param('double', 'i1')])
    ## snr-to-block-error-rate-record.h (module 'wimax'): void ns3::SNRToBlockErrorRateRecord::SetI2(double i2) [member function]
    cls.add_method('SetI2', 
                   'void', 
                   [param('double', 'i2')])
    ## snr-to-block-error-rate-record.h (module 'wimax'): void ns3::SNRToBlockErrorRateRecord::SetSNRValue(double snrValue) [member function]
    cls.add_method('SetSNRValue', 
                   'void', 
                   [param('double', 'snrValue')])
    return

def register_Ns3SSRecord_methods(root_module, cls):
    ## ss-record.h (module 'wimax'): ns3::SSRecord::SSRecord(ns3::SSRecord const & arg0) [constructor]
    cls.add_constructor([param('ns3::SSRecord const &', 'arg0')])
    ## ss-record.h (module 'wimax'): ns3::SSRecord::SSRecord() [constructor]
    cls.add_constructor([])
    ## ss-record.h (module 'wimax'): ns3::SSRecord::SSRecord(ns3::Mac48Address macAddress) [constructor]
    cls.add_constructor([param('ns3::Mac48Address', 'macAddress')])
    ## ss-record.h (module 'wimax'): ns3::SSRecord::SSRecord(ns3::Mac48Address macAddress, ns3::Ipv4Address IPaddress) [constructor]
    cls.add_constructor([param('ns3::Mac48Address', 'macAddress'), param('ns3::Ipv4Address', 'IPaddress')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::AddServiceFlow(ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('AddServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow *', 'serviceFlow')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::DisablePollForRanging() [member function]
    cls.add_method('DisablePollForRanging', 
                   'void', 
                   [])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::EnablePollForRanging() [member function]
    cls.add_method('EnablePollForRanging', 
                   'void', 
                   [])
    ## ss-record.h (module 'wimax'): bool ns3::SSRecord::GetAreServiceFlowsAllocated() const [member function]
    cls.add_method('GetAreServiceFlowsAllocated', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): ns3::Cid ns3::SSRecord::GetBasicCid() const [member function]
    cls.add_method('GetBasicCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): ns3::DsaRsp ns3::SSRecord::GetDsaRsp() const [member function]
    cls.add_method('GetDsaRsp', 
                   'ns3::DsaRsp', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): uint8_t ns3::SSRecord::GetDsaRspRetries() const [member function]
    cls.add_method('GetDsaRspRetries', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): bool ns3::SSRecord::GetHasServiceFlowBe() const [member function]
    cls.add_method('GetHasServiceFlowBe', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): bool ns3::SSRecord::GetHasServiceFlowNrtps() const [member function]
    cls.add_method('GetHasServiceFlowNrtps', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): bool ns3::SSRecord::GetHasServiceFlowRtps() const [member function]
    cls.add_method('GetHasServiceFlowRtps', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): bool ns3::SSRecord::GetHasServiceFlowUgs() const [member function]
    cls.add_method('GetHasServiceFlowUgs', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): ns3::Ipv4Address ns3::SSRecord::GetIPAddress() [member function]
    cls.add_method('GetIPAddress', 
                   'ns3::Ipv4Address', 
                   [])
    ## ss-record.h (module 'wimax'): uint8_t ns3::SSRecord::GetInvitedRangRetries() const [member function]
    cls.add_method('GetInvitedRangRetries', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): bool ns3::SSRecord::GetIsBroadcastSS() [member function]
    cls.add_method('GetIsBroadcastSS', 
                   'bool', 
                   [])
    ## ss-record.h (module 'wimax'): ns3::Mac48Address ns3::SSRecord::GetMacAddress() const [member function]
    cls.add_method('GetMacAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): ns3::WimaxPhy::ModulationType ns3::SSRecord::GetModulationType() const [member function]
    cls.add_method('GetModulationType', 
                   'ns3::WimaxPhy::ModulationType', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): bool ns3::SSRecord::GetPollForRanging() const [member function]
    cls.add_method('GetPollForRanging', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): bool ns3::SSRecord::GetPollMeBit() const [member function]
    cls.add_method('GetPollMeBit', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): ns3::Cid ns3::SSRecord::GetPrimaryCid() const [member function]
    cls.add_method('GetPrimaryCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): uint8_t ns3::SSRecord::GetRangingCorrectionRetries() const [member function]
    cls.add_method('GetRangingCorrectionRetries', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): ns3::WimaxNetDevice::RangingStatus ns3::SSRecord::GetRangingStatus() const [member function]
    cls.add_method('GetRangingStatus', 
                   'ns3::WimaxNetDevice::RangingStatus', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): std::vector<ns3::ServiceFlow *, std::allocator<ns3::ServiceFlow *>> ns3::SSRecord::GetServiceFlows(ns3::ServiceFlow::SchedulingType schedulingType) const [member function]
    cls.add_method('GetServiceFlows', 
                   'std::vector< ns3::ServiceFlow * >', 
                   [param('ns3::ServiceFlow::SchedulingType', 'schedulingType')], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): uint16_t ns3::SSRecord::GetSfTransactionId() const [member function]
    cls.add_method('GetSfTransactionId', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::IncrementDsaRspRetries() [member function]
    cls.add_method('IncrementDsaRspRetries', 
                   'void', 
                   [])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::IncrementInvitedRangingRetries() [member function]
    cls.add_method('IncrementInvitedRangingRetries', 
                   'void', 
                   [])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::IncrementRangingCorrectionRetries() [member function]
    cls.add_method('IncrementRangingCorrectionRetries', 
                   'void', 
                   [])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::ResetInvitedRangingRetries() [member function]
    cls.add_method('ResetInvitedRangingRetries', 
                   'void', 
                   [])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::ResetRangingCorrectionRetries() [member function]
    cls.add_method('ResetRangingCorrectionRetries', 
                   'void', 
                   [])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetAreServiceFlowsAllocated(bool val) [member function]
    cls.add_method('SetAreServiceFlowsAllocated', 
                   'void', 
                   [param('bool', 'val')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetBasicCid(ns3::Cid basicCid) [member function]
    cls.add_method('SetBasicCid', 
                   'void', 
                   [param('ns3::Cid', 'basicCid')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetDsaRsp(ns3::DsaRsp dsaRsp) [member function]
    cls.add_method('SetDsaRsp', 
                   'void', 
                   [param('ns3::DsaRsp', 'dsaRsp')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetDsaRspRetries(uint8_t dsaRspRetries) [member function]
    cls.add_method('SetDsaRspRetries', 
                   'void', 
                   [param('uint8_t', 'dsaRspRetries')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetIPAddress(ns3::Ipv4Address IPaddress) [member function]
    cls.add_method('SetIPAddress', 
                   'void', 
                   [param('ns3::Ipv4Address', 'IPaddress')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetIsBroadcastSS(bool broadcast_enable) [member function]
    cls.add_method('SetIsBroadcastSS', 
                   'void', 
                   [param('bool', 'broadcast_enable')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetMacAddress(ns3::Mac48Address macAddress) [member function]
    cls.add_method('SetMacAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'macAddress')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetModulationType(ns3::WimaxPhy::ModulationType modulationType) [member function]
    cls.add_method('SetModulationType', 
                   'void', 
                   [param('ns3::WimaxPhy::ModulationType', 'modulationType')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetPollMeBit(bool pollMeBit) [member function]
    cls.add_method('SetPollMeBit', 
                   'void', 
                   [param('bool', 'pollMeBit')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetPrimaryCid(ns3::Cid primaryCid) [member function]
    cls.add_method('SetPrimaryCid', 
                   'void', 
                   [param('ns3::Cid', 'primaryCid')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetRangingStatus(ns3::WimaxNetDevice::RangingStatus rangingStatus) [member function]
    cls.add_method('SetRangingStatus', 
                   'void', 
                   [param('ns3::WimaxNetDevice::RangingStatus', 'rangingStatus')])
    ## ss-record.h (module 'wimax'): void ns3::SSRecord::SetSfTransactionId(uint16_t sfTransactionId) [member function]
    cls.add_method('SetSfTransactionId', 
                   'void', 
                   [param('uint16_t', 'sfTransactionId')])
    return

def register_Ns3SendParams_methods(root_module, cls):
    ## send-params.h (module 'wimax'): ns3::SendParams::SendParams(ns3::SendParams const & arg0) [constructor]
    cls.add_constructor([param('ns3::SendParams const &', 'arg0')])
    ## send-params.h (module 'wimax'): ns3::SendParams::SendParams() [constructor]
    cls.add_constructor([])
    return

def register_Ns3ServiceFlow_methods(root_module, cls):
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::ServiceFlow(ns3::Tlv tlv) [constructor]
    cls.add_constructor([param('ns3::Tlv', 'tlv')])
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::ServiceFlow() [constructor]
    cls.add_constructor([])
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::ServiceFlow(ns3::ServiceFlow::Direction direction) [constructor]
    cls.add_constructor([param('ns3::ServiceFlow::Direction', 'direction')])
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::ServiceFlow(ns3::ServiceFlow const & sf) [constructor]
    cls.add_constructor([param('ns3::ServiceFlow const &', 'sf')])
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::ServiceFlow(uint32_t sfid, ns3::ServiceFlow::Direction direction, ns3::Ptr<ns3::WimaxConnection> connection) [constructor]
    cls.add_constructor([param('uint32_t', 'sfid'), param('ns3::ServiceFlow::Direction', 'direction'), param('ns3::Ptr< ns3::WimaxConnection >', 'connection')])
    ## service-flow.h (module 'wimax'): bool ns3::ServiceFlow::CheckClassifierMatch(ns3::Ipv4Address srcAddress, ns3::Ipv4Address dstAddress, uint16_t srcPort, uint16_t dstPort, uint8_t proto) const [member function]
    cls.add_method('CheckClassifierMatch', 
                   'bool', 
                   [param('ns3::Ipv4Address', 'srcAddress'), param('ns3::Ipv4Address', 'dstAddress'), param('uint16_t', 'srcPort'), param('uint16_t', 'dstPort'), param('uint8_t', 'proto')], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::CleanUpQueue() [member function]
    cls.add_method('CleanUpQueue', 
                   'void', 
                   [])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::CopyParametersFrom(ns3::ServiceFlow sf) [member function]
    cls.add_method('CopyParametersFrom', 
                   'void', 
                   [param('ns3::ServiceFlow', 'sf')])
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetArqBlockLifeTime() const [member function]
    cls.add_method('GetArqBlockLifeTime', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetArqBlockSize() const [member function]
    cls.add_method('GetArqBlockSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint8_t ns3::ServiceFlow::GetArqDeliverInOrder() const [member function]
    cls.add_method('GetArqDeliverInOrder', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint8_t ns3::ServiceFlow::GetArqEnable() const [member function]
    cls.add_method('GetArqEnable', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetArqPurgeTimeout() const [member function]
    cls.add_method('GetArqPurgeTimeout', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetArqRetryTimeoutRx() const [member function]
    cls.add_method('GetArqRetryTimeoutRx', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetArqRetryTimeoutTx() const [member function]
    cls.add_method('GetArqRetryTimeoutTx', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetArqSyncLoss() const [member function]
    cls.add_method('GetArqSyncLoss', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetArqWindowSize() const [member function]
    cls.add_method('GetArqWindowSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetCid() const [member function]
    cls.add_method('GetCid', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): ns3::Ptr<ns3::WimaxConnection> ns3::ServiceFlow::GetConnection() const [member function]
    cls.add_method('GetConnection', 
                   'ns3::Ptr< ns3::WimaxConnection >', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): ns3::CsParameters ns3::ServiceFlow::GetConvergenceSublayerParam() const [member function]
    cls.add_method('GetConvergenceSublayerParam', 
                   'ns3::CsParameters', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::CsSpecification ns3::ServiceFlow::GetCsSpecification() const [member function]
    cls.add_method('GetCsSpecification', 
                   'ns3::ServiceFlow::CsSpecification', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::Direction ns3::ServiceFlow::GetDirection() const [member function]
    cls.add_method('GetDirection', 
                   'ns3::ServiceFlow::Direction', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint8_t ns3::ServiceFlow::GetFixedversusVariableSduIndicator() const [member function]
    cls.add_method('GetFixedversusVariableSduIndicator', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): bool ns3::ServiceFlow::GetIsEnabled() const [member function]
    cls.add_method('GetIsEnabled', 
                   'bool', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): bool ns3::ServiceFlow::GetIsMulticast() const [member function]
    cls.add_method('GetIsMulticast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint32_t ns3::ServiceFlow::GetMaxSustainedTrafficRate() const [member function]
    cls.add_method('GetMaxSustainedTrafficRate', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint32_t ns3::ServiceFlow::GetMaxTrafficBurst() const [member function]
    cls.add_method('GetMaxTrafficBurst', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint32_t ns3::ServiceFlow::GetMaximumLatency() const [member function]
    cls.add_method('GetMaximumLatency', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint32_t ns3::ServiceFlow::GetMinReservedTrafficRate() const [member function]
    cls.add_method('GetMinReservedTrafficRate', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint32_t ns3::ServiceFlow::GetMinTolerableTrafficRate() const [member function]
    cls.add_method('GetMinTolerableTrafficRate', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): ns3::WimaxPhy::ModulationType ns3::ServiceFlow::GetModulation() const [member function]
    cls.add_method('GetModulation', 
                   'ns3::WimaxPhy::ModulationType', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint8_t ns3::ServiceFlow::GetQosParamSetType() const [member function]
    cls.add_method('GetQosParamSetType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): ns3::Ptr<ns3::WimaxMacQueue> ns3::ServiceFlow::GetQueue() const [member function]
    cls.add_method('GetQueue', 
                   'ns3::Ptr< ns3::WimaxMacQueue >', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): ns3::ServiceFlowRecord * ns3::ServiceFlow::GetRecord() const [member function]
    cls.add_method('GetRecord', 
                   'ns3::ServiceFlowRecord *', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint32_t ns3::ServiceFlow::GetRequestTransmissionPolicy() const [member function]
    cls.add_method('GetRequestTransmissionPolicy', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::SchedulingType ns3::ServiceFlow::GetSchedulingType() const [member function]
    cls.add_method('GetSchedulingType', 
                   'ns3::ServiceFlow::SchedulingType', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): char * ns3::ServiceFlow::GetSchedulingTypeStr() const [member function]
    cls.add_method('GetSchedulingTypeStr', 
                   'char *', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint8_t ns3::ServiceFlow::GetSduSize() const [member function]
    cls.add_method('GetSduSize', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): std::string ns3::ServiceFlow::GetServiceClassName() const [member function]
    cls.add_method('GetServiceClassName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::SchedulingType ns3::ServiceFlow::GetServiceSchedulingType() const [member function]
    cls.add_method('GetServiceSchedulingType', 
                   'ns3::ServiceFlow::SchedulingType', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint32_t ns3::ServiceFlow::GetSfid() const [member function]
    cls.add_method('GetSfid', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetTargetSAID() const [member function]
    cls.add_method('GetTargetSAID', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint32_t ns3::ServiceFlow::GetToleratedJitter() const [member function]
    cls.add_method('GetToleratedJitter', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint8_t ns3::ServiceFlow::GetTrafficPriority() const [member function]
    cls.add_method('GetTrafficPriority', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): ns3::ServiceFlow::Type ns3::ServiceFlow::GetType() const [member function]
    cls.add_method('GetType', 
                   'ns3::ServiceFlow::Type', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetUnsolicitedGrantInterval() const [member function]
    cls.add_method('GetUnsolicitedGrantInterval', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): uint16_t ns3::ServiceFlow::GetUnsolicitedPollingInterval() const [member function]
    cls.add_method('GetUnsolicitedPollingInterval', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): bool ns3::ServiceFlow::HasPackets() const [member function]
    cls.add_method('HasPackets', 
                   'bool', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): bool ns3::ServiceFlow::HasPackets(ns3::MacHeaderType::HeaderType packetType) const [member function]
    cls.add_method('HasPackets', 
                   'bool', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::InitValues() [member function]
    cls.add_method('InitValues', 
                   'void', 
                   [])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::PrintQoSParameters() const [member function]
    cls.add_method('PrintQoSParameters', 
                   'void', 
                   [], 
                   is_const=True)
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetArqBlockLifeTime(uint16_t lifeTime) [member function]
    cls.add_method('SetArqBlockLifeTime', 
                   'void', 
                   [param('uint16_t', 'lifeTime')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetArqBlockSize(uint16_t size) [member function]
    cls.add_method('SetArqBlockSize', 
                   'void', 
                   [param('uint16_t', 'size')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetArqDeliverInOrder(uint8_t inOrder) [member function]
    cls.add_method('SetArqDeliverInOrder', 
                   'void', 
                   [param('uint8_t', 'inOrder')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetArqEnable(uint8_t arqEnable) [member function]
    cls.add_method('SetArqEnable', 
                   'void', 
                   [param('uint8_t', 'arqEnable')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetArqPurgeTimeout(uint16_t timeout) [member function]
    cls.add_method('SetArqPurgeTimeout', 
                   'void', 
                   [param('uint16_t', 'timeout')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetArqRetryTimeoutRx(uint16_t timeout) [member function]
    cls.add_method('SetArqRetryTimeoutRx', 
                   'void', 
                   [param('uint16_t', 'timeout')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetArqRetryTimeoutTx(uint16_t timeout) [member function]
    cls.add_method('SetArqRetryTimeoutTx', 
                   'void', 
                   [param('uint16_t', 'timeout')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetArqSyncLoss(uint16_t syncLoss) [member function]
    cls.add_method('SetArqSyncLoss', 
                   'void', 
                   [param('uint16_t', 'syncLoss')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetArqWindowSize(uint16_t arqWindowSize) [member function]
    cls.add_method('SetArqWindowSize', 
                   'void', 
                   [param('uint16_t', 'arqWindowSize')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetConnection(ns3::Ptr<ns3::WimaxConnection> connection) [member function]
    cls.add_method('SetConnection', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxConnection >', 'connection')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetConvergenceSublayerParam(ns3::CsParameters csparam) [member function]
    cls.add_method('SetConvergenceSublayerParam', 
                   'void', 
                   [param('ns3::CsParameters', 'csparam')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetCsSpecification(ns3::ServiceFlow::CsSpecification spec) [member function]
    cls.add_method('SetCsSpecification', 
                   'void', 
                   [param('ns3::ServiceFlow::CsSpecification', 'spec')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetDirection(ns3::ServiceFlow::Direction direction) [member function]
    cls.add_method('SetDirection', 
                   'void', 
                   [param('ns3::ServiceFlow::Direction', 'direction')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetFixedversusVariableSduIndicator(uint8_t sduIndicator) [member function]
    cls.add_method('SetFixedversusVariableSduIndicator', 
                   'void', 
                   [param('uint8_t', 'sduIndicator')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetIsEnabled(bool isEnabled) [member function]
    cls.add_method('SetIsEnabled', 
                   'void', 
                   [param('bool', 'isEnabled')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetIsMulticast(bool isMulticast) [member function]
    cls.add_method('SetIsMulticast', 
                   'void', 
                   [param('bool', 'isMulticast')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetMaxSustainedTrafficRate(uint32_t maxSustainedRate) [member function]
    cls.add_method('SetMaxSustainedTrafficRate', 
                   'void', 
                   [param('uint32_t', 'maxSustainedRate')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetMaxTrafficBurst(uint32_t maxTrafficBurst) [member function]
    cls.add_method('SetMaxTrafficBurst', 
                   'void', 
                   [param('uint32_t', 'maxTrafficBurst')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetMaximumLatency(uint32_t MaximumLatency) [member function]
    cls.add_method('SetMaximumLatency', 
                   'void', 
                   [param('uint32_t', 'MaximumLatency')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetMinReservedTrafficRate(uint32_t minResvRate) [member function]
    cls.add_method('SetMinReservedTrafficRate', 
                   'void', 
                   [param('uint32_t', 'minResvRate')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetMinTolerableTrafficRate(uint32_t minJitter) [member function]
    cls.add_method('SetMinTolerableTrafficRate', 
                   'void', 
                   [param('uint32_t', 'minJitter')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetModulation(ns3::WimaxPhy::ModulationType modulationType) [member function]
    cls.add_method('SetModulation', 
                   'void', 
                   [param('ns3::WimaxPhy::ModulationType', 'modulationType')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetQosParamSetType(uint8_t type) [member function]
    cls.add_method('SetQosParamSetType', 
                   'void', 
                   [param('uint8_t', 'type')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetRecord(ns3::ServiceFlowRecord * record) [member function]
    cls.add_method('SetRecord', 
                   'void', 
                   [param('ns3::ServiceFlowRecord *', 'record')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetRequestTransmissionPolicy(uint32_t policy) [member function]
    cls.add_method('SetRequestTransmissionPolicy', 
                   'void', 
                   [param('uint32_t', 'policy')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetSduSize(uint8_t sduSize) [member function]
    cls.add_method('SetSduSize', 
                   'void', 
                   [param('uint8_t', 'sduSize')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetServiceClassName(std::string name) [member function]
    cls.add_method('SetServiceClassName', 
                   'void', 
                   [param('std::string', 'name')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetServiceSchedulingType(ns3::ServiceFlow::SchedulingType schedType) [member function]
    cls.add_method('SetServiceSchedulingType', 
                   'void', 
                   [param('ns3::ServiceFlow::SchedulingType', 'schedType')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetSfid(uint32_t sfid) [member function]
    cls.add_method('SetSfid', 
                   'void', 
                   [param('uint32_t', 'sfid')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetTargetSAID(uint16_t targetSaid) [member function]
    cls.add_method('SetTargetSAID', 
                   'void', 
                   [param('uint16_t', 'targetSaid')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetToleratedJitter(uint32_t jitter) [member function]
    cls.add_method('SetToleratedJitter', 
                   'void', 
                   [param('uint32_t', 'jitter')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetTrafficPriority(uint8_t priority) [member function]
    cls.add_method('SetTrafficPriority', 
                   'void', 
                   [param('uint8_t', 'priority')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetType(ns3::ServiceFlow::Type type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('ns3::ServiceFlow::Type', 'type')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetUnsolicitedGrantInterval(uint16_t unsolicitedGrantInterval) [member function]
    cls.add_method('SetUnsolicitedGrantInterval', 
                   'void', 
                   [param('uint16_t', 'unsolicitedGrantInterval')])
    ## service-flow.h (module 'wimax'): void ns3::ServiceFlow::SetUnsolicitedPollingInterval(uint16_t unsolicitedPollingInterval) [member function]
    cls.add_method('SetUnsolicitedPollingInterval', 
                   'void', 
                   [param('uint16_t', 'unsolicitedPollingInterval')])
    ## service-flow.h (module 'wimax'): ns3::Tlv ns3::ServiceFlow::ToTlv() const [member function]
    cls.add_method('ToTlv', 
                   'ns3::Tlv', 
                   [], 
                   is_const=True)
    return

def register_Ns3ServiceFlowRecord_methods(root_module, cls):
    ## service-flow-record.h (module 'wimax'): ns3::ServiceFlowRecord::ServiceFlowRecord(ns3::ServiceFlowRecord const & arg0) [constructor]
    cls.add_constructor([param('ns3::ServiceFlowRecord const &', 'arg0')])
    ## service-flow-record.h (module 'wimax'): ns3::ServiceFlowRecord::ServiceFlowRecord() [constructor]
    cls.add_constructor([])
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetBacklogged() const [member function]
    cls.add_method('GetBacklogged', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetBackloggedTemp() const [member function]
    cls.add_method('GetBackloggedTemp', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetBwSinceLastExpiry() [member function]
    cls.add_method('GetBwSinceLastExpiry', 
                   'uint32_t', 
                   [])
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetBytesRcvd() const [member function]
    cls.add_method('GetBytesRcvd', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetBytesSent() const [member function]
    cls.add_method('GetBytesSent', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow-record.h (module 'wimax'): ns3::Time ns3::ServiceFlowRecord::GetDlTimeStamp() const [member function]
    cls.add_method('GetDlTimeStamp', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetGrantSize() const [member function]
    cls.add_method('GetGrantSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow-record.h (module 'wimax'): ns3::Time ns3::ServiceFlowRecord::GetGrantTimeStamp() const [member function]
    cls.add_method('GetGrantTimeStamp', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetGrantedBandwidth() [member function]
    cls.add_method('GetGrantedBandwidth', 
                   'uint32_t', 
                   [])
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetGrantedBandwidthTemp() [member function]
    cls.add_method('GetGrantedBandwidthTemp', 
                   'uint32_t', 
                   [])
    ## service-flow-record.h (module 'wimax'): ns3::Time ns3::ServiceFlowRecord::GetLastGrantTime() const [member function]
    cls.add_method('GetLastGrantTime', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetPktsRcvd() const [member function]
    cls.add_method('GetPktsRcvd', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetPktsSent() const [member function]
    cls.add_method('GetPktsSent', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow-record.h (module 'wimax'): uint32_t ns3::ServiceFlowRecord::GetRequestedBandwidth() [member function]
    cls.add_method('GetRequestedBandwidth', 
                   'uint32_t', 
                   [])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::IncreaseBacklogged(uint32_t backlogged) [member function]
    cls.add_method('IncreaseBacklogged', 
                   'void', 
                   [param('uint32_t', 'backlogged')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::IncreaseBackloggedTemp(uint32_t backloggedTemp) [member function]
    cls.add_method('IncreaseBackloggedTemp', 
                   'void', 
                   [param('uint32_t', 'backloggedTemp')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetBacklogged(uint32_t backlogged) [member function]
    cls.add_method('SetBacklogged', 
                   'void', 
                   [param('uint32_t', 'backlogged')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetBackloggedTemp(uint32_t backloggedTemp) [member function]
    cls.add_method('SetBackloggedTemp', 
                   'void', 
                   [param('uint32_t', 'backloggedTemp')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetBwSinceLastExpiry(uint32_t bwSinceLastExpiry) [member function]
    cls.add_method('SetBwSinceLastExpiry', 
                   'void', 
                   [param('uint32_t', 'bwSinceLastExpiry')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetBytesRcvd(uint32_t bytesRcvd) [member function]
    cls.add_method('SetBytesRcvd', 
                   'void', 
                   [param('uint32_t', 'bytesRcvd')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetBytesSent(uint32_t bytesSent) [member function]
    cls.add_method('SetBytesSent', 
                   'void', 
                   [param('uint32_t', 'bytesSent')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetDlTimeStamp(ns3::Time dlTimeStamp) [member function]
    cls.add_method('SetDlTimeStamp', 
                   'void', 
                   [param('ns3::Time', 'dlTimeStamp')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetGrantSize(uint32_t grantSize) [member function]
    cls.add_method('SetGrantSize', 
                   'void', 
                   [param('uint32_t', 'grantSize')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetGrantTimeStamp(ns3::Time grantTimeStamp) [member function]
    cls.add_method('SetGrantTimeStamp', 
                   'void', 
                   [param('ns3::Time', 'grantTimeStamp')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetGrantedBandwidth(uint32_t grantedBandwidth) [member function]
    cls.add_method('SetGrantedBandwidth', 
                   'void', 
                   [param('uint32_t', 'grantedBandwidth')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetGrantedBandwidthTemp(uint32_t grantedBandwidthTemp) [member function]
    cls.add_method('SetGrantedBandwidthTemp', 
                   'void', 
                   [param('uint32_t', 'grantedBandwidthTemp')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetLastGrantTime(ns3::Time grantTime) [member function]
    cls.add_method('SetLastGrantTime', 
                   'void', 
                   [param('ns3::Time', 'grantTime')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetPktsRcvd(uint32_t pktsRcvd) [member function]
    cls.add_method('SetPktsRcvd', 
                   'void', 
                   [param('uint32_t', 'pktsRcvd')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetPktsSent(uint32_t pktsSent) [member function]
    cls.add_method('SetPktsSent', 
                   'void', 
                   [param('uint32_t', 'pktsSent')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::SetRequestedBandwidth(uint32_t requestedBandwidth) [member function]
    cls.add_method('SetRequestedBandwidth', 
                   'void', 
                   [param('uint32_t', 'requestedBandwidth')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::UpdateBwSinceLastExpiry(uint32_t bwSinceLastExpiry) [member function]
    cls.add_method('UpdateBwSinceLastExpiry', 
                   'void', 
                   [param('uint32_t', 'bwSinceLastExpiry')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::UpdateBytesRcvd(uint32_t bytesRcvd) [member function]
    cls.add_method('UpdateBytesRcvd', 
                   'void', 
                   [param('uint32_t', 'bytesRcvd')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::UpdateBytesSent(uint32_t bytesSent) [member function]
    cls.add_method('UpdateBytesSent', 
                   'void', 
                   [param('uint32_t', 'bytesSent')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::UpdateGrantedBandwidth(uint32_t grantedBandwidth) [member function]
    cls.add_method('UpdateGrantedBandwidth', 
                   'void', 
                   [param('uint32_t', 'grantedBandwidth')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::UpdateGrantedBandwidthTemp(uint32_t grantedBandwidthTemp) [member function]
    cls.add_method('UpdateGrantedBandwidthTemp', 
                   'void', 
                   [param('uint32_t', 'grantedBandwidthTemp')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::UpdatePktsRcvd(uint32_t pktsRcvd) [member function]
    cls.add_method('UpdatePktsRcvd', 
                   'void', 
                   [param('uint32_t', 'pktsRcvd')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::UpdatePktsSent(uint32_t pktsSent) [member function]
    cls.add_method('UpdatePktsSent', 
                   'void', 
                   [param('uint32_t', 'pktsSent')])
    ## service-flow-record.h (module 'wimax'): void ns3::ServiceFlowRecord::UpdateRequestedBandwidth(uint32_t requestedBandwidth) [member function]
    cls.add_method('UpdateRequestedBandwidth', 
                   'void', 
                   [param('uint32_t', 'requestedBandwidth')])
    return

def register_Ns3SimpleRefCount__Ns3Object_Ns3ObjectBase_Ns3ObjectDeleter_methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter>::SimpleRefCount(ns3::SimpleRefCount<ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter > const &', 'o')])
    return

def register_Ns3Simulator_methods(root_module, cls):
    ## simulator.h (module 'core'): ns3::Simulator::Simulator(ns3::Simulator const & arg0) [constructor]
    cls.add_constructor([param('ns3::Simulator const &', 'arg0')])
    ## simulator.h (module 'core'): static void ns3::Simulator::Cancel(ns3::EventId const & id) [member function]
    cls.add_method('Cancel', 
                   'void', 
                   [param('ns3::EventId const &', 'id')], 
                   is_static=True)
    ## simulator.h (module 'core'): static void ns3::Simulator::Destroy() [member function]
    cls.add_method('Destroy', 
                   'void', 
                   [], 
                   is_static=True)
    ## simulator.h (module 'core'): static uint32_t ns3::Simulator::GetContext() [member function]
    cls.add_method('GetContext', 
                   'uint32_t', 
                   [], 
                   is_static=True)
    ## simulator.h (module 'core'): static ns3::Time ns3::Simulator::GetDelayLeft(ns3::EventId const & id) [member function]
    cls.add_method('GetDelayLeft', 
                   'ns3::Time', 
                   [param('ns3::EventId const &', 'id')], 
                   is_static=True)
    ## simulator.h (module 'core'): static uint64_t ns3::Simulator::GetEventCount() [member function]
    cls.add_method('GetEventCount', 
                   'uint64_t', 
                   [], 
                   is_static=True)
    ## simulator.h (module 'core'): static ns3::Ptr<ns3::SimulatorImpl> ns3::Simulator::GetImplementation() [member function]
    cls.add_method('GetImplementation', 
                   'ns3::Ptr< ns3::SimulatorImpl >', 
                   [], 
                   is_static=True)
    ## simulator.h (module 'core'): static ns3::Time ns3::Simulator::GetMaximumSimulationTime() [member function]
    cls.add_method('GetMaximumSimulationTime', 
                   'ns3::Time', 
                   [], 
                   is_static=True)
    ## simulator.h (module 'core'): static uint32_t ns3::Simulator::GetSystemId() [member function]
    cls.add_method('GetSystemId', 
                   'uint32_t', 
                   [], 
                   is_static=True)
    ## simulator.h (module 'core'): static bool ns3::Simulator::IsExpired(ns3::EventId const & id) [member function]
    cls.add_method('IsExpired', 
                   'bool', 
                   [param('ns3::EventId const &', 'id')], 
                   is_static=True)
    ## simulator.h (module 'core'): static bool ns3::Simulator::IsFinished() [member function]
    cls.add_method('IsFinished', 
                   'bool', 
                   [], 
                   is_static=True)
    ## simulator.h (module 'core'): static ns3::Time ns3::Simulator::Now() [member function]
    cls.add_method('Now', 
                   'ns3::Time', 
                   [], 
                   is_static=True)
    ## simulator.h (module 'core'): static void ns3::Simulator::Remove(ns3::EventId const & id) [member function]
    cls.add_method('Remove', 
                   'void', 
                   [param('ns3::EventId const &', 'id')], 
                   is_static=True)
    ## simulator.h (module 'core'): static void ns3::Simulator::SetImplementation(ns3::Ptr<ns3::SimulatorImpl> impl) [member function]
    cls.add_method('SetImplementation', 
                   'void', 
                   [param('ns3::Ptr< ns3::SimulatorImpl >', 'impl')], 
                   is_static=True)
    ## simulator.h (module 'core'): static void ns3::Simulator::SetScheduler(ns3::ObjectFactory schedulerFactory) [member function]
    cls.add_method('SetScheduler', 
                   'void', 
                   [param('ns3::ObjectFactory', 'schedulerFactory')], 
                   is_static=True)
    ## simulator.h (module 'core'): static void ns3::Simulator::Stop() [member function]
    cls.add_method('Stop', 
                   'void', 
                   [], 
                   is_static=True)
    ## simulator.h (module 'core'): static void ns3::Simulator::Stop(ns3::Time const & delay) [member function]
    cls.add_method('Stop', 
                   'void', 
                   [param('ns3::Time const &', 'delay')], 
                   is_static=True)
    return

def register_Ns3SortProcess_methods(root_module, cls):
    ## ul-job.h (module 'wimax'): ns3::SortProcess::SortProcess() [constructor]
    cls.add_constructor([])
    ## ul-job.h (module 'wimax'): ns3::SortProcess::SortProcess(ns3::SortProcess const & arg0) [constructor]
    cls.add_constructor([param('ns3::SortProcess const &', 'arg0')])
    ## ul-job.h (module 'wimax'): bool ns3::SortProcess::operator()(ns3::PriorityUlJob & left, ns3::PriorityUlJob & right) const [member operator]
    cls.add_method('operator()', 
                   'bool', 
                   [param('ns3::PriorityUlJob &', 'left'), param('ns3::PriorityUlJob &', 'right')], 
                   custom_name='__call__', is_const=True)
    return

def register_Ns3SortProcessPtr_methods(root_module, cls):
    ## ul-job.h (module 'wimax'): ns3::SortProcessPtr::SortProcessPtr() [constructor]
    cls.add_constructor([])
    ## ul-job.h (module 'wimax'): ns3::SortProcessPtr::SortProcessPtr(ns3::SortProcessPtr const & arg0) [constructor]
    cls.add_constructor([param('ns3::SortProcessPtr const &', 'arg0')])
    ## ul-job.h (module 'wimax'): bool ns3::SortProcessPtr::operator()(ns3::Ptr<ns3::PriorityUlJob> & left, ns3::Ptr<ns3::PriorityUlJob> & right) const [member operator]
    cls.add_method('operator()', 
                   'bool', 
                   [param('ns3::Ptr< ns3::PriorityUlJob > &', 'left'), param('ns3::Ptr< ns3::PriorityUlJob > &', 'right')], 
                   custom_name='__call__', is_const=True)
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

def register_Ns3TlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::TlvValue::TlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::TlvValue::TlvValue(ns3::TlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::TlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::TlvValue * ns3::TlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::TlvValue *', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::TlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLen) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLen')], 
                   is_pure_virtual=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::TlvValue::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): void ns3::TlvValue::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3TosTlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::TosTlvValue::TosTlvValue(ns3::TosTlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::TosTlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::TosTlvValue::TosTlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::TosTlvValue::TosTlvValue(uint8_t low, uint8_t high, uint8_t mask) [constructor]
    cls.add_constructor([param('uint8_t', 'low'), param('uint8_t', 'high'), param('uint8_t', 'mask')])
    ## wimax-tlv.h (module 'wimax'): ns3::TosTlvValue * ns3::TosTlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::TosTlvValue *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::TosTlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLength) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLength')], 
                   is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint8_t ns3::TosTlvValue::GetHigh() const [member function]
    cls.add_method('GetHigh', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): uint8_t ns3::TosTlvValue::GetLow() const [member function]
    cls.add_method('GetLow', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): uint8_t ns3::TosTlvValue::GetMask() const [member function]
    cls.add_method('GetMask', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::TosTlvValue::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): void ns3::TosTlvValue::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
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

def register_Ns3U16TlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::U16TlvValue::U16TlvValue(ns3::U16TlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::U16TlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::U16TlvValue::U16TlvValue(uint16_t value) [constructor]
    cls.add_constructor([param('uint16_t', 'value')])
    ## wimax-tlv.h (module 'wimax'): ns3::U16TlvValue::U16TlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::U16TlvValue * ns3::U16TlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::U16TlvValue *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::U16TlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLen) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLen')], 
                   is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::U16TlvValue::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::U16TlvValue::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint16_t ns3::U16TlvValue::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): void ns3::U16TlvValue::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3U32TlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::U32TlvValue::U32TlvValue(ns3::U32TlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::U32TlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::U32TlvValue::U32TlvValue(uint32_t value) [constructor]
    cls.add_constructor([param('uint32_t', 'value')])
    ## wimax-tlv.h (module 'wimax'): ns3::U32TlvValue::U32TlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::U32TlvValue * ns3::U32TlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::U32TlvValue *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::U32TlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLen) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLen')], 
                   is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::U32TlvValue::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::U32TlvValue::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::U32TlvValue::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): void ns3::U32TlvValue::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3U8TlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::U8TlvValue::U8TlvValue(ns3::U8TlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::U8TlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::U8TlvValue::U8TlvValue(uint8_t value) [constructor]
    cls.add_constructor([param('uint8_t', 'value')])
    ## wimax-tlv.h (module 'wimax'): ns3::U8TlvValue::U8TlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::U8TlvValue * ns3::U8TlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::U8TlvValue *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::U8TlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLen) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLen')], 
                   is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::U8TlvValue::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::U8TlvValue::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint8_t ns3::U8TlvValue::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): void ns3::U8TlvValue::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3UcdChannelEncodings_methods(root_module, cls):
    ## ul-mac-messages.h (module 'wimax'): ns3::UcdChannelEncodings::UcdChannelEncodings(ns3::UcdChannelEncodings const & arg0) [constructor]
    cls.add_constructor([param('ns3::UcdChannelEncodings const &', 'arg0')])
    ## ul-mac-messages.h (module 'wimax'): ns3::UcdChannelEncodings::UcdChannelEncodings() [constructor]
    cls.add_constructor([])
    ## ul-mac-messages.h (module 'wimax'): uint16_t ns3::UcdChannelEncodings::GetBwReqOppSize() const [member function]
    cls.add_method('GetBwReqOppSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint32_t ns3::UcdChannelEncodings::GetFrequency() const [member function]
    cls.add_method('GetFrequency', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint16_t ns3::UcdChannelEncodings::GetRangReqOppSize() const [member function]
    cls.add_method('GetRangReqOppSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint16_t ns3::UcdChannelEncodings::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::UcdChannelEncodings::Read(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Read', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::UcdChannelEncodings::SetBwReqOppSize(uint16_t bwReqOppSize) [member function]
    cls.add_method('SetBwReqOppSize', 
                   'void', 
                   [param('uint16_t', 'bwReqOppSize')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::UcdChannelEncodings::SetFrequency(uint32_t frequency) [member function]
    cls.add_method('SetFrequency', 
                   'void', 
                   [param('uint32_t', 'frequency')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::UcdChannelEncodings::SetRangReqOppSize(uint16_t rangReqOppSize) [member function]
    cls.add_method('SetRangReqOppSize', 
                   'void', 
                   [param('uint16_t', 'rangReqOppSize')])
    ## ul-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::UcdChannelEncodings::Write(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Write', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::UcdChannelEncodings::DoRead(ns3::Buffer::Iterator start) [member function]
    cls.add_method('DoRead', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## ul-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::UcdChannelEncodings::DoWrite(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('DoWrite', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    return

def register_Ns3VectorTlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::VectorTlvValue::VectorTlvValue(ns3::VectorTlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::VectorTlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::VectorTlvValue::VectorTlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): void ns3::VectorTlvValue::Add(ns3::Tlv const & val) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::Tlv const &', 'val')])
    ## wimax-tlv.h (module 'wimax'): ns3::VectorTlvValue::Iterator ns3::VectorTlvValue::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::VectorTlvValue::Iterator', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): ns3::VectorTlvValue * ns3::VectorTlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::VectorTlvValue *', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::VectorTlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLength) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLength')], 
                   is_pure_virtual=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): ns3::VectorTlvValue::Iterator ns3::VectorTlvValue::End() const [member function]
    cls.add_method('End', 
                   'ns3::VectorTlvValue::Iterator', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::VectorTlvValue::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): void ns3::VectorTlvValue::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3WimaxHelper_methods(root_module, cls):
    ## wimax-helper.h (module 'wimax'): ns3::WimaxHelper::WimaxHelper(ns3::WimaxHelper const & arg0) [constructor]
    cls.add_constructor([param('ns3::WimaxHelper const &', 'arg0')])
    ## wimax-helper.h (module 'wimax'): ns3::WimaxHelper::WimaxHelper() [constructor]
    cls.add_constructor([])
    ## wimax-helper.h (module 'wimax'): int64_t ns3::WimaxHelper::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')])
    ## wimax-helper.h (module 'wimax'): int64_t ns3::WimaxHelper::AssignStreams(ns3::NetDeviceContainer c, int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('ns3::NetDeviceContainer', 'c'), param('int64_t', 'stream')])
    ## wimax-helper.h (module 'wimax'): ns3::Ptr<ns3::BSScheduler> ns3::WimaxHelper::CreateBSScheduler(ns3::WimaxHelper::SchedulerType schedulerType) [member function]
    cls.add_method('CreateBSScheduler', 
                   'ns3::Ptr< ns3::BSScheduler >', 
                   [param('ns3::WimaxHelper::SchedulerType', 'schedulerType')])
    ## wimax-helper.h (module 'wimax'): ns3::Ptr<ns3::WimaxPhy> ns3::WimaxHelper::CreatePhy(ns3::WimaxHelper::PhyType phyType) [member function]
    cls.add_method('CreatePhy', 
                   'ns3::Ptr< ns3::WimaxPhy >', 
                   [param('ns3::WimaxHelper::PhyType', 'phyType')])
    ## wimax-helper.h (module 'wimax'): ns3::Ptr<ns3::WimaxPhy> ns3::WimaxHelper::CreatePhy(ns3::WimaxHelper::PhyType phyType, char * SNRTraceFilePath, bool activateLoss) [member function]
    cls.add_method('CreatePhy', 
                   'ns3::Ptr< ns3::WimaxPhy >', 
                   [param('ns3::WimaxHelper::PhyType', 'phyType'), param('char *', 'SNRTraceFilePath'), param('bool', 'activateLoss')])
    ## wimax-helper.h (module 'wimax'): ns3::Ptr<ns3::WimaxPhy> ns3::WimaxHelper::CreatePhyWithoutChannel(ns3::WimaxHelper::PhyType phyType) [member function]
    cls.add_method('CreatePhyWithoutChannel', 
                   'ns3::Ptr< ns3::WimaxPhy >', 
                   [param('ns3::WimaxHelper::PhyType', 'phyType')])
    ## wimax-helper.h (module 'wimax'): ns3::Ptr<ns3::WimaxPhy> ns3::WimaxHelper::CreatePhyWithoutChannel(ns3::WimaxHelper::PhyType phyType, char * SNRTraceFilePath, bool activateLoss) [member function]
    cls.add_method('CreatePhyWithoutChannel', 
                   'ns3::Ptr< ns3::WimaxPhy >', 
                   [param('ns3::WimaxHelper::PhyType', 'phyType'), param('char *', 'SNRTraceFilePath'), param('bool', 'activateLoss')])
    ## wimax-helper.h (module 'wimax'): ns3::ServiceFlow ns3::WimaxHelper::CreateServiceFlow(ns3::ServiceFlow::Direction direction, ns3::ServiceFlow::SchedulingType schedulinType, ns3::IpcsClassifierRecord classifier) [member function]
    cls.add_method('CreateServiceFlow', 
                   'ns3::ServiceFlow', 
                   [param('ns3::ServiceFlow::Direction', 'direction'), param('ns3::ServiceFlow::SchedulingType', 'schedulinType'), param('ns3::IpcsClassifierRecord', 'classifier')])
    ## wimax-helper.h (module 'wimax'): ns3::Ptr<ns3::UplinkScheduler> ns3::WimaxHelper::CreateUplinkScheduler(ns3::WimaxHelper::SchedulerType schedulerType) [member function]
    cls.add_method('CreateUplinkScheduler', 
                   'ns3::Ptr< ns3::UplinkScheduler >', 
                   [param('ns3::WimaxHelper::SchedulerType', 'schedulerType')])
    ## wimax-helper.h (module 'wimax'): static void ns3::WimaxHelper::EnableAsciiForConnection(ns3::Ptr<ns3::OutputStreamWrapper> oss, uint32_t nodeid, uint32_t deviceid, char * netdevice, char * connection) [member function]
    cls.add_method('EnableAsciiForConnection', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'oss'), param('uint32_t', 'nodeid'), param('uint32_t', 'deviceid'), param('char *', 'netdevice'), param('char *', 'connection')], 
                   is_static=True)
    ## wimax-helper.h (module 'wimax'): static void ns3::WimaxHelper::EnableLogComponents() [member function]
    cls.add_method('EnableLogComponents', 
                   'void', 
                   [], 
                   is_static=True)
    ## wimax-helper.h (module 'wimax'): ns3::NetDeviceContainer ns3::WimaxHelper::Install(ns3::NodeContainer c, ns3::WimaxHelper::NetDeviceType type, ns3::WimaxHelper::PhyType phyType, ns3::WimaxHelper::SchedulerType schedulerType) [member function]
    cls.add_method('Install', 
                   'ns3::NetDeviceContainer', 
                   [param('ns3::NodeContainer', 'c'), param('ns3::WimaxHelper::NetDeviceType', 'type'), param('ns3::WimaxHelper::PhyType', 'phyType'), param('ns3::WimaxHelper::SchedulerType', 'schedulerType')])
    ## wimax-helper.h (module 'wimax'): ns3::NetDeviceContainer ns3::WimaxHelper::Install(ns3::NodeContainer c, ns3::WimaxHelper::NetDeviceType deviceType, ns3::WimaxHelper::PhyType phyType, ns3::Ptr<ns3::WimaxChannel> channel, ns3::WimaxHelper::SchedulerType schedulerType) [member function]
    cls.add_method('Install', 
                   'ns3::NetDeviceContainer', 
                   [param('ns3::NodeContainer', 'c'), param('ns3::WimaxHelper::NetDeviceType', 'deviceType'), param('ns3::WimaxHelper::PhyType', 'phyType'), param('ns3::Ptr< ns3::WimaxChannel >', 'channel'), param('ns3::WimaxHelper::SchedulerType', 'schedulerType')])
    ## wimax-helper.h (module 'wimax'): ns3::NetDeviceContainer ns3::WimaxHelper::Install(ns3::NodeContainer c, ns3::WimaxHelper::NetDeviceType deviceType, ns3::WimaxHelper::PhyType phyType, ns3::WimaxHelper::SchedulerType schedulerType, double frameDuration) [member function]
    cls.add_method('Install', 
                   'ns3::NetDeviceContainer', 
                   [param('ns3::NodeContainer', 'c'), param('ns3::WimaxHelper::NetDeviceType', 'deviceType'), param('ns3::WimaxHelper::PhyType', 'phyType'), param('ns3::WimaxHelper::SchedulerType', 'schedulerType'), param('double', 'frameDuration')])
    ## wimax-helper.h (module 'wimax'): ns3::Ptr<ns3::WimaxNetDevice> ns3::WimaxHelper::Install(ns3::Ptr<ns3::Node> node, ns3::WimaxHelper::NetDeviceType deviceType, ns3::WimaxHelper::PhyType phyType, ns3::Ptr<ns3::WimaxChannel> channel, ns3::WimaxHelper::SchedulerType schedulerType) [member function]
    cls.add_method('Install', 
                   'ns3::Ptr< ns3::WimaxNetDevice >', 
                   [param('ns3::Ptr< ns3::Node >', 'node'), param('ns3::WimaxHelper::NetDeviceType', 'deviceType'), param('ns3::WimaxHelper::PhyType', 'phyType'), param('ns3::Ptr< ns3::WimaxChannel >', 'channel'), param('ns3::WimaxHelper::SchedulerType', 'schedulerType')])
    ## wimax-helper.h (module 'wimax'): void ns3::WimaxHelper::SetPropagationLossModel(ns3::SimpleOfdmWimaxChannel::PropModel propagationModel) [member function]
    cls.add_method('SetPropagationLossModel', 
                   'void', 
                   [param('ns3::SimpleOfdmWimaxChannel::PropModel', 'propagationModel')])
    ## wimax-helper.h (module 'wimax'): void ns3::WimaxHelper::EnableAsciiInternal(ns3::Ptr<ns3::OutputStreamWrapper> stream, std::string prefix, ns3::Ptr<ns3::NetDevice> nd, bool explicitFilename) [member function]
    cls.add_method('EnableAsciiInternal', 
                   'void', 
                   [param('ns3::Ptr< ns3::OutputStreamWrapper >', 'stream'), param('std::string', 'prefix'), param('ns3::Ptr< ns3::NetDevice >', 'nd'), param('bool', 'explicitFilename')], 
                   is_virtual=True, visibility='private')
    ## wimax-helper.h (module 'wimax'): void ns3::WimaxHelper::EnablePcapInternal(std::string prefix, ns3::Ptr<ns3::NetDevice> nd, bool explicitFilename, bool promiscuous) [member function]
    cls.add_method('EnablePcapInternal', 
                   'void', 
                   [param('std::string', 'prefix'), param('ns3::Ptr< ns3::NetDevice >', 'nd'), param('bool', 'explicitFilename'), param('bool', 'promiscuous')], 
                   is_virtual=True, visibility='private')
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

def register_Ns3SimpleOfdmSendParam_methods(root_module, cls):
    ## simple-ofdm-send-param.h (module 'wimax'): ns3::simpleOfdmSendParam::simpleOfdmSendParam(ns3::simpleOfdmSendParam const & arg0) [constructor]
    cls.add_constructor([param('ns3::simpleOfdmSendParam const &', 'arg0')])
    ## simple-ofdm-send-param.h (module 'wimax'): ns3::simpleOfdmSendParam::simpleOfdmSendParam() [constructor]
    cls.add_constructor([])
    ## simple-ofdm-send-param.h (module 'wimax'): ns3::simpleOfdmSendParam::simpleOfdmSendParam(ns3::bvec const & fecBlock, uint32_t burstSize, bool isFirstBlock, uint64_t Frequency, ns3::WimaxPhy::ModulationType modulationType, uint8_t direction, double rxPowerDbm) [constructor]
    cls.add_constructor([param('ns3::bvec const &', 'fecBlock'), param('uint32_t', 'burstSize'), param('bool', 'isFirstBlock'), param('uint64_t', 'Frequency'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('uint8_t', 'direction'), param('double', 'rxPowerDbm')])
    ## simple-ofdm-send-param.h (module 'wimax'): ns3::simpleOfdmSendParam::simpleOfdmSendParam(uint32_t burstSize, bool isFirstBlock, uint64_t Frequency, ns3::WimaxPhy::ModulationType modulationType, uint8_t direction, double rxPowerDbm, ns3::Ptr<ns3::PacketBurst> burst) [constructor]
    cls.add_constructor([param('uint32_t', 'burstSize'), param('bool', 'isFirstBlock'), param('uint64_t', 'Frequency'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('uint8_t', 'direction'), param('double', 'rxPowerDbm'), param('ns3::Ptr< ns3::PacketBurst >', 'burst')])
    ## simple-ofdm-send-param.h (module 'wimax'): ns3::Ptr<ns3::PacketBurst> ns3::simpleOfdmSendParam::GetBurst() [member function]
    cls.add_method('GetBurst', 
                   'ns3::Ptr< ns3::PacketBurst >', 
                   [])
    ## simple-ofdm-send-param.h (module 'wimax'): uint32_t ns3::simpleOfdmSendParam::GetBurstSize() [member function]
    cls.add_method('GetBurstSize', 
                   'uint32_t', 
                   [])
    ## simple-ofdm-send-param.h (module 'wimax'): uint8_t ns3::simpleOfdmSendParam::GetDirection() [member function]
    cls.add_method('GetDirection', 
                   'uint8_t', 
                   [])
    ## simple-ofdm-send-param.h (module 'wimax'): ns3::bvec ns3::simpleOfdmSendParam::GetFecBlock() [member function]
    cls.add_method('GetFecBlock', 
                   'ns3::bvec', 
                   [])
    ## simple-ofdm-send-param.h (module 'wimax'): uint64_t ns3::simpleOfdmSendParam::GetFrequency() [member function]
    cls.add_method('GetFrequency', 
                   'uint64_t', 
                   [])
    ## simple-ofdm-send-param.h (module 'wimax'): bool ns3::simpleOfdmSendParam::GetIsFirstBlock() [member function]
    cls.add_method('GetIsFirstBlock', 
                   'bool', 
                   [])
    ## simple-ofdm-send-param.h (module 'wimax'): ns3::WimaxPhy::ModulationType ns3::simpleOfdmSendParam::GetModulationType() [member function]
    cls.add_method('GetModulationType', 
                   'ns3::WimaxPhy::ModulationType', 
                   [])
    ## simple-ofdm-send-param.h (module 'wimax'): double ns3::simpleOfdmSendParam::GetRxPowerDbm() [member function]
    cls.add_method('GetRxPowerDbm', 
                   'double', 
                   [])
    ## simple-ofdm-send-param.h (module 'wimax'): void ns3::simpleOfdmSendParam::SetBurstSize(uint32_t burstSize) [member function]
    cls.add_method('SetBurstSize', 
                   'void', 
                   [param('uint32_t', 'burstSize')])
    ## simple-ofdm-send-param.h (module 'wimax'): void ns3::simpleOfdmSendParam::SetDirection(uint8_t direction) [member function]
    cls.add_method('SetDirection', 
                   'void', 
                   [param('uint8_t', 'direction')])
    ## simple-ofdm-send-param.h (module 'wimax'): void ns3::simpleOfdmSendParam::SetFecBlock(ns3::bvec const & fecBlock) [member function]
    cls.add_method('SetFecBlock', 
                   'void', 
                   [param('ns3::bvec const &', 'fecBlock')])
    ## simple-ofdm-send-param.h (module 'wimax'): void ns3::simpleOfdmSendParam::SetFrequency(uint64_t Frequency) [member function]
    cls.add_method('SetFrequency', 
                   'void', 
                   [param('uint64_t', 'Frequency')])
    ## simple-ofdm-send-param.h (module 'wimax'): void ns3::simpleOfdmSendParam::SetIsFirstBlock(bool isFirstBlock) [member function]
    cls.add_method('SetIsFirstBlock', 
                   'void', 
                   [param('bool', 'isFirstBlock')])
    ## simple-ofdm-send-param.h (module 'wimax'): void ns3::simpleOfdmSendParam::SetModulationType(ns3::WimaxPhy::ModulationType modulationType) [member function]
    cls.add_method('SetModulationType', 
                   'void', 
                   [param('ns3::WimaxPhy::ModulationType', 'modulationType')])
    ## simple-ofdm-send-param.h (module 'wimax'): void ns3::simpleOfdmSendParam::SetRxPowerDbm(double rxPowerDbm) [member function]
    cls.add_method('SetRxPowerDbm', 
                   'void', 
                   [param('double', 'rxPowerDbm')])
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

def register_Ns3ClassificationRuleVectorTlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::ClassificationRuleVectorTlvValue::ClassificationRuleVectorTlvValue(ns3::ClassificationRuleVectorTlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::ClassificationRuleVectorTlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::ClassificationRuleVectorTlvValue::ClassificationRuleVectorTlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::ClassificationRuleVectorTlvValue * ns3::ClassificationRuleVectorTlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::ClassificationRuleVectorTlvValue *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::ClassificationRuleVectorTlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLength) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLength')], 
                   is_virtual=True)
    return

def register_Ns3CsParamVectorTlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::CsParamVectorTlvValue::CsParamVectorTlvValue(ns3::CsParamVectorTlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::CsParamVectorTlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::CsParamVectorTlvValue::CsParamVectorTlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::CsParamVectorTlvValue * ns3::CsParamVectorTlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::CsParamVectorTlvValue *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::CsParamVectorTlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLength) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLength')], 
                   is_virtual=True)
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

def register_Ns3Ipv4AddressTlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue::Ipv4AddressTlvValue(ns3::Ipv4AddressTlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv4AddressTlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue::Ipv4AddressTlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): void ns3::Ipv4AddressTlvValue::Add(ns3::Ipv4Address address, ns3::Ipv4Mask Mask) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::Ipv4Address', 'address'), param('ns3::Ipv4Mask', 'Mask')])
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue::Iterator ns3::Ipv4AddressTlvValue::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::Ipv4AddressTlvValue::Iterator', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue * ns3::Ipv4AddressTlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ipv4AddressTlvValue *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::Ipv4AddressTlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLength) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLength')], 
                   is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue::Iterator ns3::Ipv4AddressTlvValue::End() const [member function]
    cls.add_method('End', 
                   'ns3::Ipv4AddressTlvValue::Iterator', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::Ipv4AddressTlvValue::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): void ns3::Ipv4AddressTlvValue::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3Ipv4AddressTlvValueIpv4Addr_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue::ipv4Addr::ipv4Addr() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue::ipv4Addr::ipv4Addr(ns3::Ipv4AddressTlvValue::ipv4Addr const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ipv4AddressTlvValue::ipv4Addr const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue::ipv4Addr::Address [variable]
    cls.add_instance_attribute('Address', 'ns3::Ipv4Address', is_const=False)
    ## wimax-tlv.h (module 'wimax'): ns3::Ipv4AddressTlvValue::ipv4Addr::Mask [variable]
    cls.add_instance_attribute('Mask', 'ns3::Ipv4Mask', is_const=False)
    return

def register_Ns3MacHeaderType_methods(root_module, cls):
    ## wimax-mac-header.h (module 'wimax'): ns3::MacHeaderType::MacHeaderType(ns3::MacHeaderType const & arg0) [constructor]
    cls.add_constructor([param('ns3::MacHeaderType const &', 'arg0')])
    ## wimax-mac-header.h (module 'wimax'): ns3::MacHeaderType::MacHeaderType() [constructor]
    cls.add_constructor([])
    ## wimax-mac-header.h (module 'wimax'): ns3::MacHeaderType::MacHeaderType(uint8_t type) [constructor]
    cls.add_constructor([param('uint8_t', 'type')])
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::MacHeaderType::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): ns3::TypeId ns3::MacHeaderType::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): std::string ns3::MacHeaderType::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::MacHeaderType::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::MacHeaderType::GetType() const [member function]
    cls.add_method('GetType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): static ns3::TypeId ns3::MacHeaderType::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::MacHeaderType::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::MacHeaderType::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::MacHeaderType::SetType(uint8_t type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('uint8_t', 'type')])
    return

def register_Ns3ManagementMessageType_methods(root_module, cls):
    ## mac-messages.h (module 'wimax'): ns3::ManagementMessageType::ManagementMessageType(ns3::ManagementMessageType const & arg0) [constructor]
    cls.add_constructor([param('ns3::ManagementMessageType const &', 'arg0')])
    ## mac-messages.h (module 'wimax'): ns3::ManagementMessageType::ManagementMessageType() [constructor]
    cls.add_constructor([])
    ## mac-messages.h (module 'wimax'): ns3::ManagementMessageType::ManagementMessageType(uint8_t type) [constructor]
    cls.add_constructor([param('uint8_t', 'type')])
    ## mac-messages.h (module 'wimax'): uint32_t ns3::ManagementMessageType::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mac-messages.h (module 'wimax'): ns3::TypeId ns3::ManagementMessageType::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): std::string ns3::ManagementMessageType::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::ManagementMessageType::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): uint8_t ns3::ManagementMessageType::GetType() const [member function]
    cls.add_method('GetType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): static ns3::TypeId ns3::ManagementMessageType::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mac-messages.h (module 'wimax'): void ns3::ManagementMessageType::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::ManagementMessageType::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::ManagementMessageType::SetType(uint8_t type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('uint8_t', 'type')])
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

def register_Ns3OfdmDownlinkFramePrefix_methods(root_module, cls):
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): ns3::OfdmDownlinkFramePrefix::OfdmDownlinkFramePrefix(ns3::OfdmDownlinkFramePrefix const & arg0) [constructor]
    cls.add_constructor([param('ns3::OfdmDownlinkFramePrefix const &', 'arg0')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): ns3::OfdmDownlinkFramePrefix::OfdmDownlinkFramePrefix() [constructor]
    cls.add_constructor([])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::OfdmDownlinkFramePrefix::AddDlFramePrefixElement(ns3::DlFramePrefixIe dlFramePrefixElement) [member function]
    cls.add_method('AddDlFramePrefixElement', 
                   'void', 
                   [param('ns3::DlFramePrefixIe', 'dlFramePrefixElement')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint32_t ns3::OfdmDownlinkFramePrefix::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): ns3::Mac48Address ns3::OfdmDownlinkFramePrefix::GetBaseStationId() const [member function]
    cls.add_method('GetBaseStationId', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint8_t ns3::OfdmDownlinkFramePrefix::GetConfigurationChangeCount() const [member function]
    cls.add_method('GetConfigurationChangeCount', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): std::vector<ns3::DlFramePrefixIe, std::allocator<ns3::DlFramePrefixIe>> ns3::OfdmDownlinkFramePrefix::GetDlFramePrefixElements() const [member function]
    cls.add_method('GetDlFramePrefixElements', 
                   'std::vector< ns3::DlFramePrefixIe >', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint32_t ns3::OfdmDownlinkFramePrefix::GetFrameNumber() const [member function]
    cls.add_method('GetFrameNumber', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint8_t ns3::OfdmDownlinkFramePrefix::GetHcs() const [member function]
    cls.add_method('GetHcs', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): std::string ns3::OfdmDownlinkFramePrefix::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): uint32_t ns3::OfdmDownlinkFramePrefix::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): static ns3::TypeId ns3::OfdmDownlinkFramePrefix::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::OfdmDownlinkFramePrefix::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::OfdmDownlinkFramePrefix::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::OfdmDownlinkFramePrefix::SetBaseStationId(ns3::Mac48Address baseStationId) [member function]
    cls.add_method('SetBaseStationId', 
                   'void', 
                   [param('ns3::Mac48Address', 'baseStationId')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::OfdmDownlinkFramePrefix::SetConfigurationChangeCount(uint8_t configurationChangeCount) [member function]
    cls.add_method('SetConfigurationChangeCount', 
                   'void', 
                   [param('uint8_t', 'configurationChangeCount')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::OfdmDownlinkFramePrefix::SetFrameNumber(uint32_t frameNumber) [member function]
    cls.add_method('SetFrameNumber', 
                   'void', 
                   [param('uint32_t', 'frameNumber')])
    ## ofdm-downlink-frame-prefix.h (module 'wimax'): void ns3::OfdmDownlinkFramePrefix::SetHcs(uint8_t hcs) [member function]
    cls.add_method('SetHcs', 
                   'void', 
                   [param('uint8_t', 'hcs')])
    return

def register_Ns3OfdmSendParams_methods(root_module, cls):
    ## send-params.h (module 'wimax'): ns3::OfdmSendParams::OfdmSendParams(ns3::OfdmSendParams const & arg0) [constructor]
    cls.add_constructor([param('ns3::OfdmSendParams const &', 'arg0')])
    ## send-params.h (module 'wimax'): ns3::OfdmSendParams::OfdmSendParams(ns3::Ptr<ns3::PacketBurst> burst, uint8_t modulationType, uint8_t direction) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::PacketBurst >', 'burst'), param('uint8_t', 'modulationType'), param('uint8_t', 'direction')])
    ## send-params.h (module 'wimax'): ns3::Ptr<ns3::PacketBurst> ns3::OfdmSendParams::GetBurst() const [member function]
    cls.add_method('GetBurst', 
                   'ns3::Ptr< ns3::PacketBurst >', 
                   [], 
                   is_const=True)
    ## send-params.h (module 'wimax'): uint8_t ns3::OfdmSendParams::GetDirection() const [member function]
    cls.add_method('GetDirection', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## send-params.h (module 'wimax'): uint8_t ns3::OfdmSendParams::GetModulationType() const [member function]
    cls.add_method('GetModulationType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    return

def register_Ns3OfdmUcdChannelEncodings_methods(root_module, cls):
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUcdChannelEncodings::OfdmUcdChannelEncodings(ns3::OfdmUcdChannelEncodings const & arg0) [constructor]
    cls.add_constructor([param('ns3::OfdmUcdChannelEncodings const &', 'arg0')])
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUcdChannelEncodings::OfdmUcdChannelEncodings() [constructor]
    cls.add_constructor([])
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmUcdChannelEncodings::GetSbchnlFocContCodes() const [member function]
    cls.add_method('GetSbchnlFocContCodes', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::OfdmUcdChannelEncodings::GetSbchnlReqRegionFullParams() const [member function]
    cls.add_method('GetSbchnlReqRegionFullParams', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint16_t ns3::OfdmUcdChannelEncodings::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUcdChannelEncodings::SetSbchnlFocContCodes(uint8_t sbchnlFocContCodes) [member function]
    cls.add_method('SetSbchnlFocContCodes', 
                   'void', 
                   [param('uint8_t', 'sbchnlFocContCodes')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::OfdmUcdChannelEncodings::SetSbchnlReqRegionFullParams(uint8_t sbchnlReqRegionFullParams) [member function]
    cls.add_method('SetSbchnlReqRegionFullParams', 
                   'void', 
                   [param('uint8_t', 'sbchnlReqRegionFullParams')])
    ## ul-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmUcdChannelEncodings::DoRead(ns3::Buffer::Iterator start) [member function]
    cls.add_method('DoRead', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True, visibility='private')
    ## ul-mac-messages.h (module 'wimax'): ns3::Buffer::Iterator ns3::OfdmUcdChannelEncodings::DoWrite(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('DoWrite', 
                   'ns3::Buffer::Iterator', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True, visibility='private')
    return

def register_Ns3PacketBurst_methods(root_module, cls):
    ## packet-burst.h (module 'network'): ns3::PacketBurst::PacketBurst(ns3::PacketBurst const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketBurst const &', 'arg0')])
    ## packet-burst.h (module 'network'): ns3::PacketBurst::PacketBurst() [constructor]
    cls.add_constructor([])
    ## packet-burst.h (module 'network'): void ns3::PacketBurst::AddPacket(ns3::Ptr<ns3::Packet> packet) [member function]
    cls.add_method('AddPacket', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet')])
    ## packet-burst.h (module 'network'): std::list<ns3::Ptr<ns3::Packet>, std::allocator<ns3::Ptr<ns3::Packet>>>::const_iterator ns3::PacketBurst::Begin() const [member function]
    cls.add_method('Begin', 
                   'std::list< ns3::Ptr< ns3::Packet > > const_iterator', 
                   [], 
                   is_const=True)
    ## packet-burst.h (module 'network'): ns3::Ptr<ns3::PacketBurst> ns3::PacketBurst::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::PacketBurst >', 
                   [], 
                   is_const=True)
    ## packet-burst.h (module 'network'): std::list<ns3::Ptr<ns3::Packet>, std::allocator<ns3::Ptr<ns3::Packet>>>::const_iterator ns3::PacketBurst::End() const [member function]
    cls.add_method('End', 
                   'std::list< ns3::Ptr< ns3::Packet > > const_iterator', 
                   [], 
                   is_const=True)
    ## packet-burst.h (module 'network'): uint32_t ns3::PacketBurst::GetNPackets() const [member function]
    cls.add_method('GetNPackets', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packet-burst.h (module 'network'): std::list<ns3::Ptr<ns3::Packet>, std::allocator<ns3::Ptr<ns3::Packet>>> ns3::PacketBurst::GetPackets() const [member function]
    cls.add_method('GetPackets', 
                   'std::list< ns3::Ptr< ns3::Packet > >', 
                   [], 
                   is_const=True)
    ## packet-burst.h (module 'network'): uint32_t ns3::PacketBurst::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packet-burst.h (module 'network'): static ns3::TypeId ns3::PacketBurst::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## packet-burst.h (module 'network'): void ns3::PacketBurst::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3PcapFileWrapper_methods(root_module, cls):
    ## pcap-file-wrapper.h (module 'network'): static ns3::TypeId ns3::PcapFileWrapper::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## pcap-file-wrapper.h (module 'network'): ns3::PcapFileWrapper::PcapFileWrapper() [constructor]
    cls.add_constructor([])
    ## pcap-file-wrapper.h (module 'network'): bool ns3::PcapFileWrapper::Fail() const [member function]
    cls.add_method('Fail', 
                   'bool', 
                   [], 
                   is_const=True)
    ## pcap-file-wrapper.h (module 'network'): bool ns3::PcapFileWrapper::Eof() const [member function]
    cls.add_method('Eof', 
                   'bool', 
                   [], 
                   is_const=True)
    ## pcap-file-wrapper.h (module 'network'): void ns3::PcapFileWrapper::Clear() [member function]
    cls.add_method('Clear', 
                   'void', 
                   [])
    ## pcap-file-wrapper.h (module 'network'): void ns3::PcapFileWrapper::Open(std::string const & filename, std::ios_base::openmode mode) [member function]
    cls.add_method('Open', 
                   'void', 
                   [param('std::string const &', 'filename'), param('std::ios_base::openmode', 'mode')])
    ## pcap-file-wrapper.h (module 'network'): void ns3::PcapFileWrapper::Close() [member function]
    cls.add_method('Close', 
                   'void', 
                   [])
    ## pcap-file-wrapper.h (module 'network'): void ns3::PcapFileWrapper::Init(uint32_t dataLinkType, uint32_t snapLen=std::numeric_limits<unsigned int>::max(), int32_t tzCorrection=ns3::PcapFile::ZONE_DEFAULT) [member function]
    cls.add_method('Init', 
                   'void', 
                   [param('uint32_t', 'dataLinkType'), param('uint32_t', 'snapLen', default_value='std::numeric_limits<unsigned int>::max()'), param('int32_t', 'tzCorrection', default_value='ns3::PcapFile::ZONE_DEFAULT')])
    ## pcap-file-wrapper.h (module 'network'): void ns3::PcapFileWrapper::Write(ns3::Time t, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('Write', 
                   'void', 
                   [param('ns3::Time', 't'), param('ns3::Ptr< ns3::Packet const >', 'p')])
    ## pcap-file-wrapper.h (module 'network'): void ns3::PcapFileWrapper::Write(ns3::Time t, ns3::Header const & header, ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('Write', 
                   'void', 
                   [param('ns3::Time', 't'), param('ns3::Header const &', 'header'), param('ns3::Ptr< ns3::Packet const >', 'p')])
    ## pcap-file-wrapper.h (module 'network'): void ns3::PcapFileWrapper::Write(ns3::Time t, uint8_t const * buffer, uint32_t length) [member function]
    cls.add_method('Write', 
                   'void', 
                   [param('ns3::Time', 't'), param('uint8_t const *', 'buffer'), param('uint32_t', 'length')])
    ## pcap-file-wrapper.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::PcapFileWrapper::Read(ns3::Time & t) [member function]
    cls.add_method('Read', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('ns3::Time &', 't')])
    ## pcap-file-wrapper.h (module 'network'): uint32_t ns3::PcapFileWrapper::GetMagic() [member function]
    cls.add_method('GetMagic', 
                   'uint32_t', 
                   [])
    ## pcap-file-wrapper.h (module 'network'): uint16_t ns3::PcapFileWrapper::GetVersionMajor() [member function]
    cls.add_method('GetVersionMajor', 
                   'uint16_t', 
                   [])
    ## pcap-file-wrapper.h (module 'network'): uint16_t ns3::PcapFileWrapper::GetVersionMinor() [member function]
    cls.add_method('GetVersionMinor', 
                   'uint16_t', 
                   [])
    ## pcap-file-wrapper.h (module 'network'): int32_t ns3::PcapFileWrapper::GetTimeZoneOffset() [member function]
    cls.add_method('GetTimeZoneOffset', 
                   'int32_t', 
                   [])
    ## pcap-file-wrapper.h (module 'network'): uint32_t ns3::PcapFileWrapper::GetSigFigs() [member function]
    cls.add_method('GetSigFigs', 
                   'uint32_t', 
                   [])
    ## pcap-file-wrapper.h (module 'network'): uint32_t ns3::PcapFileWrapper::GetSnapLen() [member function]
    cls.add_method('GetSnapLen', 
                   'uint32_t', 
                   [])
    ## pcap-file-wrapper.h (module 'network'): uint32_t ns3::PcapFileWrapper::GetDataLinkType() [member function]
    cls.add_method('GetDataLinkType', 
                   'uint32_t', 
                   [])
    return

def register_Ns3PortRangeTlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue::PortRangeTlvValue(ns3::PortRangeTlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::PortRangeTlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue::PortRangeTlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): void ns3::PortRangeTlvValue::Add(uint16_t portLow, uint16_t portHigh) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('uint16_t', 'portLow'), param('uint16_t', 'portHigh')])
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue::Iterator ns3::PortRangeTlvValue::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::PortRangeTlvValue::Iterator', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue * ns3::PortRangeTlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::PortRangeTlvValue *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::PortRangeTlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLength) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLength')], 
                   is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue::Iterator ns3::PortRangeTlvValue::End() const [member function]
    cls.add_method('End', 
                   'ns3::PortRangeTlvValue::Iterator', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::PortRangeTlvValue::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): void ns3::PortRangeTlvValue::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3PortRangeTlvValuePortRange_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue::PortRange::PortRange() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue::PortRange::PortRange(ns3::PortRangeTlvValue::PortRange const & arg0) [constructor]
    cls.add_constructor([param('ns3::PortRangeTlvValue::PortRange const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue::PortRange::PortHigh [variable]
    cls.add_instance_attribute('PortHigh', 'uint16_t', is_const=False)
    ## wimax-tlv.h (module 'wimax'): ns3::PortRangeTlvValue::PortRange::PortLow [variable]
    cls.add_instance_attribute('PortLow', 'uint16_t', is_const=False)
    return

def register_Ns3PriorityUlJob_methods(root_module, cls):
    ## ul-job.h (module 'wimax'): ns3::PriorityUlJob::PriorityUlJob(ns3::PriorityUlJob const & arg0) [constructor]
    cls.add_constructor([param('ns3::PriorityUlJob const &', 'arg0')])
    ## ul-job.h (module 'wimax'): ns3::PriorityUlJob::PriorityUlJob() [constructor]
    cls.add_constructor([])
    ## ul-job.h (module 'wimax'): int ns3::PriorityUlJob::GetPriority() [member function]
    cls.add_method('GetPriority', 
                   'int', 
                   [])
    ## ul-job.h (module 'wimax'): ns3::Ptr<ns3::UlJob> ns3::PriorityUlJob::GetUlJob() [member function]
    cls.add_method('GetUlJob', 
                   'ns3::Ptr< ns3::UlJob >', 
                   [])
    ## ul-job.h (module 'wimax'): void ns3::PriorityUlJob::SetPriority(int priority) [member function]
    cls.add_method('SetPriority', 
                   'void', 
                   [param('int', 'priority')])
    ## ul-job.h (module 'wimax'): void ns3::PriorityUlJob::SetUlJob(ns3::Ptr<ns3::UlJob> job) [member function]
    cls.add_method('SetUlJob', 
                   'void', 
                   [param('ns3::Ptr< ns3::UlJob >', 'job')])
    return

def register_Ns3PropagationLossModel_methods(root_module, cls):
    ## propagation-loss-model.h (module 'propagation'): static ns3::TypeId ns3::PropagationLossModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## propagation-loss-model.h (module 'propagation'): ns3::PropagationLossModel::PropagationLossModel() [constructor]
    cls.add_constructor([])
    ## propagation-loss-model.h (module 'propagation'): void ns3::PropagationLossModel::SetNext(ns3::Ptr<ns3::PropagationLossModel> next) [member function]
    cls.add_method('SetNext', 
                   'void', 
                   [param('ns3::Ptr< ns3::PropagationLossModel >', 'next')])
    ## propagation-loss-model.h (module 'propagation'): ns3::Ptr<ns3::PropagationLossModel> ns3::PropagationLossModel::GetNext() [member function]
    cls.add_method('GetNext', 
                   'ns3::Ptr< ns3::PropagationLossModel >', 
                   [])
    ## propagation-loss-model.h (module 'propagation'): double ns3::PropagationLossModel::CalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('CalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True)
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::PropagationLossModel::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')])
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::PropagationLossModel::DoAssignStreams(int64_t stream) [member function]
    cls.add_method('DoAssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_pure_virtual=True, is_virtual=True, visibility='protected')
    ## propagation-loss-model.h (module 'propagation'): double ns3::PropagationLossModel::DoCalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('DoCalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    return

def register_Ns3ProtocolTlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::ProtocolTlvValue::ProtocolTlvValue(ns3::ProtocolTlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::ProtocolTlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::ProtocolTlvValue::ProtocolTlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): void ns3::ProtocolTlvValue::Add(uint8_t protocol) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('uint8_t', 'protocol')])
    ## wimax-tlv.h (module 'wimax'): ns3::ProtocolTlvValue::Iterator ns3::ProtocolTlvValue::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::ProtocolTlvValue::Iterator', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): ns3::ProtocolTlvValue * ns3::ProtocolTlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::ProtocolTlvValue *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::ProtocolTlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLength) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLength')], 
                   is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): ns3::ProtocolTlvValue::Iterator ns3::ProtocolTlvValue::End() const [member function]
    cls.add_method('End', 
                   'ns3::ProtocolTlvValue::Iterator', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::ProtocolTlvValue::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): void ns3::ProtocolTlvValue::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3RandomPropagationLossModel_methods(root_module, cls):
    ## propagation-loss-model.h (module 'propagation'): static ns3::TypeId ns3::RandomPropagationLossModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## propagation-loss-model.h (module 'propagation'): ns3::RandomPropagationLossModel::RandomPropagationLossModel() [constructor]
    cls.add_constructor([])
    ## propagation-loss-model.h (module 'propagation'): double ns3::RandomPropagationLossModel::DoCalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('DoCalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::RandomPropagationLossModel::DoAssignStreams(int64_t stream) [member function]
    cls.add_method('DoAssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True, visibility='private')
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

def register_Ns3RangePropagationLossModel_methods(root_module, cls):
    ## propagation-loss-model.h (module 'propagation'): static ns3::TypeId ns3::RangePropagationLossModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## propagation-loss-model.h (module 'propagation'): ns3::RangePropagationLossModel::RangePropagationLossModel() [constructor]
    cls.add_constructor([])
    ## propagation-loss-model.h (module 'propagation'): double ns3::RangePropagationLossModel::DoCalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('DoCalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::RangePropagationLossModel::DoAssignStreams(int64_t stream) [member function]
    cls.add_method('DoAssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3RngReq_methods(root_module, cls):
    ## mac-messages.h (module 'wimax'): ns3::RngReq::RngReq(ns3::RngReq const & arg0) [constructor]
    cls.add_constructor([param('ns3::RngReq const &', 'arg0')])
    ## mac-messages.h (module 'wimax'): ns3::RngReq::RngReq() [constructor]
    cls.add_constructor([])
    ## mac-messages.h (module 'wimax'): uint32_t ns3::RngReq::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mac-messages.h (module 'wimax'): ns3::TypeId ns3::RngReq::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): ns3::Mac48Address ns3::RngReq::GetMacAddress() const [member function]
    cls.add_method('GetMacAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): std::string ns3::RngReq::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint8_t ns3::RngReq::GetRangingAnomalies() const [member function]
    cls.add_method('GetRangingAnomalies', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint8_t ns3::RngReq::GetReqDlBurstProfile() const [member function]
    cls.add_method('GetReqDlBurstProfile', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::RngReq::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): static ns3::TypeId ns3::RngReq::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mac-messages.h (module 'wimax'): void ns3::RngReq::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::RngReq::PrintDebug() const [member function]
    cls.add_method('PrintDebug', 
                   'void', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): void ns3::RngReq::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::RngReq::SetMacAddress(ns3::Mac48Address macAddress) [member function]
    cls.add_method('SetMacAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'macAddress')])
    ## mac-messages.h (module 'wimax'): void ns3::RngReq::SetRangingAnomalies(uint8_t rangingAnomalies) [member function]
    cls.add_method('SetRangingAnomalies', 
                   'void', 
                   [param('uint8_t', 'rangingAnomalies')])
    ## mac-messages.h (module 'wimax'): void ns3::RngReq::SetReqDlBurstProfile(uint8_t reqDlBurstProfile) [member function]
    cls.add_method('SetReqDlBurstProfile', 
                   'void', 
                   [param('uint8_t', 'reqDlBurstProfile')])
    return

def register_Ns3RngRsp_methods(root_module, cls):
    ## mac-messages.h (module 'wimax'): ns3::RngRsp::RngRsp(ns3::RngRsp const & arg0) [constructor]
    cls.add_constructor([param('ns3::RngRsp const &', 'arg0')])
    ## mac-messages.h (module 'wimax'): ns3::RngRsp::RngRsp() [constructor]
    cls.add_constructor([])
    ## mac-messages.h (module 'wimax'): uint32_t ns3::RngRsp::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mac-messages.h (module 'wimax'): uint8_t ns3::RngRsp::GetAasBdcastPermission() const [member function]
    cls.add_method('GetAasBdcastPermission', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): ns3::Cid ns3::RngRsp::GetBasicCid() const [member function]
    cls.add_method('GetBasicCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::RngRsp::GetDlFreqOverride() const [member function]
    cls.add_method('GetDlFreqOverride', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint16_t ns3::RngRsp::GetDlOperBurstProfile() const [member function]
    cls.add_method('GetDlOperBurstProfile', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::RngRsp::GetFrameNumber() const [member function]
    cls.add_method('GetFrameNumber', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint8_t ns3::RngRsp::GetInitRangOppNumber() const [member function]
    cls.add_method('GetInitRangOppNumber', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): ns3::TypeId ns3::RngRsp::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): ns3::Mac48Address ns3::RngRsp::GetMacAddress() const [member function]
    cls.add_method('GetMacAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): std::string ns3::RngRsp::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::RngRsp::GetOffsetFreqAdjust() const [member function]
    cls.add_method('GetOffsetFreqAdjust', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint8_t ns3::RngRsp::GetPowerLevelAdjust() const [member function]
    cls.add_method('GetPowerLevelAdjust', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): ns3::Cid ns3::RngRsp::GetPrimaryCid() const [member function]
    cls.add_method('GetPrimaryCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint8_t ns3::RngRsp::GetRangStatus() const [member function]
    cls.add_method('GetRangStatus', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint8_t ns3::RngRsp::GetRangSubchnl() const [member function]
    cls.add_method('GetRangSubchnl', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::RngRsp::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::RngRsp::GetTimingAdjust() const [member function]
    cls.add_method('GetTimingAdjust', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): static ns3::TypeId ns3::RngRsp::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mac-messages.h (module 'wimax'): uint8_t ns3::RngRsp::GetUlChnlIdOverride() const [member function]
    cls.add_method('GetUlChnlIdOverride', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetAasBdcastPermission(uint8_t aasBdcastPermission) [member function]
    cls.add_method('SetAasBdcastPermission', 
                   'void', 
                   [param('uint8_t', 'aasBdcastPermission')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetBasicCid(ns3::Cid basicCid) [member function]
    cls.add_method('SetBasicCid', 
                   'void', 
                   [param('ns3::Cid', 'basicCid')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetDlFreqOverride(uint32_t dlFreqOverride) [member function]
    cls.add_method('SetDlFreqOverride', 
                   'void', 
                   [param('uint32_t', 'dlFreqOverride')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetDlOperBurstProfile(uint16_t dlOperBurstProfile) [member function]
    cls.add_method('SetDlOperBurstProfile', 
                   'void', 
                   [param('uint16_t', 'dlOperBurstProfile')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetFrameNumber(uint32_t frameNumber) [member function]
    cls.add_method('SetFrameNumber', 
                   'void', 
                   [param('uint32_t', 'frameNumber')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetInitRangOppNumber(uint8_t initRangOppNumber) [member function]
    cls.add_method('SetInitRangOppNumber', 
                   'void', 
                   [param('uint8_t', 'initRangOppNumber')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetMacAddress(ns3::Mac48Address macAddress) [member function]
    cls.add_method('SetMacAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'macAddress')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetOffsetFreqAdjust(uint32_t offsetFreqAdjust) [member function]
    cls.add_method('SetOffsetFreqAdjust', 
                   'void', 
                   [param('uint32_t', 'offsetFreqAdjust')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetPowerLevelAdjust(uint8_t powerLevelAdjust) [member function]
    cls.add_method('SetPowerLevelAdjust', 
                   'void', 
                   [param('uint8_t', 'powerLevelAdjust')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetPrimaryCid(ns3::Cid primaryCid) [member function]
    cls.add_method('SetPrimaryCid', 
                   'void', 
                   [param('ns3::Cid', 'primaryCid')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetRangStatus(uint8_t rangStatus) [member function]
    cls.add_method('SetRangStatus', 
                   'void', 
                   [param('uint8_t', 'rangStatus')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetRangSubchnl(uint8_t rangSubchnl) [member function]
    cls.add_method('SetRangSubchnl', 
                   'void', 
                   [param('uint8_t', 'rangSubchnl')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetTimingAdjust(uint32_t timingAdjust) [member function]
    cls.add_method('SetTimingAdjust', 
                   'void', 
                   [param('uint32_t', 'timingAdjust')])
    ## mac-messages.h (module 'wimax'): void ns3::RngRsp::SetUlChnlIdOverride(uint8_t ulChnlIdOverride) [member function]
    cls.add_method('SetUlChnlIdOverride', 
                   'void', 
                   [param('uint8_t', 'ulChnlIdOverride')])
    return

def register_Ns3SSManager_methods(root_module, cls):
    ## ss-manager.h (module 'wimax'): ns3::SSManager::SSManager(ns3::SSManager const & arg0) [constructor]
    cls.add_constructor([param('ns3::SSManager const &', 'arg0')])
    ## ss-manager.h (module 'wimax'): ns3::SSManager::SSManager() [constructor]
    cls.add_constructor([])
    ## ss-manager.h (module 'wimax'): ns3::SSRecord * ns3::SSManager::CreateSSRecord(ns3::Mac48Address const & macAddress) [member function]
    cls.add_method('CreateSSRecord', 
                   'ns3::SSRecord *', 
                   [param('ns3::Mac48Address const &', 'macAddress')])
    ## ss-manager.h (module 'wimax'): void ns3::SSManager::DeleteSSRecord(ns3::Cid cid) [member function]
    cls.add_method('DeleteSSRecord', 
                   'void', 
                   [param('ns3::Cid', 'cid')])
    ## ss-manager.h (module 'wimax'): ns3::Mac48Address ns3::SSManager::GetMacAddress(ns3::Cid cid) const [member function]
    cls.add_method('GetMacAddress', 
                   'ns3::Mac48Address', 
                   [param('ns3::Cid', 'cid')], 
                   is_const=True)
    ## ss-manager.h (module 'wimax'): uint32_t ns3::SSManager::GetNRegisteredSSs() const [member function]
    cls.add_method('GetNRegisteredSSs', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ss-manager.h (module 'wimax'): uint32_t ns3::SSManager::GetNSSs() const [member function]
    cls.add_method('GetNSSs', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ss-manager.h (module 'wimax'): ns3::SSRecord * ns3::SSManager::GetSSRecord(ns3::Mac48Address const & macAddress) const [member function]
    cls.add_method('GetSSRecord', 
                   'ns3::SSRecord *', 
                   [param('ns3::Mac48Address const &', 'macAddress')], 
                   is_const=True)
    ## ss-manager.h (module 'wimax'): ns3::SSRecord * ns3::SSManager::GetSSRecord(ns3::Cid cid) const [member function]
    cls.add_method('GetSSRecord', 
                   'ns3::SSRecord *', 
                   [param('ns3::Cid', 'cid')], 
                   is_const=True)
    ## ss-manager.h (module 'wimax'): std::vector<ns3::SSRecord *, std::allocator<ns3::SSRecord *>> * ns3::SSManager::GetSSRecords() const [member function]
    cls.add_method('GetSSRecords', 
                   'std::vector< ns3::SSRecord * > *', 
                   [], 
                   is_const=True)
    ## ss-manager.h (module 'wimax'): static ns3::TypeId ns3::SSManager::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## ss-manager.h (module 'wimax'): bool ns3::SSManager::IsInRecord(ns3::Mac48Address const & macAddress) const [member function]
    cls.add_method('IsInRecord', 
                   'bool', 
                   [param('ns3::Mac48Address const &', 'macAddress')], 
                   is_const=True)
    ## ss-manager.h (module 'wimax'): bool ns3::SSManager::IsRegistered(ns3::Mac48Address const & macAddress) const [member function]
    cls.add_method('IsRegistered', 
                   'bool', 
                   [param('ns3::Mac48Address const &', 'macAddress')], 
                   is_const=True)
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

def register_Ns3ServiceFlowManager_methods(root_module, cls):
    ## service-flow-manager.h (module 'wimax'): ns3::ServiceFlowManager::ServiceFlowManager(ns3::ServiceFlowManager const & arg0) [constructor]
    cls.add_constructor([param('ns3::ServiceFlowManager const &', 'arg0')])
    ## service-flow-manager.h (module 'wimax'): ns3::ServiceFlowManager::ServiceFlowManager() [constructor]
    cls.add_constructor([])
    ## service-flow-manager.h (module 'wimax'): void ns3::ServiceFlowManager::AddServiceFlow(ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('AddServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow *', 'serviceFlow')])
    ## service-flow-manager.h (module 'wimax'): bool ns3::ServiceFlowManager::AreServiceFlowsAllocated() [member function]
    cls.add_method('AreServiceFlowsAllocated', 
                   'bool', 
                   [])
    ## service-flow-manager.h (module 'wimax'): bool ns3::ServiceFlowManager::AreServiceFlowsAllocated(std::vector<ns3::ServiceFlow *, std::allocator<ns3::ServiceFlow *>> * serviceFlows) [member function]
    cls.add_method('AreServiceFlowsAllocated', 
                   'bool', 
                   [param('std::vector< ns3::ServiceFlow * > *', 'serviceFlows')])
    ## service-flow-manager.h (module 'wimax'): bool ns3::ServiceFlowManager::AreServiceFlowsAllocated(std::vector<ns3::ServiceFlow *, std::allocator<ns3::ServiceFlow *>> serviceFlows) [member function]
    cls.add_method('AreServiceFlowsAllocated', 
                   'bool', 
                   [param('std::vector< ns3::ServiceFlow * >', 'serviceFlows')])
    ## service-flow-manager.h (module 'wimax'): ns3::ServiceFlow * ns3::ServiceFlowManager::DoClassify(ns3::Ipv4Address SrcAddress, ns3::Ipv4Address DstAddress, uint16_t SrcPort, uint16_t DstPort, uint8_t Proto, ns3::ServiceFlow::Direction dir) const [member function]
    cls.add_method('DoClassify', 
                   'ns3::ServiceFlow *', 
                   [param('ns3::Ipv4Address', 'SrcAddress'), param('ns3::Ipv4Address', 'DstAddress'), param('uint16_t', 'SrcPort'), param('uint16_t', 'DstPort'), param('uint8_t', 'Proto'), param('ns3::ServiceFlow::Direction', 'dir')], 
                   is_const=True)
    ## service-flow-manager.h (module 'wimax'): void ns3::ServiceFlowManager::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## service-flow-manager.h (module 'wimax'): ns3::ServiceFlow * ns3::ServiceFlowManager::GetNextServiceFlowToAllocate() [member function]
    cls.add_method('GetNextServiceFlowToAllocate', 
                   'ns3::ServiceFlow *', 
                   [])
    ## service-flow-manager.h (module 'wimax'): uint32_t ns3::ServiceFlowManager::GetNrServiceFlows() const [member function]
    cls.add_method('GetNrServiceFlows', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## service-flow-manager.h (module 'wimax'): ns3::ServiceFlow * ns3::ServiceFlowManager::GetServiceFlow(uint32_t sfid) const [member function]
    cls.add_method('GetServiceFlow', 
                   'ns3::ServiceFlow *', 
                   [param('uint32_t', 'sfid')], 
                   is_const=True)
    ## service-flow-manager.h (module 'wimax'): ns3::ServiceFlow * ns3::ServiceFlowManager::GetServiceFlow(ns3::Cid cid) const [member function]
    cls.add_method('GetServiceFlow', 
                   'ns3::ServiceFlow *', 
                   [param('ns3::Cid', 'cid')], 
                   is_const=True)
    ## service-flow-manager.h (module 'wimax'): std::vector<ns3::ServiceFlow *, std::allocator<ns3::ServiceFlow *>> ns3::ServiceFlowManager::GetServiceFlows(ns3::ServiceFlow::SchedulingType schedulingType) const [member function]
    cls.add_method('GetServiceFlows', 
                   'std::vector< ns3::ServiceFlow * >', 
                   [param('ns3::ServiceFlow::SchedulingType', 'schedulingType')], 
                   is_const=True)
    ## service-flow-manager.h (module 'wimax'): static ns3::TypeId ns3::ServiceFlowManager::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    return

def register_Ns3SfVectorTlvValue_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::SfVectorTlvValue::SfVectorTlvValue(ns3::SfVectorTlvValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::SfVectorTlvValue const &', 'arg0')])
    ## wimax-tlv.h (module 'wimax'): ns3::SfVectorTlvValue::SfVectorTlvValue() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::SfVectorTlvValue * ns3::SfVectorTlvValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::SfVectorTlvValue *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::SfVectorTlvValue::Deserialize(ns3::Buffer::Iterator start, uint64_t valueLength) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start'), param('uint64_t', 'valueLength')], 
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

def register_Ns3SimpleRefCount__Ns3NixVector_Ns3Empty_Ns3DefaultDeleter__lt__ns3NixVector__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector>>::SimpleRefCount(ns3::SimpleRefCount<ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::NixVector, ns3::empty, ns3::DefaultDeleter< ns3::NixVector > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3OutputStreamWrapper_Ns3Empty_Ns3DefaultDeleter__lt__ns3OutputStreamWrapper__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::OutputStreamWrapper, ns3::empty, ns3::DefaultDeleter<ns3::OutputStreamWrapper>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::OutputStreamWrapper, ns3::empty, ns3::DefaultDeleter<ns3::OutputStreamWrapper>>::SimpleRefCount(ns3::SimpleRefCount<ns3::OutputStreamWrapper, ns3::empty, ns3::DefaultDeleter<ns3::OutputStreamWrapper>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::OutputStreamWrapper, ns3::empty, ns3::DefaultDeleter< ns3::OutputStreamWrapper > > const &', 'o')])
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

def register_Ns3SsServiceFlowManager_methods(root_module, cls):
    ## ss-service-flow-manager.h (module 'wimax'): ns3::SsServiceFlowManager::SsServiceFlowManager(ns3::SsServiceFlowManager const & arg0) [constructor]
    cls.add_constructor([param('ns3::SsServiceFlowManager const &', 'arg0')])
    ## ss-service-flow-manager.h (module 'wimax'): ns3::SsServiceFlowManager::SsServiceFlowManager(ns3::Ptr<ns3::SubscriberStationNetDevice> device) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::SubscriberStationNetDevice >', 'device')])
    ## ss-service-flow-manager.h (module 'wimax'): void ns3::SsServiceFlowManager::AddServiceFlow(ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('AddServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow *', 'serviceFlow')])
    ## ss-service-flow-manager.h (module 'wimax'): void ns3::SsServiceFlowManager::AddServiceFlow(ns3::ServiceFlow serviceFlow) [member function]
    cls.add_method('AddServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow', 'serviceFlow')])
    ## ss-service-flow-manager.h (module 'wimax'): ns3::Ptr<ns3::Packet> ns3::SsServiceFlowManager::CreateDsaAck() [member function]
    cls.add_method('CreateDsaAck', 
                   'ns3::Ptr< ns3::Packet >', 
                   [])
    ## ss-service-flow-manager.h (module 'wimax'): ns3::DsaReq ns3::SsServiceFlowManager::CreateDsaReq(ns3::ServiceFlow const * serviceFlow) [member function]
    cls.add_method('CreateDsaReq', 
                   'ns3::DsaReq', 
                   [param('ns3::ServiceFlow const *', 'serviceFlow')])
    ## ss-service-flow-manager.h (module 'wimax'): void ns3::SsServiceFlowManager::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## ss-service-flow-manager.h (module 'wimax'): ns3::EventId ns3::SsServiceFlowManager::GetDsaAckTimeoutEvent() const [member function]
    cls.add_method('GetDsaAckTimeoutEvent', 
                   'ns3::EventId', 
                   [], 
                   is_const=True)
    ## ss-service-flow-manager.h (module 'wimax'): ns3::EventId ns3::SsServiceFlowManager::GetDsaRspTimeoutEvent() const [member function]
    cls.add_method('GetDsaRspTimeoutEvent', 
                   'ns3::EventId', 
                   [], 
                   is_const=True)
    ## ss-service-flow-manager.h (module 'wimax'): uint8_t ns3::SsServiceFlowManager::GetMaxDsaReqRetries() const [member function]
    cls.add_method('GetMaxDsaReqRetries', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ss-service-flow-manager.h (module 'wimax'): static ns3::TypeId ns3::SsServiceFlowManager::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## ss-service-flow-manager.h (module 'wimax'): void ns3::SsServiceFlowManager::InitiateServiceFlows() [member function]
    cls.add_method('InitiateServiceFlows', 
                   'void', 
                   [])
    ## ss-service-flow-manager.h (module 'wimax'): void ns3::SsServiceFlowManager::ProcessDsaRsp(ns3::DsaRsp const & dsaRsp) [member function]
    cls.add_method('ProcessDsaRsp', 
                   'void', 
                   [param('ns3::DsaRsp const &', 'dsaRsp')])
    ## ss-service-flow-manager.h (module 'wimax'): void ns3::SsServiceFlowManager::ScheduleDsaReq(ns3::ServiceFlow const * serviceFlow) [member function]
    cls.add_method('ScheduleDsaReq', 
                   'void', 
                   [param('ns3::ServiceFlow const *', 'serviceFlow')])
    ## ss-service-flow-manager.h (module 'wimax'): void ns3::SsServiceFlowManager::SetMaxDsaReqRetries(uint8_t maxDsaReqRetries) [member function]
    cls.add_method('SetMaxDsaReqRetries', 
                   'void', 
                   [param('uint8_t', 'maxDsaReqRetries')])
    return

def register_Ns3ThreeLogDistancePropagationLossModel_methods(root_module, cls):
    ## propagation-loss-model.h (module 'propagation'): static ns3::TypeId ns3::ThreeLogDistancePropagationLossModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## propagation-loss-model.h (module 'propagation'): ns3::ThreeLogDistancePropagationLossModel::ThreeLogDistancePropagationLossModel() [constructor]
    cls.add_constructor([])
    ## propagation-loss-model.h (module 'propagation'): double ns3::ThreeLogDistancePropagationLossModel::DoCalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('DoCalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::ThreeLogDistancePropagationLossModel::DoAssignStreams(int64_t stream) [member function]
    cls.add_method('DoAssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3Tlv_methods(root_module, cls):
    ## wimax-tlv.h (module 'wimax'): ns3::Tlv::Tlv(uint8_t type, uint64_t length, ns3::TlvValue const & value) [constructor]
    cls.add_constructor([param('uint8_t', 'type'), param('uint64_t', 'length'), param('ns3::TlvValue const &', 'value')])
    ## wimax-tlv.h (module 'wimax'): ns3::Tlv::Tlv() [constructor]
    cls.add_constructor([])
    ## wimax-tlv.h (module 'wimax'): ns3::Tlv::Tlv(ns3::Tlv const & tlv) [constructor]
    cls.add_constructor([param('ns3::Tlv const &', 'tlv')])
    ## wimax-tlv.h (module 'wimax'): ns3::Tlv * ns3::Tlv::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Tlv *', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): ns3::TlvValue * ns3::Tlv::CopyValue() const [member function]
    cls.add_method('CopyValue', 
                   'ns3::TlvValue *', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::Tlv::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): ns3::TypeId ns3::Tlv::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint64_t ns3::Tlv::GetLength() const [member function]
    cls.add_method('GetLength', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): uint32_t ns3::Tlv::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): uint8_t ns3::Tlv::GetSizeOfLen() const [member function]
    cls.add_method('GetSizeOfLen', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): uint8_t ns3::Tlv::GetType() const [member function]
    cls.add_method('GetType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-tlv.h (module 'wimax'): static ns3::TypeId ns3::Tlv::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-tlv.h (module 'wimax'): ns3::TlvValue * ns3::Tlv::PeekValue() [member function]
    cls.add_method('PeekValue', 
                   'ns3::TlvValue *', 
                   [])
    ## wimax-tlv.h (module 'wimax'): void ns3::Tlv::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## wimax-tlv.h (module 'wimax'): void ns3::Tlv::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
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

def register_Ns3TwoRayGroundPropagationLossModel_methods(root_module, cls):
    ## propagation-loss-model.h (module 'propagation'): static ns3::TypeId ns3::TwoRayGroundPropagationLossModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## propagation-loss-model.h (module 'propagation'): ns3::TwoRayGroundPropagationLossModel::TwoRayGroundPropagationLossModel() [constructor]
    cls.add_constructor([])
    ## propagation-loss-model.h (module 'propagation'): void ns3::TwoRayGroundPropagationLossModel::SetFrequency(double frequency) [member function]
    cls.add_method('SetFrequency', 
                   'void', 
                   [param('double', 'frequency')])
    ## propagation-loss-model.h (module 'propagation'): void ns3::TwoRayGroundPropagationLossModel::SetSystemLoss(double systemLoss) [member function]
    cls.add_method('SetSystemLoss', 
                   'void', 
                   [param('double', 'systemLoss')])
    ## propagation-loss-model.h (module 'propagation'): void ns3::TwoRayGroundPropagationLossModel::SetMinDistance(double minDistance) [member function]
    cls.add_method('SetMinDistance', 
                   'void', 
                   [param('double', 'minDistance')])
    ## propagation-loss-model.h (module 'propagation'): double ns3::TwoRayGroundPropagationLossModel::GetMinDistance() const [member function]
    cls.add_method('GetMinDistance', 
                   'double', 
                   [], 
                   is_const=True)
    ## propagation-loss-model.h (module 'propagation'): double ns3::TwoRayGroundPropagationLossModel::GetFrequency() const [member function]
    cls.add_method('GetFrequency', 
                   'double', 
                   [], 
                   is_const=True)
    ## propagation-loss-model.h (module 'propagation'): double ns3::TwoRayGroundPropagationLossModel::GetSystemLoss() const [member function]
    cls.add_method('GetSystemLoss', 
                   'double', 
                   [], 
                   is_const=True)
    ## propagation-loss-model.h (module 'propagation'): void ns3::TwoRayGroundPropagationLossModel::SetHeightAboveZ(double heightAboveZ) [member function]
    cls.add_method('SetHeightAboveZ', 
                   'void', 
                   [param('double', 'heightAboveZ')])
    ## propagation-loss-model.h (module 'propagation'): double ns3::TwoRayGroundPropagationLossModel::DoCalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('DoCalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::TwoRayGroundPropagationLossModel::DoAssignStreams(int64_t stream) [member function]
    cls.add_method('DoAssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3Ucd_methods(root_module, cls):
    ## ul-mac-messages.h (module 'wimax'): ns3::Ucd::Ucd(ns3::Ucd const & arg0) [constructor]
    cls.add_constructor([param('ns3::Ucd const &', 'arg0')])
    ## ul-mac-messages.h (module 'wimax'): ns3::Ucd::Ucd() [constructor]
    cls.add_constructor([])
    ## ul-mac-messages.h (module 'wimax'): void ns3::Ucd::AddUlBurstProfile(ns3::OfdmUlBurstProfile ulBurstProfile) [member function]
    cls.add_method('AddUlBurstProfile', 
                   'void', 
                   [param('ns3::OfdmUlBurstProfile', 'ulBurstProfile')])
    ## ul-mac-messages.h (module 'wimax'): uint32_t ns3::Ucd::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## ul-mac-messages.h (module 'wimax'): ns3::OfdmUcdChannelEncodings ns3::Ucd::GetChannelEncodings() const [member function]
    cls.add_method('GetChannelEncodings', 
                   'ns3::OfdmUcdChannelEncodings', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::Ucd::GetConfigurationChangeCount() const [member function]
    cls.add_method('GetConfigurationChangeCount', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): ns3::TypeId ns3::Ucd::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ul-mac-messages.h (module 'wimax'): std::string ns3::Ucd::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::Ucd::GetNrUlBurstProfiles() const [member function]
    cls.add_method('GetNrUlBurstProfiles', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::Ucd::GetRangingBackoffEnd() const [member function]
    cls.add_method('GetRangingBackoffEnd', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::Ucd::GetRangingBackoffStart() const [member function]
    cls.add_method('GetRangingBackoffStart', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::Ucd::GetRequestBackoffEnd() const [member function]
    cls.add_method('GetRequestBackoffEnd', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::Ucd::GetRequestBackoffStart() const [member function]
    cls.add_method('GetRequestBackoffStart', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint32_t ns3::Ucd::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ul-mac-messages.h (module 'wimax'): static ns3::TypeId ns3::Ucd::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## ul-mac-messages.h (module 'wimax'): std::vector<ns3::OfdmUlBurstProfile, std::allocator<ns3::OfdmUlBurstProfile>> ns3::Ucd::GetUlBurstProfiles() const [member function]
    cls.add_method('GetUlBurstProfiles', 
                   'std::vector< ns3::OfdmUlBurstProfile >', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): void ns3::Ucd::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ul-mac-messages.h (module 'wimax'): void ns3::Ucd::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## ul-mac-messages.h (module 'wimax'): void ns3::Ucd::SetChannelEncodings(ns3::OfdmUcdChannelEncodings channelEncodings) [member function]
    cls.add_method('SetChannelEncodings', 
                   'void', 
                   [param('ns3::OfdmUcdChannelEncodings', 'channelEncodings')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::Ucd::SetConfigurationChangeCount(uint8_t ucdCount) [member function]
    cls.add_method('SetConfigurationChangeCount', 
                   'void', 
                   [param('uint8_t', 'ucdCount')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::Ucd::SetNrUlBurstProfiles(uint8_t nrUlBurstProfiles) [member function]
    cls.add_method('SetNrUlBurstProfiles', 
                   'void', 
                   [param('uint8_t', 'nrUlBurstProfiles')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::Ucd::SetRangingBackoffEnd(uint8_t rangingBackoffEnd) [member function]
    cls.add_method('SetRangingBackoffEnd', 
                   'void', 
                   [param('uint8_t', 'rangingBackoffEnd')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::Ucd::SetRangingBackoffStart(uint8_t rangingBackoffStart) [member function]
    cls.add_method('SetRangingBackoffStart', 
                   'void', 
                   [param('uint8_t', 'rangingBackoffStart')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::Ucd::SetRequestBackoffEnd(uint8_t requestBackoffEnd) [member function]
    cls.add_method('SetRequestBackoffEnd', 
                   'void', 
                   [param('uint8_t', 'requestBackoffEnd')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::Ucd::SetRequestBackoffStart(uint8_t requestBackoffStart) [member function]
    cls.add_method('SetRequestBackoffStart', 
                   'void', 
                   [param('uint8_t', 'requestBackoffStart')])
    return

def register_Ns3UlJob_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    ## ul-job.h (module 'wimax'): ns3::UlJob::UlJob(ns3::UlJob const & arg0) [constructor]
    cls.add_constructor([param('ns3::UlJob const &', 'arg0')])
    ## ul-job.h (module 'wimax'): ns3::UlJob::UlJob() [constructor]
    cls.add_constructor([])
    ## ul-job.h (module 'wimax'): ns3::Time ns3::UlJob::GetDeadline() [member function]
    cls.add_method('GetDeadline', 
                   'ns3::Time', 
                   [])
    ## ul-job.h (module 'wimax'): ns3::Time ns3::UlJob::GetPeriod() [member function]
    cls.add_method('GetPeriod', 
                   'ns3::Time', 
                   [])
    ## ul-job.h (module 'wimax'): ns3::Time ns3::UlJob::GetReleaseTime() [member function]
    cls.add_method('GetReleaseTime', 
                   'ns3::Time', 
                   [])
    ## ul-job.h (module 'wimax'): ns3::ServiceFlow::SchedulingType ns3::UlJob::GetSchedulingType() [member function]
    cls.add_method('GetSchedulingType', 
                   'ns3::ServiceFlow::SchedulingType', 
                   [])
    ## ul-job.h (module 'wimax'): ns3::ServiceFlow * ns3::UlJob::GetServiceFlow() [member function]
    cls.add_method('GetServiceFlow', 
                   'ns3::ServiceFlow *', 
                   [])
    ## ul-job.h (module 'wimax'): uint32_t ns3::UlJob::GetSize() [member function]
    cls.add_method('GetSize', 
                   'uint32_t', 
                   [])
    ## ul-job.h (module 'wimax'): ns3::SSRecord * ns3::UlJob::GetSsRecord() [member function]
    cls.add_method('GetSsRecord', 
                   'ns3::SSRecord *', 
                   [])
    ## ul-job.h (module 'wimax'): ns3::ReqType ns3::UlJob::GetType() [member function]
    cls.add_method('GetType', 
                   'ns3::ReqType', 
                   [])
    ## ul-job.h (module 'wimax'): void ns3::UlJob::SetDeadline(ns3::Time deadline) [member function]
    cls.add_method('SetDeadline', 
                   'void', 
                   [param('ns3::Time', 'deadline')])
    ## ul-job.h (module 'wimax'): void ns3::UlJob::SetPeriod(ns3::Time period) [member function]
    cls.add_method('SetPeriod', 
                   'void', 
                   [param('ns3::Time', 'period')])
    ## ul-job.h (module 'wimax'): void ns3::UlJob::SetReleaseTime(ns3::Time releaseTime) [member function]
    cls.add_method('SetReleaseTime', 
                   'void', 
                   [param('ns3::Time', 'releaseTime')])
    ## ul-job.h (module 'wimax'): void ns3::UlJob::SetSchedulingType(ns3::ServiceFlow::SchedulingType schedulingType) [member function]
    cls.add_method('SetSchedulingType', 
                   'void', 
                   [param('ns3::ServiceFlow::SchedulingType', 'schedulingType')])
    ## ul-job.h (module 'wimax'): void ns3::UlJob::SetServiceFlow(ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('SetServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow *', 'serviceFlow')])
    ## ul-job.h (module 'wimax'): void ns3::UlJob::SetSize(uint32_t size) [member function]
    cls.add_method('SetSize', 
                   'void', 
                   [param('uint32_t', 'size')])
    ## ul-job.h (module 'wimax'): void ns3::UlJob::SetSsRecord(ns3::SSRecord * ssRecord) [member function]
    cls.add_method('SetSsRecord', 
                   'void', 
                   [param('ns3::SSRecord *', 'ssRecord')])
    ## ul-job.h (module 'wimax'): void ns3::UlJob::SetType(ns3::ReqType type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('ns3::ReqType', 'type')])
    return

def register_Ns3UlMap_methods(root_module, cls):
    ## ul-mac-messages.h (module 'wimax'): ns3::UlMap::UlMap(ns3::UlMap const & arg0) [constructor]
    cls.add_constructor([param('ns3::UlMap const &', 'arg0')])
    ## ul-mac-messages.h (module 'wimax'): ns3::UlMap::UlMap() [constructor]
    cls.add_constructor([])
    ## ul-mac-messages.h (module 'wimax'): void ns3::UlMap::AddUlMapElement(ns3::OfdmUlMapIe ulMapElement) [member function]
    cls.add_method('AddUlMapElement', 
                   'void', 
                   [param('ns3::OfdmUlMapIe', 'ulMapElement')])
    ## ul-mac-messages.h (module 'wimax'): uint32_t ns3::UlMap::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## ul-mac-messages.h (module 'wimax'): uint32_t ns3::UlMap::GetAllocationStartTime() const [member function]
    cls.add_method('GetAllocationStartTime', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): ns3::TypeId ns3::UlMap::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ul-mac-messages.h (module 'wimax'): std::string ns3::UlMap::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): uint32_t ns3::UlMap::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ul-mac-messages.h (module 'wimax'): static ns3::TypeId ns3::UlMap::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## ul-mac-messages.h (module 'wimax'): uint8_t ns3::UlMap::GetUcdCount() const [member function]
    cls.add_method('GetUcdCount', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): std::list<ns3::OfdmUlMapIe, std::allocator<ns3::OfdmUlMapIe>> ns3::UlMap::GetUlMapElements() const [member function]
    cls.add_method('GetUlMapElements', 
                   'std::list< ns3::OfdmUlMapIe >', 
                   [], 
                   is_const=True)
    ## ul-mac-messages.h (module 'wimax'): void ns3::UlMap::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ul-mac-messages.h (module 'wimax'): void ns3::UlMap::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## ul-mac-messages.h (module 'wimax'): void ns3::UlMap::SetAllocationStartTime(uint32_t allocationStartTime) [member function]
    cls.add_method('SetAllocationStartTime', 
                   'void', 
                   [param('uint32_t', 'allocationStartTime')])
    ## ul-mac-messages.h (module 'wimax'): void ns3::UlMap::SetUcdCount(uint8_t ucdCount) [member function]
    cls.add_method('SetUcdCount', 
                   'void', 
                   [param('uint8_t', 'ucdCount')])
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

def register_Ns3UplinkScheduler_methods(root_module, cls):
    ## bs-uplink-scheduler.h (module 'wimax'): ns3::UplinkScheduler::UplinkScheduler(ns3::UplinkScheduler const & arg0) [constructor]
    cls.add_constructor([param('ns3::UplinkScheduler const &', 'arg0')])
    ## bs-uplink-scheduler.h (module 'wimax'): ns3::UplinkScheduler::UplinkScheduler() [constructor]
    cls.add_constructor([])
    ## bs-uplink-scheduler.h (module 'wimax'): ns3::UplinkScheduler::UplinkScheduler(ns3::Ptr<ns3::BaseStationNetDevice> bs) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::BaseStationNetDevice >', 'bs')])
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::AddUplinkAllocation(ns3::OfdmUlMapIe & ulMapIe, uint32_t const & allocationSize, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('AddUplinkAllocation', 
                   'void', 
                   [param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('uint32_t const &', 'allocationSize'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::AllocateInitialRangingInterval(uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('AllocateInitialRangingInterval', 
                   'void', 
                   [param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): uint32_t ns3::UplinkScheduler::CalculateAllocationStartTime() [member function]
    cls.add_method('CalculateAllocationStartTime', 
                   'uint32_t', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): ns3::Ptr<ns3::BaseStationNetDevice> ns3::UplinkScheduler::GetBs() [member function]
    cls.add_method('GetBs', 
                   'ns3::Ptr< ns3::BaseStationNetDevice >', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::GetChannelDescriptorsToUpdate(bool & arg0, bool & arg1, bool & arg2, bool & arg3) [member function]
    cls.add_method('GetChannelDescriptorsToUpdate', 
                   'void', 
                   [param('bool &', 'arg0'), param('bool &', 'arg1'), param('bool &', 'arg2'), param('bool &', 'arg3')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): ns3::Time ns3::UplinkScheduler::GetDcdTimeStamp() const [member function]
    cls.add_method('GetDcdTimeStamp', 
                   'ns3::Time', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): bool ns3::UplinkScheduler::GetIsInvIrIntrvlAllocated() const [member function]
    cls.add_method('GetIsInvIrIntrvlAllocated', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): bool ns3::UplinkScheduler::GetIsIrIntrvlAllocated() const [member function]
    cls.add_method('GetIsIrIntrvlAllocated', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): uint8_t ns3::UplinkScheduler::GetNrIrOppsAllocated() const [member function]
    cls.add_method('GetNrIrOppsAllocated', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): ns3::Time ns3::UplinkScheduler::GetTimeStampIrInterval() [member function]
    cls.add_method('GetTimeStampIrInterval', 
                   'ns3::Time', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): static ns3::TypeId ns3::UplinkScheduler::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## bs-uplink-scheduler.h (module 'wimax'): ns3::Time ns3::UplinkScheduler::GetUcdTimeStamp() const [member function]
    cls.add_method('GetUcdTimeStamp', 
                   'ns3::Time', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): std::list<ns3::OfdmUlMapIe, std::allocator<ns3::OfdmUlMapIe>> ns3::UplinkScheduler::GetUplinkAllocations() const [member function]
    cls.add_method('GetUplinkAllocations', 
                   'std::list< ns3::OfdmUlMapIe >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::InitOnce() [member function]
    cls.add_method('InitOnce', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::OnSetRequestedBandwidth(ns3::ServiceFlowRecord * sfr) [member function]
    cls.add_method('OnSetRequestedBandwidth', 
                   'void', 
                   [param('ns3::ServiceFlowRecord *', 'sfr')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::ProcessBandwidthRequest(ns3::BandwidthRequestHeader const & bwRequestHdr) [member function]
    cls.add_method('ProcessBandwidthRequest', 
                   'void', 
                   [param('ns3::BandwidthRequestHeader const &', 'bwRequestHdr')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::Schedule() [member function]
    cls.add_method('Schedule', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::ServiceBandwidthRequests(ns3::SSRecord const * ssRecord, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceBandwidthRequests', 
                   'void', 
                   [param('ns3::SSRecord const *', 'ssRecord'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): bool ns3::UplinkScheduler::ServiceBandwidthRequests(ns3::ServiceFlow * serviceFlow, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceBandwidthRequests', 
                   'bool', 
                   [param('ns3::ServiceFlow *', 'serviceFlow'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::ServiceUnsolicitedGrants(ns3::SSRecord const * ssRecord, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceUnsolicitedGrants', 
                   'void', 
                   [param('ns3::SSRecord const *', 'ssRecord'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::SetBs(ns3::Ptr<ns3::BaseStationNetDevice> bs) [member function]
    cls.add_method('SetBs', 
                   'void', 
                   [param('ns3::Ptr< ns3::BaseStationNetDevice >', 'bs')], 
                   is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::SetDcdTimeStamp(ns3::Time dcdTimeStamp) [member function]
    cls.add_method('SetDcdTimeStamp', 
                   'void', 
                   [param('ns3::Time', 'dcdTimeStamp')], 
                   is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::SetIsInvIrIntrvlAllocated(bool isInvIrIntrvlAllocated) [member function]
    cls.add_method('SetIsInvIrIntrvlAllocated', 
                   'void', 
                   [param('bool', 'isInvIrIntrvlAllocated')], 
                   is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::SetIsIrIntrvlAllocated(bool isIrIntrvlAllocated) [member function]
    cls.add_method('SetIsIrIntrvlAllocated', 
                   'void', 
                   [param('bool', 'isIrIntrvlAllocated')], 
                   is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::SetNrIrOppsAllocated(uint8_t nrIrOppsAllocated) [member function]
    cls.add_method('SetNrIrOppsAllocated', 
                   'void', 
                   [param('uint8_t', 'nrIrOppsAllocated')], 
                   is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::SetTimeStampIrInterval(ns3::Time timeStampIrInterval) [member function]
    cls.add_method('SetTimeStampIrInterval', 
                   'void', 
                   [param('ns3::Time', 'timeStampIrInterval')], 
                   is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::SetUcdTimeStamp(ns3::Time ucdTimeStamp) [member function]
    cls.add_method('SetUcdTimeStamp', 
                   'void', 
                   [param('ns3::Time', 'ucdTimeStamp')], 
                   is_virtual=True)
    ## bs-uplink-scheduler.h (module 'wimax'): void ns3::UplinkScheduler::SetupServiceFlow(ns3::SSRecord * ssRecord, ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('SetupServiceFlow', 
                   'void', 
                   [param('ns3::SSRecord *', 'ssRecord'), param('ns3::ServiceFlow *', 'serviceFlow')], 
                   is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3UplinkSchedulerMBQoS_methods(root_module, cls):
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): ns3::UplinkSchedulerMBQoS::UplinkSchedulerMBQoS(ns3::UplinkSchedulerMBQoS const & arg0) [constructor]
    cls.add_constructor([param('ns3::UplinkSchedulerMBQoS const &', 'arg0')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): ns3::UplinkSchedulerMBQoS::UplinkSchedulerMBQoS() [constructor]
    cls.add_constructor([])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): ns3::UplinkSchedulerMBQoS::UplinkSchedulerMBQoS(ns3::Time time) [constructor]
    cls.add_constructor([param('ns3::Time', 'time')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::AddUplinkAllocation(ns3::OfdmUlMapIe & ulMapIe, uint32_t const & allocationSize, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('AddUplinkAllocation', 
                   'void', 
                   [param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('uint32_t const &', 'allocationSize'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::AllocateInitialRangingInterval(uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('AllocateInitialRangingInterval', 
                   'void', 
                   [param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): uint32_t ns3::UplinkSchedulerMBQoS::CalculateAllocationStartTime() [member function]
    cls.add_method('CalculateAllocationStartTime', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::CheckDeadline(uint32_t & availableSymbols) [member function]
    cls.add_method('CheckDeadline', 
                   'void', 
                   [param('uint32_t &', 'availableSymbols')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::CheckMinimumBandwidth(uint32_t & availableSymbols) [member function]
    cls.add_method('CheckMinimumBandwidth', 
                   'void', 
                   [param('uint32_t &', 'availableSymbols')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): uint32_t ns3::UplinkSchedulerMBQoS::CountSymbolsJobs(ns3::Ptr<ns3::UlJob> job) [member function]
    cls.add_method('CountSymbolsJobs', 
                   'uint32_t', 
                   [param('ns3::Ptr< ns3::UlJob >', 'job')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): uint32_t ns3::UplinkSchedulerMBQoS::CountSymbolsQueue(std::list<ns3::Ptr<ns3::UlJob>, std::allocator<ns3::Ptr<ns3::UlJob>>> jobs) [member function]
    cls.add_method('CountSymbolsQueue', 
                   'uint32_t', 
                   [param('std::list< ns3::Ptr< ns3::UlJob > >', 'jobs')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): ns3::Ptr<ns3::UlJob> ns3::UplinkSchedulerMBQoS::CreateUlJob(ns3::SSRecord * ssRecord, ns3::ServiceFlow::SchedulingType schedType, ns3::ReqType reqType) [member function]
    cls.add_method('CreateUlJob', 
                   'ns3::Ptr< ns3::UlJob >', 
                   [param('ns3::SSRecord *', 'ssRecord'), param('ns3::ServiceFlow::SchedulingType', 'schedType'), param('ns3::ReqType', 'reqType')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): ns3::Ptr<ns3::UlJob> ns3::UplinkSchedulerMBQoS::DequeueJob(ns3::UlJob::JobPriority priority) [member function]
    cls.add_method('DequeueJob', 
                   'ns3::Ptr< ns3::UlJob >', 
                   [param('ns3::UlJob::JobPriority', 'priority')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): ns3::Time ns3::UplinkSchedulerMBQoS::DetermineDeadline(ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('DetermineDeadline', 
                   'ns3::Time', 
                   [param('ns3::ServiceFlow *', 'serviceFlow')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::EnqueueJob(ns3::UlJob::JobPriority priority, ns3::Ptr<ns3::UlJob> job) [member function]
    cls.add_method('EnqueueJob', 
                   'void', 
                   [param('ns3::UlJob::JobPriority', 'priority'), param('ns3::Ptr< ns3::UlJob >', 'job')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::GetChannelDescriptorsToUpdate(bool & updateDcd, bool & updateUcd, bool & sendDcd, bool & sendUcd) [member function]
    cls.add_method('GetChannelDescriptorsToUpdate', 
                   'void', 
                   [param('bool &', 'updateDcd'), param('bool &', 'updateUcd'), param('bool &', 'sendDcd'), param('bool &', 'sendUcd')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): uint32_t ns3::UplinkSchedulerMBQoS::GetPendingSize(ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('GetPendingSize', 
                   'uint32_t', 
                   [param('ns3::ServiceFlow *', 'serviceFlow')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): static ns3::TypeId ns3::UplinkSchedulerMBQoS::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): std::list<ns3::OfdmUlMapIe, std::allocator<ns3::OfdmUlMapIe>> ns3::UplinkSchedulerMBQoS::GetUplinkAllocations() const [member function]
    cls.add_method('GetUplinkAllocations', 
                   'std::list< ns3::OfdmUlMapIe >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::InitOnce() [member function]
    cls.add_method('InitOnce', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::OnSetRequestedBandwidth(ns3::ServiceFlowRecord * sfr) [member function]
    cls.add_method('OnSetRequestedBandwidth', 
                   'void', 
                   [param('ns3::ServiceFlowRecord *', 'sfr')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::ProcessBandwidthRequest(ns3::BandwidthRequestHeader const & bwRequestHdr) [member function]
    cls.add_method('ProcessBandwidthRequest', 
                   'void', 
                   [param('ns3::BandwidthRequestHeader const &', 'bwRequestHdr')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::Schedule() [member function]
    cls.add_method('Schedule', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::ServiceBandwidthRequests(ns3::SSRecord const * ssRecord, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceBandwidthRequests', 
                   'void', 
                   [param('ns3::SSRecord const *', 'ssRecord'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): bool ns3::UplinkSchedulerMBQoS::ServiceBandwidthRequests(ns3::ServiceFlow * serviceFlow, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceBandwidthRequests', 
                   'bool', 
                   [param('ns3::ServiceFlow *', 'serviceFlow'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): bool ns3::UplinkSchedulerMBQoS::ServiceBandwidthRequestsBytes(ns3::ServiceFlow * serviceFlow, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols, uint32_t allocationSizeBytes) [member function]
    cls.add_method('ServiceBandwidthRequestsBytes', 
                   'bool', 
                   [param('ns3::ServiceFlow *', 'serviceFlow'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols'), param('uint32_t', 'allocationSizeBytes')])
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::ServiceUnsolicitedGrants(ns3::SSRecord const * ssRecord, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceUnsolicitedGrants', 
                   'void', 
                   [param('ns3::SSRecord const *', 'ssRecord'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::SetupServiceFlow(ns3::SSRecord * ssRecord, ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('SetupServiceFlow', 
                   'void', 
                   [param('ns3::SSRecord *', 'ssRecord'), param('ns3::ServiceFlow *', 'serviceFlow')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-mbqos.h (module 'wimax'): void ns3::UplinkSchedulerMBQoS::UplinkSchedWindowTimer() [member function]
    cls.add_method('UplinkSchedWindowTimer', 
                   'void', 
                   [])
    return

def register_Ns3UplinkSchedulerRtps_methods(root_module, cls):
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): ns3::UplinkSchedulerRtps::UplinkSchedulerRtps(ns3::UplinkSchedulerRtps const & arg0) [constructor]
    cls.add_constructor([param('ns3::UplinkSchedulerRtps const &', 'arg0')])
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): ns3::UplinkSchedulerRtps::UplinkSchedulerRtps() [constructor]
    cls.add_constructor([])
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): ns3::UplinkSchedulerRtps::UplinkSchedulerRtps(ns3::Ptr<ns3::BaseStationNetDevice> bs) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::BaseStationNetDevice >', 'bs')])
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::AddUplinkAllocation(ns3::OfdmUlMapIe & ulMapIe, uint32_t const & allocationSize, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('AddUplinkAllocation', 
                   'void', 
                   [param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('uint32_t const &', 'allocationSize'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::AllocateInitialRangingInterval(uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('AllocateInitialRangingInterval', 
                   'void', 
                   [param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): uint32_t ns3::UplinkSchedulerRtps::CalculateAllocationStartTime() [member function]
    cls.add_method('CalculateAllocationStartTime', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::GetChannelDescriptorsToUpdate(bool & updateDcd, bool & updateUcd, bool & sendDcd, bool & sendUcd) [member function]
    cls.add_method('GetChannelDescriptorsToUpdate', 
                   'void', 
                   [param('bool &', 'updateDcd'), param('bool &', 'updateUcd'), param('bool &', 'sendDcd'), param('bool &', 'sendUcd')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): static ns3::TypeId ns3::UplinkSchedulerRtps::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): std::list<ns3::OfdmUlMapIe, std::allocator<ns3::OfdmUlMapIe>> ns3::UplinkSchedulerRtps::GetUplinkAllocations() const [member function]
    cls.add_method('GetUplinkAllocations', 
                   'std::list< ns3::OfdmUlMapIe >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::InitOnce() [member function]
    cls.add_method('InitOnce', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::OnSetRequestedBandwidth(ns3::ServiceFlowRecord * sfr) [member function]
    cls.add_method('OnSetRequestedBandwidth', 
                   'void', 
                   [param('ns3::ServiceFlowRecord *', 'sfr')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::ProcessBandwidthRequest(ns3::BandwidthRequestHeader const & bwRequestHdr) [member function]
    cls.add_method('ProcessBandwidthRequest', 
                   'void', 
                   [param('ns3::BandwidthRequestHeader const &', 'bwRequestHdr')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::Schedule() [member function]
    cls.add_method('Schedule', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::ServiceBandwidthRequests(ns3::SSRecord const * ssRecord, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceBandwidthRequests', 
                   'void', 
                   [param('ns3::SSRecord const *', 'ssRecord'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): bool ns3::UplinkSchedulerRtps::ServiceBandwidthRequests(ns3::ServiceFlow * serviceFlow, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceBandwidthRequests', 
                   'bool', 
                   [param('ns3::ServiceFlow *', 'serviceFlow'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::ServiceUnsolicitedGrants(ns3::SSRecord const * ssRecord, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceUnsolicitedGrants', 
                   'void', 
                   [param('ns3::SSRecord const *', 'ssRecord'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::SetupServiceFlow(ns3::SSRecord * ssRecord, ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('SetupServiceFlow', 
                   'void', 
                   [param('ns3::SSRecord *', 'ssRecord'), param('ns3::ServiceFlow *', 'serviceFlow')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-rtps.h (module 'wimax'): void ns3::UplinkSchedulerRtps::ULSchedulerRTPSConnection(uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ULSchedulerRTPSConnection', 
                   'void', 
                   [param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')])
    return

def register_Ns3UplinkSchedulerSimple_methods(root_module, cls):
    ## bs-uplink-scheduler-simple.h (module 'wimax'): ns3::UplinkSchedulerSimple::UplinkSchedulerSimple(ns3::UplinkSchedulerSimple const & arg0) [constructor]
    cls.add_constructor([param('ns3::UplinkSchedulerSimple const &', 'arg0')])
    ## bs-uplink-scheduler-simple.h (module 'wimax'): ns3::UplinkSchedulerSimple::UplinkSchedulerSimple() [constructor]
    cls.add_constructor([])
    ## bs-uplink-scheduler-simple.h (module 'wimax'): ns3::UplinkSchedulerSimple::UplinkSchedulerSimple(ns3::Ptr<ns3::BaseStationNetDevice> bs) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::BaseStationNetDevice >', 'bs')])
    ## bs-uplink-scheduler-simple.h (module 'wimax'): void ns3::UplinkSchedulerSimple::AddUplinkAllocation(ns3::OfdmUlMapIe & ulMapIe, uint32_t const & allocationSize, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('AddUplinkAllocation', 
                   'void', 
                   [param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('uint32_t const &', 'allocationSize'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): void ns3::UplinkSchedulerSimple::AllocateInitialRangingInterval(uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('AllocateInitialRangingInterval', 
                   'void', 
                   [param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): uint32_t ns3::UplinkSchedulerSimple::CalculateAllocationStartTime() [member function]
    cls.add_method('CalculateAllocationStartTime', 
                   'uint32_t', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): void ns3::UplinkSchedulerSimple::GetChannelDescriptorsToUpdate(bool & updateDcd, bool & updateUcd, bool & sendDcd, bool & sendUcd) [member function]
    cls.add_method('GetChannelDescriptorsToUpdate', 
                   'void', 
                   [param('bool &', 'updateDcd'), param('bool &', 'updateUcd'), param('bool &', 'sendDcd'), param('bool &', 'sendUcd')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): static ns3::TypeId ns3::UplinkSchedulerSimple::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): std::list<ns3::OfdmUlMapIe, std::allocator<ns3::OfdmUlMapIe>> ns3::UplinkSchedulerSimple::GetUplinkAllocations() const [member function]
    cls.add_method('GetUplinkAllocations', 
                   'std::list< ns3::OfdmUlMapIe >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): void ns3::UplinkSchedulerSimple::InitOnce() [member function]
    cls.add_method('InitOnce', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): void ns3::UplinkSchedulerSimple::OnSetRequestedBandwidth(ns3::ServiceFlowRecord * sfr) [member function]
    cls.add_method('OnSetRequestedBandwidth', 
                   'void', 
                   [param('ns3::ServiceFlowRecord *', 'sfr')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): void ns3::UplinkSchedulerSimple::ProcessBandwidthRequest(ns3::BandwidthRequestHeader const & bwRequestHdr) [member function]
    cls.add_method('ProcessBandwidthRequest', 
                   'void', 
                   [param('ns3::BandwidthRequestHeader const &', 'bwRequestHdr')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): void ns3::UplinkSchedulerSimple::Schedule() [member function]
    cls.add_method('Schedule', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): void ns3::UplinkSchedulerSimple::ServiceBandwidthRequests(ns3::SSRecord const * ssRecord, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceBandwidthRequests', 
                   'void', 
                   [param('ns3::SSRecord const *', 'ssRecord'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): bool ns3::UplinkSchedulerSimple::ServiceBandwidthRequests(ns3::ServiceFlow * serviceFlow, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceBandwidthRequests', 
                   'bool', 
                   [param('ns3::ServiceFlow *', 'serviceFlow'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): void ns3::UplinkSchedulerSimple::ServiceUnsolicitedGrants(ns3::SSRecord const * ssRecord, ns3::ServiceFlow::SchedulingType schedulingType, ns3::OfdmUlMapIe & ulMapIe, ns3::WimaxPhy::ModulationType const modulationType, uint32_t & symbolsToAllocation, uint32_t & availableSymbols) [member function]
    cls.add_method('ServiceUnsolicitedGrants', 
                   'void', 
                   [param('ns3::SSRecord const *', 'ssRecord'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType'), param('ns3::OfdmUlMapIe &', 'ulMapIe'), param('ns3::WimaxPhy::ModulationType const', 'modulationType'), param('uint32_t &', 'symbolsToAllocation'), param('uint32_t &', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-uplink-scheduler-simple.h (module 'wimax'): void ns3::UplinkSchedulerSimple::SetupServiceFlow(ns3::SSRecord * ssRecord, ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('SetupServiceFlow', 
                   'void', 
                   [param('ns3::SSRecord *', 'ssRecord'), param('ns3::ServiceFlow *', 'serviceFlow')], 
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

def register_Ns3WimaxConnection_methods(root_module, cls):
    ## wimax-connection.h (module 'wimax'): ns3::WimaxConnection::WimaxConnection(ns3::WimaxConnection const & arg0) [constructor]
    cls.add_constructor([param('ns3::WimaxConnection const &', 'arg0')])
    ## wimax-connection.h (module 'wimax'): ns3::WimaxConnection::WimaxConnection(ns3::Cid cid, ns3::Cid::Type type) [constructor]
    cls.add_constructor([param('ns3::Cid', 'cid'), param('ns3::Cid::Type', 'type')])
    ## wimax-connection.h (module 'wimax'): void ns3::WimaxConnection::ClearFragmentsQueue() [member function]
    cls.add_method('ClearFragmentsQueue', 
                   'void', 
                   [])
    ## wimax-connection.h (module 'wimax'): ns3::Ptr<ns3::Packet> ns3::WimaxConnection::Dequeue(ns3::MacHeaderType::HeaderType packetType=::ns3::MacHeaderType::HeaderType::HEADER_TYPE_GENERIC) [member function]
    cls.add_method('Dequeue', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType', default_value='::ns3::MacHeaderType::HeaderType::HEADER_TYPE_GENERIC')])
    ## wimax-connection.h (module 'wimax'): ns3::Ptr<ns3::Packet> ns3::WimaxConnection::Dequeue(ns3::MacHeaderType::HeaderType packetType, uint32_t availableByte) [member function]
    cls.add_method('Dequeue', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType'), param('uint32_t', 'availableByte')])
    ## wimax-connection.h (module 'wimax'): bool ns3::WimaxConnection::Enqueue(ns3::Ptr<ns3::Packet> packet, ns3::MacHeaderType const & hdrType, ns3::GenericMacHeader const & hdr) [member function]
    cls.add_method('Enqueue', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::MacHeaderType const &', 'hdrType'), param('ns3::GenericMacHeader const &', 'hdr')])
    ## wimax-connection.h (module 'wimax'): void ns3::WimaxConnection::FragmentEnqueue(ns3::Ptr<const ns3::Packet> fragment) [member function]
    cls.add_method('FragmentEnqueue', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'fragment')])
    ## wimax-connection.h (module 'wimax'): ns3::Cid ns3::WimaxConnection::GetCid() const [member function]
    cls.add_method('GetCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## wimax-connection.h (module 'wimax'): ns3::WimaxConnection::FragmentsQueue const ns3::WimaxConnection::GetFragmentsQueue() const [member function]
    cls.add_method('GetFragmentsQueue', 
                   'ns3::WimaxConnection::FragmentsQueue const', 
                   [], 
                   is_const=True)
    ## wimax-connection.h (module 'wimax'): ns3::Ptr<ns3::WimaxMacQueue> ns3::WimaxConnection::GetQueue() const [member function]
    cls.add_method('GetQueue', 
                   'ns3::Ptr< ns3::WimaxMacQueue >', 
                   [], 
                   is_const=True)
    ## wimax-connection.h (module 'wimax'): uint8_t ns3::WimaxConnection::GetSchedulingType() const [member function]
    cls.add_method('GetSchedulingType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-connection.h (module 'wimax'): ns3::ServiceFlow * ns3::WimaxConnection::GetServiceFlow() const [member function]
    cls.add_method('GetServiceFlow', 
                   'ns3::ServiceFlow *', 
                   [], 
                   is_const=True)
    ## wimax-connection.h (module 'wimax'): ns3::Cid::Type ns3::WimaxConnection::GetType() const [member function]
    cls.add_method('GetType', 
                   'ns3::Cid::Type', 
                   [], 
                   is_const=True)
    ## wimax-connection.h (module 'wimax'): static ns3::TypeId ns3::WimaxConnection::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-connection.h (module 'wimax'): std::string ns3::WimaxConnection::GetTypeStr() const [member function]
    cls.add_method('GetTypeStr', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## wimax-connection.h (module 'wimax'): bool ns3::WimaxConnection::HasPackets() const [member function]
    cls.add_method('HasPackets', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wimax-connection.h (module 'wimax'): bool ns3::WimaxConnection::HasPackets(ns3::MacHeaderType::HeaderType packetType) const [member function]
    cls.add_method('HasPackets', 
                   'bool', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')], 
                   is_const=True)
    ## wimax-connection.h (module 'wimax'): void ns3::WimaxConnection::SetServiceFlow(ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('SetServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow *', 'serviceFlow')])
    ## wimax-connection.h (module 'wimax'): void ns3::WimaxConnection::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3WimaxMacQueue_methods(root_module, cls):
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::WimaxMacQueue(ns3::WimaxMacQueue const & arg0) [constructor]
    cls.add_constructor([param('ns3::WimaxMacQueue const &', 'arg0')])
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::WimaxMacQueue() [constructor]
    cls.add_constructor([])
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::WimaxMacQueue(uint32_t maxSize) [constructor]
    cls.add_constructor([param('uint32_t', 'maxSize')])
    ## wimax-mac-queue.h (module 'wimax'): bool ns3::WimaxMacQueue::CheckForFragmentation(ns3::MacHeaderType::HeaderType packetType) [member function]
    cls.add_method('CheckForFragmentation', 
                   'bool', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')])
    ## wimax-mac-queue.h (module 'wimax'): ns3::Ptr<ns3::Packet> ns3::WimaxMacQueue::Dequeue(ns3::MacHeaderType::HeaderType packetType) [member function]
    cls.add_method('Dequeue', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')])
    ## wimax-mac-queue.h (module 'wimax'): ns3::Ptr<ns3::Packet> ns3::WimaxMacQueue::Dequeue(ns3::MacHeaderType::HeaderType packetType, uint32_t availableByte) [member function]
    cls.add_method('Dequeue', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType'), param('uint32_t', 'availableByte')])
    ## wimax-mac-queue.h (module 'wimax'): bool ns3::WimaxMacQueue::Enqueue(ns3::Ptr<ns3::Packet> packet, ns3::MacHeaderType const & hdrType, ns3::GenericMacHeader const & hdr) [member function]
    cls.add_method('Enqueue', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::MacHeaderType const &', 'hdrType'), param('ns3::GenericMacHeader const &', 'hdr')])
    ## wimax-mac-queue.h (module 'wimax'): uint32_t ns3::WimaxMacQueue::GetFirstPacketHdrSize(ns3::MacHeaderType::HeaderType packetType) [member function]
    cls.add_method('GetFirstPacketHdrSize', 
                   'uint32_t', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')])
    ## wimax-mac-queue.h (module 'wimax'): uint32_t ns3::WimaxMacQueue::GetFirstPacketPayloadSize(ns3::MacHeaderType::HeaderType packetType) [member function]
    cls.add_method('GetFirstPacketPayloadSize', 
                   'uint32_t', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')])
    ## wimax-mac-queue.h (module 'wimax'): uint32_t ns3::WimaxMacQueue::GetFirstPacketRequiredByte(ns3::MacHeaderType::HeaderType packetType) [member function]
    cls.add_method('GetFirstPacketRequiredByte', 
                   'uint32_t', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')])
    ## wimax-mac-queue.h (module 'wimax'): uint32_t ns3::WimaxMacQueue::GetMaxSize() const [member function]
    cls.add_method('GetMaxSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): uint32_t ns3::WimaxMacQueue::GetNBytes() const [member function]
    cls.add_method('GetNBytes', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::PacketQueue const & ns3::WimaxMacQueue::GetPacketQueue() const [member function]
    cls.add_method('GetPacketQueue', 
                   'ns3::WimaxMacQueue::PacketQueue const &', 
                   [], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): uint32_t ns3::WimaxMacQueue::GetQueueLengthWithMACOverhead() [member function]
    cls.add_method('GetQueueLengthWithMACOverhead', 
                   'uint32_t', 
                   [])
    ## wimax-mac-queue.h (module 'wimax'): uint32_t ns3::WimaxMacQueue::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): static ns3::TypeId ns3::WimaxMacQueue::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-mac-queue.h (module 'wimax'): bool ns3::WimaxMacQueue::IsEmpty() const [member function]
    cls.add_method('IsEmpty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): bool ns3::WimaxMacQueue::IsEmpty(ns3::MacHeaderType::HeaderType packetType) const [member function]
    cls.add_method('IsEmpty', 
                   'bool', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): ns3::Ptr<ns3::Packet> ns3::WimaxMacQueue::Peek(ns3::GenericMacHeader & hdr) const [member function]
    cls.add_method('Peek', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('ns3::GenericMacHeader &', 'hdr')], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): ns3::Ptr<ns3::Packet> ns3::WimaxMacQueue::Peek(ns3::GenericMacHeader & hdr, ns3::Time & timeStamp) const [member function]
    cls.add_method('Peek', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('ns3::GenericMacHeader &', 'hdr'), param('ns3::Time &', 'timeStamp')], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): ns3::Ptr<ns3::Packet> ns3::WimaxMacQueue::Peek(ns3::MacHeaderType::HeaderType packetType) const [member function]
    cls.add_method('Peek', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): ns3::Ptr<ns3::Packet> ns3::WimaxMacQueue::Peek(ns3::MacHeaderType::HeaderType packetType, ns3::Time & timeStamp) const [member function]
    cls.add_method('Peek', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType'), param('ns3::Time &', 'timeStamp')], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): void ns3::WimaxMacQueue::SetFragmentNumber(ns3::MacHeaderType::HeaderType packetType) [member function]
    cls.add_method('SetFragmentNumber', 
                   'void', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')])
    ## wimax-mac-queue.h (module 'wimax'): void ns3::WimaxMacQueue::SetFragmentOffset(ns3::MacHeaderType::HeaderType packetType, uint32_t offset) [member function]
    cls.add_method('SetFragmentOffset', 
                   'void', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType'), param('uint32_t', 'offset')])
    ## wimax-mac-queue.h (module 'wimax'): void ns3::WimaxMacQueue::SetFragmentation(ns3::MacHeaderType::HeaderType packetType) [member function]
    cls.add_method('SetFragmentation', 
                   'void', 
                   [param('ns3::MacHeaderType::HeaderType', 'packetType')])
    ## wimax-mac-queue.h (module 'wimax'): void ns3::WimaxMacQueue::SetMaxSize(uint32_t maxSize) [member function]
    cls.add_method('SetMaxSize', 
                   'void', 
                   [param('uint32_t', 'maxSize')])
    return

def register_Ns3WimaxMacQueueQueueElement_methods(root_module, cls):
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement::QueueElement(ns3::WimaxMacQueue::QueueElement const & arg0) [constructor]
    cls.add_constructor([param('ns3::WimaxMacQueue::QueueElement const &', 'arg0')])
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement::QueueElement() [constructor]
    cls.add_constructor([])
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement::QueueElement(ns3::Ptr<ns3::Packet> packet, ns3::MacHeaderType const & hdrType, ns3::GenericMacHeader const & hdr, ns3::Time timeStamp) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::MacHeaderType const &', 'hdrType'), param('ns3::GenericMacHeader const &', 'hdr'), param('ns3::Time', 'timeStamp')])
    ## wimax-mac-queue.h (module 'wimax'): uint32_t ns3::WimaxMacQueue::QueueElement::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-queue.h (module 'wimax'): void ns3::WimaxMacQueue::QueueElement::SetFragmentNumber() [member function]
    cls.add_method('SetFragmentNumber', 
                   'void', 
                   [])
    ## wimax-mac-queue.h (module 'wimax'): void ns3::WimaxMacQueue::QueueElement::SetFragmentOffset(uint32_t offset) [member function]
    cls.add_method('SetFragmentOffset', 
                   'void', 
                   [param('uint32_t', 'offset')])
    ## wimax-mac-queue.h (module 'wimax'): void ns3::WimaxMacQueue::QueueElement::SetFragmentation() [member function]
    cls.add_method('SetFragmentation', 
                   'void', 
                   [])
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement::m_fragmentNumber [variable]
    cls.add_instance_attribute('m_fragmentNumber', 'uint32_t', is_const=False)
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement::m_fragmentOffset [variable]
    cls.add_instance_attribute('m_fragmentOffset', 'uint32_t', is_const=False)
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement::m_fragmentation [variable]
    cls.add_instance_attribute('m_fragmentation', 'bool', is_const=False)
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement::m_hdr [variable]
    cls.add_instance_attribute('m_hdr', 'ns3::GenericMacHeader', is_const=False)
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement::m_hdrType [variable]
    cls.add_instance_attribute('m_hdrType', 'ns3::MacHeaderType', is_const=False)
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement::m_packet [variable]
    cls.add_instance_attribute('m_packet', 'ns3::Ptr< ns3::Packet >', is_const=False)
    ## wimax-mac-queue.h (module 'wimax'): ns3::WimaxMacQueue::QueueElement::m_timeStamp [variable]
    cls.add_instance_attribute('m_timeStamp', 'ns3::Time', is_const=False)
    return

def register_Ns3WimaxMacToMacHeader_methods(root_module, cls):
    ## wimax-mac-to-mac-header.h (module 'wimax'): ns3::WimaxMacToMacHeader::WimaxMacToMacHeader(ns3::WimaxMacToMacHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::WimaxMacToMacHeader const &', 'arg0')])
    ## wimax-mac-to-mac-header.h (module 'wimax'): ns3::WimaxMacToMacHeader::WimaxMacToMacHeader() [constructor]
    cls.add_constructor([])
    ## wimax-mac-to-mac-header.h (module 'wimax'): ns3::WimaxMacToMacHeader::WimaxMacToMacHeader(uint32_t len) [constructor]
    cls.add_constructor([param('uint32_t', 'len')])
    ## wimax-mac-to-mac-header.h (module 'wimax'): uint32_t ns3::WimaxMacToMacHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## wimax-mac-to-mac-header.h (module 'wimax'): ns3::TypeId ns3::WimaxMacToMacHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-to-mac-header.h (module 'wimax'): uint32_t ns3::WimaxMacToMacHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-to-mac-header.h (module 'wimax'): uint8_t ns3::WimaxMacToMacHeader::GetSizeOfLen() const [member function]
    cls.add_method('GetSizeOfLen', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-to-mac-header.h (module 'wimax'): static ns3::TypeId ns3::WimaxMacToMacHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-mac-to-mac-header.h (module 'wimax'): void ns3::WimaxMacToMacHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-to-mac-header.h (module 'wimax'): void ns3::WimaxMacToMacHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3WimaxPhy_methods(root_module, cls):
    ## wimax-phy.h (module 'wimax'): ns3::WimaxPhy::WimaxPhy(ns3::WimaxPhy const & arg0) [constructor]
    cls.add_constructor([param('ns3::WimaxPhy const &', 'arg0')])
    ## wimax-phy.h (module 'wimax'): ns3::WimaxPhy::WimaxPhy() [constructor]
    cls.add_constructor([])
    ## wimax-phy.h (module 'wimax'): int64_t ns3::WimaxPhy::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_pure_virtual=True, is_virtual=True)
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::Attach(ns3::Ptr<ns3::WimaxChannel> channel) [member function]
    cls.add_method('Attach', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxChannel >', 'channel')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## wimax-phy.h (module 'wimax'): ns3::Ptr<ns3::WimaxChannel> ns3::WimaxPhy::GetChannel() const [member function]
    cls.add_method('GetChannel', 
                   'ns3::Ptr< ns3::WimaxChannel >', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint32_t ns3::WimaxPhy::GetChannelBandwidth() const [member function]
    cls.add_method('GetChannelBandwidth', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::EventId ns3::WimaxPhy::GetChnlSrchTimeoutEvent() const [member function]
    cls.add_method('GetChnlSrchTimeoutEvent', 
                   'ns3::EventId', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint32_t ns3::WimaxPhy::GetDataRate(ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('GetDataRate', 
                   'uint32_t', 
                   [param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::Ptr<ns3::NetDevice> ns3::WimaxPhy::GetDevice() const [member function]
    cls.add_method('GetDevice', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::Time ns3::WimaxPhy::GetFrameDuration() const [member function]
    cls.add_method('GetFrameDuration', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::Time ns3::WimaxPhy::GetFrameDuration(uint8_t frameDurationCode) const [member function]
    cls.add_method('GetFrameDuration', 
                   'ns3::Time', 
                   [param('uint8_t', 'frameDurationCode')], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint8_t ns3::WimaxPhy::GetFrameDurationCode() const [member function]
    cls.add_method('GetFrameDurationCode', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::Time ns3::WimaxPhy::GetFrameDurationSec() const [member function]
    cls.add_method('GetFrameDurationSec', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint32_t ns3::WimaxPhy::GetFrequency() const [member function]
    cls.add_method('GetFrequency', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): double ns3::WimaxPhy::GetGValue() const [member function]
    cls.add_method('GetGValue', 
                   'double', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::Ptr<ns3::Object> ns3::WimaxPhy::GetMobility() [member function]
    cls.add_method('GetMobility', 
                   'ns3::Ptr< ns3::Object >', 
                   [], 
                   is_virtual=True)
    ## wimax-phy.h (module 'wimax'): uint16_t ns3::WimaxPhy::GetNfft() const [member function]
    cls.add_method('GetNfft', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint64_t ns3::WimaxPhy::GetNrBytes(uint32_t symbols, ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('GetNrBytes', 
                   'uint64_t', 
                   [param('uint32_t', 'symbols'), param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint8_t ns3::WimaxPhy::GetNrCarriers() const [member function]
    cls.add_method('GetNrCarriers', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint64_t ns3::WimaxPhy::GetNrSymbols(uint32_t size, ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('GetNrSymbols', 
                   'uint64_t', 
                   [param('uint32_t', 'size'), param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::WimaxPhy::PhyType ns3::WimaxPhy::GetPhyType() const [member function]
    cls.add_method('GetPhyType', 
                   'ns3::WimaxPhy::PhyType', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## wimax-phy.h (module 'wimax'): ns3::Time ns3::WimaxPhy::GetPsDuration() const [member function]
    cls.add_method('GetPsDuration', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint16_t ns3::WimaxPhy::GetPsPerFrame() const [member function]
    cls.add_method('GetPsPerFrame', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint16_t ns3::WimaxPhy::GetPsPerSymbol() const [member function]
    cls.add_method('GetPsPerSymbol', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::Callback<void, ns3::Ptr<const ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> ns3::WimaxPhy::GetReceiveCallback() const [member function]
    cls.add_method('GetReceiveCallback', 
                   'ns3::Callback< void, ns3::Ptr< ns3::PacketBurst const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint16_t ns3::WimaxPhy::GetRtg() const [member function]
    cls.add_method('GetRtg', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint64_t ns3::WimaxPhy::GetRxFrequency() const [member function]
    cls.add_method('GetRxFrequency', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): double ns3::WimaxPhy::GetSamplingFactor() const [member function]
    cls.add_method('GetSamplingFactor', 
                   'double', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): double ns3::WimaxPhy::GetSamplingFrequency() const [member function]
    cls.add_method('GetSamplingFrequency', 
                   'double', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint64_t ns3::WimaxPhy::GetScanningFrequency() const [member function]
    cls.add_method('GetScanningFrequency', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::WimaxPhy::PhyState ns3::WimaxPhy::GetState() const [member function]
    cls.add_method('GetState', 
                   'ns3::WimaxPhy::PhyState', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::Time ns3::WimaxPhy::GetSymbolDuration() const [member function]
    cls.add_method('GetSymbolDuration', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint32_t ns3::WimaxPhy::GetSymbolsPerFrame() const [member function]
    cls.add_method('GetSymbolsPerFrame', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): ns3::Time ns3::WimaxPhy::GetTransmissionTime(uint32_t size, ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('GetTransmissionTime', 
                   'ns3::Time', 
                   [param('uint32_t', 'size'), param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint16_t ns3::WimaxPhy::GetTtg() const [member function]
    cls.add_method('GetTtg', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): uint64_t ns3::WimaxPhy::GetTxFrequency() const [member function]
    cls.add_method('GetTxFrequency', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): static ns3::TypeId ns3::WimaxPhy::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-phy.h (module 'wimax'): bool ns3::WimaxPhy::IsDuplex() const [member function]
    cls.add_method('IsDuplex', 
                   'bool', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::Send(ns3::SendParams * params) [member function]
    cls.add_method('Send', 
                   'void', 
                   [param('ns3::SendParams *', 'params')], 
                   is_pure_virtual=True, is_virtual=True)
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetChannelBandwidth(uint32_t channelBandwidth) [member function]
    cls.add_method('SetChannelBandwidth', 
                   'void', 
                   [param('uint32_t', 'channelBandwidth')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetDataRates() [member function]
    cls.add_method('SetDataRates', 
                   'void', 
                   [])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetDevice(ns3::Ptr<ns3::WimaxNetDevice> device) [member function]
    cls.add_method('SetDevice', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxNetDevice >', 'device')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetDuplex(uint64_t rxFrequency, uint64_t txFrequency) [member function]
    cls.add_method('SetDuplex', 
                   'void', 
                   [param('uint64_t', 'rxFrequency'), param('uint64_t', 'txFrequency')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetFrameDuration(ns3::Time frameDuration) [member function]
    cls.add_method('SetFrameDuration', 
                   'void', 
                   [param('ns3::Time', 'frameDuration')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetFrequency(uint32_t frequency) [member function]
    cls.add_method('SetFrequency', 
                   'void', 
                   [param('uint32_t', 'frequency')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetMobility(ns3::Ptr<ns3::Object> mobility) [member function]
    cls.add_method('SetMobility', 
                   'void', 
                   [param('ns3::Ptr< ns3::Object >', 'mobility')], 
                   is_virtual=True)
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetNrCarriers(uint8_t nrCarriers) [member function]
    cls.add_method('SetNrCarriers', 
                   'void', 
                   [param('uint8_t', 'nrCarriers')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetPhyParameters() [member function]
    cls.add_method('SetPhyParameters', 
                   'void', 
                   [])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetPsDuration(ns3::Time psDuration) [member function]
    cls.add_method('SetPsDuration', 
                   'void', 
                   [param('ns3::Time', 'psDuration')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetPsPerFrame(uint16_t psPerFrame) [member function]
    cls.add_method('SetPsPerFrame', 
                   'void', 
                   [param('uint16_t', 'psPerFrame')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetPsPerSymbol(uint16_t psPerSymbol) [member function]
    cls.add_method('SetPsPerSymbol', 
                   'void', 
                   [param('uint16_t', 'psPerSymbol')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetReceiveCallback(ns3::Callback<void, ns3::Ptr<const ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> callback) [member function]
    cls.add_method('SetReceiveCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::PacketBurst const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'callback')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetScanningCallback() const [member function]
    cls.add_method('SetScanningCallback', 
                   'void', 
                   [], 
                   is_const=True)
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetSimplex(uint64_t frequency) [member function]
    cls.add_method('SetSimplex', 
                   'void', 
                   [param('uint64_t', 'frequency')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetState(ns3::WimaxPhy::PhyState state) [member function]
    cls.add_method('SetState', 
                   'void', 
                   [param('ns3::WimaxPhy::PhyState', 'state')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetSymbolDuration(ns3::Time symbolDuration) [member function]
    cls.add_method('SetSymbolDuration', 
                   'void', 
                   [param('ns3::Time', 'symbolDuration')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::SetSymbolsPerFrame(uint32_t symbolsPerFrame) [member function]
    cls.add_method('SetSymbolsPerFrame', 
                   'void', 
                   [param('uint32_t', 'symbolsPerFrame')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::StartScanning(uint64_t frequency, ns3::Time timeout, ns3::Callback<void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> callback) [member function]
    cls.add_method('StartScanning', 
                   'void', 
                   [param('uint64_t', 'frequency'), param('ns3::Time', 'timeout'), param('ns3::Callback< void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'callback')])
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::DoAttach(ns3::Ptr<ns3::WimaxChannel> channel) [member function]
    cls.add_method('DoAttach', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxChannel >', 'channel')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): uint32_t ns3::WimaxPhy::DoGetDataRate(ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('DoGetDataRate', 
                   'uint32_t', 
                   [param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): ns3::Time ns3::WimaxPhy::DoGetFrameDuration(uint8_t frameDurationCode) const [member function]
    cls.add_method('DoGetFrameDuration', 
                   'ns3::Time', 
                   [param('uint8_t', 'frameDurationCode')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): uint8_t ns3::WimaxPhy::DoGetFrameDurationCode() const [member function]
    cls.add_method('DoGetFrameDurationCode', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): double ns3::WimaxPhy::DoGetGValue() const [member function]
    cls.add_method('DoGetGValue', 
                   'double', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): uint16_t ns3::WimaxPhy::DoGetNfft() const [member function]
    cls.add_method('DoGetNfft', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): uint64_t ns3::WimaxPhy::DoGetNrBytes(uint32_t symbols, ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('DoGetNrBytes', 
                   'uint64_t', 
                   [param('uint32_t', 'symbols'), param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): uint64_t ns3::WimaxPhy::DoGetNrSymbols(uint32_t size, ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('DoGetNrSymbols', 
                   'uint64_t', 
                   [param('uint32_t', 'size'), param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): uint16_t ns3::WimaxPhy::DoGetRtg() const [member function]
    cls.add_method('DoGetRtg', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): double ns3::WimaxPhy::DoGetSamplingFactor() const [member function]
    cls.add_method('DoGetSamplingFactor', 
                   'double', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): double ns3::WimaxPhy::DoGetSamplingFrequency() const [member function]
    cls.add_method('DoGetSamplingFrequency', 
                   'double', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): ns3::Time ns3::WimaxPhy::DoGetTransmissionTime(uint32_t size, ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('DoGetTransmissionTime', 
                   'ns3::Time', 
                   [param('uint32_t', 'size'), param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): uint16_t ns3::WimaxPhy::DoGetTtg() const [member function]
    cls.add_method('DoGetTtg', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::DoSetDataRates() [member function]
    cls.add_method('DoSetDataRates', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-phy.h (module 'wimax'): void ns3::WimaxPhy::DoSetPhyParameters() [member function]
    cls.add_method('DoSetPhyParameters', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
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

def register_Ns3BSScheduler_methods(root_module, cls):
    ## bs-scheduler.h (module 'wimax'): ns3::BSScheduler::BSScheduler(ns3::BSScheduler const & arg0) [constructor]
    cls.add_constructor([param('ns3::BSScheduler const &', 'arg0')])
    ## bs-scheduler.h (module 'wimax'): ns3::BSScheduler::BSScheduler() [constructor]
    cls.add_constructor([])
    ## bs-scheduler.h (module 'wimax'): ns3::BSScheduler::BSScheduler(ns3::Ptr<ns3::BaseStationNetDevice> bs) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::BaseStationNetDevice >', 'bs')])
    ## bs-scheduler.h (module 'wimax'): void ns3::BSScheduler::AddDownlinkBurst(ns3::Ptr<const ns3::WimaxConnection> connection, uint8_t diuc, ns3::WimaxPhy::ModulationType modulationType, ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('AddDownlinkBurst', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxConnection const >', 'connection'), param('uint8_t', 'diuc'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('ns3::Ptr< ns3::PacketBurst >', 'burst')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-scheduler.h (module 'wimax'): bool ns3::BSScheduler::CheckForFragmentation(ns3::Ptr<ns3::WimaxConnection> connection, int availableSymbols, ns3::WimaxPhy::ModulationType modulationType) [member function]
    cls.add_method('CheckForFragmentation', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WimaxConnection >', 'connection'), param('int', 'availableSymbols'), param('ns3::WimaxPhy::ModulationType', 'modulationType')])
    ## bs-scheduler.h (module 'wimax'): ns3::Ptr<ns3::PacketBurst> ns3::BSScheduler::CreateUgsBurst(ns3::ServiceFlow * serviceFlow, ns3::WimaxPhy::ModulationType modulationType, uint32_t availableSymbols) [member function]
    cls.add_method('CreateUgsBurst', 
                   'ns3::Ptr< ns3::PacketBurst >', 
                   [param('ns3::ServiceFlow *', 'serviceFlow'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('uint32_t', 'availableSymbols')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-scheduler.h (module 'wimax'): ns3::Ptr<ns3::BaseStationNetDevice> ns3::BSScheduler::GetBs() [member function]
    cls.add_method('GetBs', 
                   'ns3::Ptr< ns3::BaseStationNetDevice >', 
                   [], 
                   is_virtual=True)
    ## bs-scheduler.h (module 'wimax'): std::list<std::pair<ns3::OfdmDlMapIe *, ns3::Ptr<ns3::PacketBurst>>, std::allocator<std::pair<ns3::OfdmDlMapIe *, ns3::Ptr<ns3::PacketBurst>>>> * ns3::BSScheduler::GetDownlinkBursts() const [member function]
    cls.add_method('GetDownlinkBursts', 
                   'std::list< std::pair< ns3::OfdmDlMapIe *, ns3::Ptr< ns3::PacketBurst > > > *', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## bs-scheduler.h (module 'wimax'): static ns3::TypeId ns3::BSScheduler::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## bs-scheduler.h (module 'wimax'): void ns3::BSScheduler::Schedule() [member function]
    cls.add_method('Schedule', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-scheduler.h (module 'wimax'): bool ns3::BSScheduler::SelectConnection(ns3::Ptr<ns3::WimaxConnection> & connection) [member function]
    cls.add_method('SelectConnection', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WimaxConnection > &', 'connection')], 
                   is_pure_virtual=True, is_virtual=True)
    ## bs-scheduler.h (module 'wimax'): void ns3::BSScheduler::SetBs(ns3::Ptr<ns3::BaseStationNetDevice> bs) [member function]
    cls.add_method('SetBs', 
                   'void', 
                   [param('ns3::Ptr< ns3::BaseStationNetDevice >', 'bs')], 
                   is_virtual=True)
    return

def register_Ns3BSSchedulerRtps_methods(root_module, cls):
    ## bs-scheduler-rtps.h (module 'wimax'): ns3::BSSchedulerRtps::BSSchedulerRtps(ns3::BSSchedulerRtps const & arg0) [constructor]
    cls.add_constructor([param('ns3::BSSchedulerRtps const &', 'arg0')])
    ## bs-scheduler-rtps.h (module 'wimax'): ns3::BSSchedulerRtps::BSSchedulerRtps() [constructor]
    cls.add_constructor([])
    ## bs-scheduler-rtps.h (module 'wimax'): ns3::BSSchedulerRtps::BSSchedulerRtps(ns3::Ptr<ns3::BaseStationNetDevice> bs) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::BaseStationNetDevice >', 'bs')])
    ## bs-scheduler-rtps.h (module 'wimax'): void ns3::BSSchedulerRtps::AddDownlinkBurst(ns3::Ptr<const ns3::WimaxConnection> connection, uint8_t diuc, ns3::WimaxPhy::ModulationType modulationType, ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('AddDownlinkBurst', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxConnection const >', 'connection'), param('uint8_t', 'diuc'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('ns3::Ptr< ns3::PacketBurst >', 'burst')], 
                   is_virtual=True)
    ## bs-scheduler-rtps.h (module 'wimax'): void ns3::BSSchedulerRtps::BSSchedulerBEConnection(uint32_t & availableSymbols) [member function]
    cls.add_method('BSSchedulerBEConnection', 
                   'void', 
                   [param('uint32_t &', 'availableSymbols')])
    ## bs-scheduler-rtps.h (module 'wimax'): void ns3::BSSchedulerRtps::BSSchedulerBasicConnection(uint32_t & availableSymbols) [member function]
    cls.add_method('BSSchedulerBasicConnection', 
                   'void', 
                   [param('uint32_t &', 'availableSymbols')])
    ## bs-scheduler-rtps.h (module 'wimax'): void ns3::BSSchedulerRtps::BSSchedulerBroadcastConnection(uint32_t & availableSymbols) [member function]
    cls.add_method('BSSchedulerBroadcastConnection', 
                   'void', 
                   [param('uint32_t &', 'availableSymbols')])
    ## bs-scheduler-rtps.h (module 'wimax'): void ns3::BSSchedulerRtps::BSSchedulerInitialRangingConnection(uint32_t & availableSymbols) [member function]
    cls.add_method('BSSchedulerInitialRangingConnection', 
                   'void', 
                   [param('uint32_t &', 'availableSymbols')])
    ## bs-scheduler-rtps.h (module 'wimax'): void ns3::BSSchedulerRtps::BSSchedulerNRTPSConnection(uint32_t & availableSymbols) [member function]
    cls.add_method('BSSchedulerNRTPSConnection', 
                   'void', 
                   [param('uint32_t &', 'availableSymbols')])
    ## bs-scheduler-rtps.h (module 'wimax'): void ns3::BSSchedulerRtps::BSSchedulerPrimaryConnection(uint32_t & availableSymbols) [member function]
    cls.add_method('BSSchedulerPrimaryConnection', 
                   'void', 
                   [param('uint32_t &', 'availableSymbols')])
    ## bs-scheduler-rtps.h (module 'wimax'): void ns3::BSSchedulerRtps::BSSchedulerRTPSConnection(uint32_t & availableSymbols) [member function]
    cls.add_method('BSSchedulerRTPSConnection', 
                   'void', 
                   [param('uint32_t &', 'availableSymbols')])
    ## bs-scheduler-rtps.h (module 'wimax'): void ns3::BSSchedulerRtps::BSSchedulerUGSConnection(uint32_t & availableSymbols) [member function]
    cls.add_method('BSSchedulerUGSConnection', 
                   'void', 
                   [param('uint32_t &', 'availableSymbols')])
    ## bs-scheduler-rtps.h (module 'wimax'): ns3::Ptr<ns3::PacketBurst> ns3::BSSchedulerRtps::CreateUgsBurst(ns3::ServiceFlow * serviceFlow, ns3::WimaxPhy::ModulationType modulationType, uint32_t availableSymbols) [member function]
    cls.add_method('CreateUgsBurst', 
                   'ns3::Ptr< ns3::PacketBurst >', 
                   [param('ns3::ServiceFlow *', 'serviceFlow'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('uint32_t', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-scheduler-rtps.h (module 'wimax'): std::list<std::pair<ns3::OfdmDlMapIe *, ns3::Ptr<ns3::PacketBurst>>, std::allocator<std::pair<ns3::OfdmDlMapIe *, ns3::Ptr<ns3::PacketBurst>>>> * ns3::BSSchedulerRtps::GetDownlinkBursts() const [member function]
    cls.add_method('GetDownlinkBursts', 
                   'std::list< std::pair< ns3::OfdmDlMapIe *, ns3::Ptr< ns3::PacketBurst > > > *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-scheduler-rtps.h (module 'wimax'): static ns3::TypeId ns3::BSSchedulerRtps::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## bs-scheduler-rtps.h (module 'wimax'): void ns3::BSSchedulerRtps::Schedule() [member function]
    cls.add_method('Schedule', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-scheduler-rtps.h (module 'wimax'): bool ns3::BSSchedulerRtps::SelectBEConnection(ns3::Ptr<ns3::WimaxConnection> & connection) [member function]
    cls.add_method('SelectBEConnection', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WimaxConnection > &', 'connection')])
    ## bs-scheduler-rtps.h (module 'wimax'): bool ns3::BSSchedulerRtps::SelectConnection(ns3::Ptr<ns3::WimaxConnection> & connection) [member function]
    cls.add_method('SelectConnection', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WimaxConnection > &', 'connection')], 
                   is_virtual=True)
    ## bs-scheduler-rtps.h (module 'wimax'): bool ns3::BSSchedulerRtps::SelectIRandBCConnection(ns3::Ptr<ns3::WimaxConnection> & connection) [member function]
    cls.add_method('SelectIRandBCConnection', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WimaxConnection > &', 'connection')])
    ## bs-scheduler-rtps.h (module 'wimax'): bool ns3::BSSchedulerRtps::SelectMenagementConnection(ns3::Ptr<ns3::WimaxConnection> & connection) [member function]
    cls.add_method('SelectMenagementConnection', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WimaxConnection > &', 'connection')])
    ## bs-scheduler-rtps.h (module 'wimax'): bool ns3::BSSchedulerRtps::SelectNRTPSConnection(ns3::Ptr<ns3::WimaxConnection> & connection) [member function]
    cls.add_method('SelectNRTPSConnection', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WimaxConnection > &', 'connection')])
    ## bs-scheduler-rtps.h (module 'wimax'): bool ns3::BSSchedulerRtps::SelectRTPSConnection(ns3::Ptr<ns3::WimaxConnection> & connection) [member function]
    cls.add_method('SelectRTPSConnection', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WimaxConnection > &', 'connection')])
    ## bs-scheduler-rtps.h (module 'wimax'): bool ns3::BSSchedulerRtps::SelectUGSConnection(ns3::Ptr<ns3::WimaxConnection> & connection) [member function]
    cls.add_method('SelectUGSConnection', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WimaxConnection > &', 'connection')])
    return

def register_Ns3BSSchedulerSimple_methods(root_module, cls):
    ## bs-scheduler-simple.h (module 'wimax'): ns3::BSSchedulerSimple::BSSchedulerSimple(ns3::BSSchedulerSimple const & arg0) [constructor]
    cls.add_constructor([param('ns3::BSSchedulerSimple const &', 'arg0')])
    ## bs-scheduler-simple.h (module 'wimax'): ns3::BSSchedulerSimple::BSSchedulerSimple() [constructor]
    cls.add_constructor([])
    ## bs-scheduler-simple.h (module 'wimax'): ns3::BSSchedulerSimple::BSSchedulerSimple(ns3::Ptr<ns3::BaseStationNetDevice> bs) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::BaseStationNetDevice >', 'bs')])
    ## bs-scheduler-simple.h (module 'wimax'): void ns3::BSSchedulerSimple::AddDownlinkBurst(ns3::Ptr<const ns3::WimaxConnection> connection, uint8_t diuc, ns3::WimaxPhy::ModulationType modulationType, ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('AddDownlinkBurst', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxConnection const >', 'connection'), param('uint8_t', 'diuc'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('ns3::Ptr< ns3::PacketBurst >', 'burst')], 
                   is_virtual=True)
    ## bs-scheduler-simple.h (module 'wimax'): ns3::Ptr<ns3::PacketBurst> ns3::BSSchedulerSimple::CreateUgsBurst(ns3::ServiceFlow * serviceFlow, ns3::WimaxPhy::ModulationType modulationType, uint32_t availableSymbols) [member function]
    cls.add_method('CreateUgsBurst', 
                   'ns3::Ptr< ns3::PacketBurst >', 
                   [param('ns3::ServiceFlow *', 'serviceFlow'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('uint32_t', 'availableSymbols')], 
                   is_virtual=True)
    ## bs-scheduler-simple.h (module 'wimax'): std::list<std::pair<ns3::OfdmDlMapIe *, ns3::Ptr<ns3::PacketBurst>>, std::allocator<std::pair<ns3::OfdmDlMapIe *, ns3::Ptr<ns3::PacketBurst>>>> * ns3::BSSchedulerSimple::GetDownlinkBursts() const [member function]
    cls.add_method('GetDownlinkBursts', 
                   'std::list< std::pair< ns3::OfdmDlMapIe *, ns3::Ptr< ns3::PacketBurst > > > *', 
                   [], 
                   is_const=True, is_virtual=True)
    ## bs-scheduler-simple.h (module 'wimax'): static ns3::TypeId ns3::BSSchedulerSimple::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## bs-scheduler-simple.h (module 'wimax'): void ns3::BSSchedulerSimple::Schedule() [member function]
    cls.add_method('Schedule', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-scheduler-simple.h (module 'wimax'): bool ns3::BSSchedulerSimple::SelectConnection(ns3::Ptr<ns3::WimaxConnection> & connection) [member function]
    cls.add_method('SelectConnection', 
                   'bool', 
                   [param('ns3::Ptr< ns3::WimaxConnection > &', 'connection')], 
                   is_virtual=True)
    return

def register_Ns3BandwidthRequestHeader_methods(root_module, cls):
    ## wimax-mac-header.h (module 'wimax'): ns3::BandwidthRequestHeader::BandwidthRequestHeader(ns3::BandwidthRequestHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::BandwidthRequestHeader const &', 'arg0')])
    ## wimax-mac-header.h (module 'wimax'): ns3::BandwidthRequestHeader::BandwidthRequestHeader() [constructor]
    cls.add_constructor([])
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::BandwidthRequestHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::BandwidthRequestHeader::GetBr() const [member function]
    cls.add_method('GetBr', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): ns3::Cid ns3::BandwidthRequestHeader::GetCid() const [member function]
    cls.add_method('GetCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::BandwidthRequestHeader::GetEc() const [member function]
    cls.add_method('GetEc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::BandwidthRequestHeader::GetHcs() const [member function]
    cls.add_method('GetHcs', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::BandwidthRequestHeader::GetHt() const [member function]
    cls.add_method('GetHt', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): ns3::TypeId ns3::BandwidthRequestHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): std::string ns3::BandwidthRequestHeader::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::BandwidthRequestHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::BandwidthRequestHeader::GetType() const [member function]
    cls.add_method('GetType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): static ns3::TypeId ns3::BandwidthRequestHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::BandwidthRequestHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::BandwidthRequestHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::BandwidthRequestHeader::SetBr(uint32_t br) [member function]
    cls.add_method('SetBr', 
                   'void', 
                   [param('uint32_t', 'br')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::BandwidthRequestHeader::SetCid(ns3::Cid cid) [member function]
    cls.add_method('SetCid', 
                   'void', 
                   [param('ns3::Cid', 'cid')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::BandwidthRequestHeader::SetEc(uint8_t ec) [member function]
    cls.add_method('SetEc', 
                   'void', 
                   [param('uint8_t', 'ec')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::BandwidthRequestHeader::SetHcs(uint8_t hcs) [member function]
    cls.add_method('SetHcs', 
                   'void', 
                   [param('uint8_t', 'hcs')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::BandwidthRequestHeader::SetHt(uint8_t ht) [member function]
    cls.add_method('SetHt', 
                   'void', 
                   [param('uint8_t', 'ht')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::BandwidthRequestHeader::SetType(uint8_t type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('uint8_t', 'type')])
    ## wimax-mac-header.h (module 'wimax'): bool ns3::BandwidthRequestHeader::check_hcs() const [member function]
    cls.add_method('check_hcs', 
                   'bool', 
                   [], 
                   is_const=True)
    return

def register_Ns3BsServiceFlowManager_methods(root_module, cls):
    ## bs-service-flow-manager.h (module 'wimax'): ns3::BsServiceFlowManager::BsServiceFlowManager(ns3::BsServiceFlowManager const & arg0) [constructor]
    cls.add_constructor([param('ns3::BsServiceFlowManager const &', 'arg0')])
    ## bs-service-flow-manager.h (module 'wimax'): ns3::BsServiceFlowManager::BsServiceFlowManager(ns3::Ptr<ns3::BaseStationNetDevice> device) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::BaseStationNetDevice >', 'device')])
    ## bs-service-flow-manager.h (module 'wimax'): void ns3::BsServiceFlowManager::AddMulticastServiceFlow(ns3::ServiceFlow sf, ns3::WimaxPhy::ModulationType modulation) [member function]
    cls.add_method('AddMulticastServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow', 'sf'), param('ns3::WimaxPhy::ModulationType', 'modulation')])
    ## bs-service-flow-manager.h (module 'wimax'): void ns3::BsServiceFlowManager::AddServiceFlow(ns3::ServiceFlow * serviceFlow) [member function]
    cls.add_method('AddServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow *', 'serviceFlow')])
    ## bs-service-flow-manager.h (module 'wimax'): void ns3::BsServiceFlowManager::AllocateServiceFlows(ns3::DsaReq const & dsaReq, ns3::Cid cid) [member function]
    cls.add_method('AllocateServiceFlows', 
                   'void', 
                   [param('ns3::DsaReq const &', 'dsaReq'), param('ns3::Cid', 'cid')])
    ## bs-service-flow-manager.h (module 'wimax'): void ns3::BsServiceFlowManager::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-service-flow-manager.h (module 'wimax'): ns3::EventId ns3::BsServiceFlowManager::GetDsaAckTimeoutEvent() const [member function]
    cls.add_method('GetDsaAckTimeoutEvent', 
                   'ns3::EventId', 
                   [], 
                   is_const=True)
    ## bs-service-flow-manager.h (module 'wimax'): ns3::ServiceFlow * ns3::BsServiceFlowManager::GetServiceFlow(uint32_t sfid) const [member function]
    cls.add_method('GetServiceFlow', 
                   'ns3::ServiceFlow *', 
                   [param('uint32_t', 'sfid')], 
                   is_const=True)
    ## bs-service-flow-manager.h (module 'wimax'): ns3::ServiceFlow * ns3::BsServiceFlowManager::GetServiceFlow(ns3::Cid cid) const [member function]
    cls.add_method('GetServiceFlow', 
                   'ns3::ServiceFlow *', 
                   [param('ns3::Cid', 'cid')], 
                   is_const=True)
    ## bs-service-flow-manager.h (module 'wimax'): std::vector<ns3::ServiceFlow *, std::allocator<ns3::ServiceFlow *>> ns3::BsServiceFlowManager::GetServiceFlows(ns3::ServiceFlow::SchedulingType schedulingType) const [member function]
    cls.add_method('GetServiceFlows', 
                   'std::vector< ns3::ServiceFlow * >', 
                   [param('ns3::ServiceFlow::SchedulingType', 'schedulingType')], 
                   is_const=True)
    ## bs-service-flow-manager.h (module 'wimax'): static ns3::TypeId ns3::BsServiceFlowManager::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## bs-service-flow-manager.h (module 'wimax'): void ns3::BsServiceFlowManager::ProcessDsaAck(ns3::DsaAck const & dsaAck, ns3::Cid cid) [member function]
    cls.add_method('ProcessDsaAck', 
                   'void', 
                   [param('ns3::DsaAck const &', 'dsaAck'), param('ns3::Cid', 'cid')])
    ## bs-service-flow-manager.h (module 'wimax'): ns3::ServiceFlow * ns3::BsServiceFlowManager::ProcessDsaReq(ns3::DsaReq const & dsaReq, ns3::Cid cid) [member function]
    cls.add_method('ProcessDsaReq', 
                   'ns3::ServiceFlow *', 
                   [param('ns3::DsaReq const &', 'dsaReq'), param('ns3::Cid', 'cid')])
    ## bs-service-flow-manager.h (module 'wimax'): void ns3::BsServiceFlowManager::SetMaxDsaRspRetries(uint8_t maxDsaRspRetries) [member function]
    cls.add_method('SetMaxDsaRspRetries', 
                   'void', 
                   [param('uint8_t', 'maxDsaRspRetries')])
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
                   is_static=True, template_parameters=['ns3::Ptr<ns3::PacketBurst const> '], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['bool'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['unsigned long long'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Mac48Address const&'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Mac48Address'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Cid const&'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Cid'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Ptr<ns3::PacketBurst> '], visibility='protected')
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

def register_Ns3ConnectionManager_methods(root_module, cls):
    ## connection-manager.h (module 'wimax'): ns3::ConnectionManager::ConnectionManager(ns3::ConnectionManager const & arg0) [constructor]
    cls.add_constructor([param('ns3::ConnectionManager const &', 'arg0')])
    ## connection-manager.h (module 'wimax'): ns3::ConnectionManager::ConnectionManager() [constructor]
    cls.add_constructor([])
    ## connection-manager.h (module 'wimax'): void ns3::ConnectionManager::AddConnection(ns3::Ptr<ns3::WimaxConnection> connection, ns3::Cid::Type type) [member function]
    cls.add_method('AddConnection', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxConnection >', 'connection'), param('ns3::Cid::Type', 'type')])
    ## connection-manager.h (module 'wimax'): void ns3::ConnectionManager::AllocateManagementConnections(ns3::SSRecord * ssRecord, ns3::RngRsp * rngrsp) [member function]
    cls.add_method('AllocateManagementConnections', 
                   'void', 
                   [param('ns3::SSRecord *', 'ssRecord'), param('ns3::RngRsp *', 'rngrsp')])
    ## connection-manager.h (module 'wimax'): ns3::Ptr<ns3::WimaxConnection> ns3::ConnectionManager::CreateConnection(ns3::Cid::Type type) [member function]
    cls.add_method('CreateConnection', 
                   'ns3::Ptr< ns3::WimaxConnection >', 
                   [param('ns3::Cid::Type', 'type')])
    ## connection-manager.h (module 'wimax'): void ns3::ConnectionManager::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## connection-manager.h (module 'wimax'): ns3::Ptr<ns3::WimaxConnection> ns3::ConnectionManager::GetConnection(ns3::Cid cid) [member function]
    cls.add_method('GetConnection', 
                   'ns3::Ptr< ns3::WimaxConnection >', 
                   [param('ns3::Cid', 'cid')])
    ## connection-manager.h (module 'wimax'): std::vector<ns3::Ptr<ns3::WimaxConnection>, std::allocator<ns3::Ptr<ns3::WimaxConnection>>> ns3::ConnectionManager::GetConnections(ns3::Cid::Type type) const [member function]
    cls.add_method('GetConnections', 
                   'std::vector< ns3::Ptr< ns3::WimaxConnection > >', 
                   [param('ns3::Cid::Type', 'type')], 
                   is_const=True)
    ## connection-manager.h (module 'wimax'): uint32_t ns3::ConnectionManager::GetNPackets(ns3::Cid::Type type, ns3::ServiceFlow::SchedulingType schedulingType) const [member function]
    cls.add_method('GetNPackets', 
                   'uint32_t', 
                   [param('ns3::Cid::Type', 'type'), param('ns3::ServiceFlow::SchedulingType', 'schedulingType')], 
                   is_const=True)
    ## connection-manager.h (module 'wimax'): static ns3::TypeId ns3::ConnectionManager::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## connection-manager.h (module 'wimax'): bool ns3::ConnectionManager::HasPackets() const [member function]
    cls.add_method('HasPackets', 
                   'bool', 
                   [], 
                   is_const=True)
    ## connection-manager.h (module 'wimax'): void ns3::ConnectionManager::SetCidFactory(ns3::CidFactory * cidFactory) [member function]
    cls.add_method('SetCidFactory', 
                   'void', 
                   [param('ns3::CidFactory *', 'cidFactory')])
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

def register_Ns3Dcd_methods(root_module, cls):
    ## dl-mac-messages.h (module 'wimax'): ns3::Dcd::Dcd(ns3::Dcd const & arg0) [constructor]
    cls.add_constructor([param('ns3::Dcd const &', 'arg0')])
    ## dl-mac-messages.h (module 'wimax'): ns3::Dcd::Dcd() [constructor]
    cls.add_constructor([])
    ## dl-mac-messages.h (module 'wimax'): void ns3::Dcd::AddDlBurstProfile(ns3::OfdmDlBurstProfile dlBurstProfile) [member function]
    cls.add_method('AddDlBurstProfile', 
                   'void', 
                   [param('ns3::OfdmDlBurstProfile', 'dlBurstProfile')])
    ## dl-mac-messages.h (module 'wimax'): uint32_t ns3::Dcd::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## dl-mac-messages.h (module 'wimax'): ns3::OfdmDcdChannelEncodings ns3::Dcd::GetChannelEncodings() const [member function]
    cls.add_method('GetChannelEncodings', 
                   'ns3::OfdmDcdChannelEncodings', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::Dcd::GetConfigurationChangeCount() const [member function]
    cls.add_method('GetConfigurationChangeCount', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): std::vector<ns3::OfdmDlBurstProfile, std::allocator<ns3::OfdmDlBurstProfile>> ns3::Dcd::GetDlBurstProfiles() const [member function]
    cls.add_method('GetDlBurstProfiles', 
                   'std::vector< ns3::OfdmDlBurstProfile >', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): ns3::TypeId ns3::Dcd::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## dl-mac-messages.h (module 'wimax'): std::string ns3::Dcd::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::Dcd::GetNrDlBurstProfiles() const [member function]
    cls.add_method('GetNrDlBurstProfiles', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint32_t ns3::Dcd::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## dl-mac-messages.h (module 'wimax'): static ns3::TypeId ns3::Dcd::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## dl-mac-messages.h (module 'wimax'): void ns3::Dcd::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## dl-mac-messages.h (module 'wimax'): void ns3::Dcd::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## dl-mac-messages.h (module 'wimax'): void ns3::Dcd::SetChannelEncodings(ns3::OfdmDcdChannelEncodings channelEncodings) [member function]
    cls.add_method('SetChannelEncodings', 
                   'void', 
                   [param('ns3::OfdmDcdChannelEncodings', 'channelEncodings')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::Dcd::SetConfigurationChangeCount(uint8_t configurationChangeCount) [member function]
    cls.add_method('SetConfigurationChangeCount', 
                   'void', 
                   [param('uint8_t', 'configurationChangeCount')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::Dcd::SetNrDlBurstProfiles(uint8_t nrDlBurstProfiles) [member function]
    cls.add_method('SetNrDlBurstProfiles', 
                   'void', 
                   [param('uint8_t', 'nrDlBurstProfiles')])
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

def register_Ns3DlMap_methods(root_module, cls):
    ## dl-mac-messages.h (module 'wimax'): ns3::DlMap::DlMap(ns3::DlMap const & arg0) [constructor]
    cls.add_constructor([param('ns3::DlMap const &', 'arg0')])
    ## dl-mac-messages.h (module 'wimax'): ns3::DlMap::DlMap() [constructor]
    cls.add_constructor([])
    ## dl-mac-messages.h (module 'wimax'): void ns3::DlMap::AddDlMapElement(ns3::OfdmDlMapIe dlMapElement) [member function]
    cls.add_method('AddDlMapElement', 
                   'void', 
                   [param('ns3::OfdmDlMapIe', 'dlMapElement')])
    ## dl-mac-messages.h (module 'wimax'): uint32_t ns3::DlMap::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## dl-mac-messages.h (module 'wimax'): ns3::Mac48Address ns3::DlMap::GetBaseStationId() const [member function]
    cls.add_method('GetBaseStationId', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint8_t ns3::DlMap::GetDcdCount() const [member function]
    cls.add_method('GetDcdCount', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): std::list<ns3::OfdmDlMapIe, std::allocator<ns3::OfdmDlMapIe>> ns3::DlMap::GetDlMapElements() const [member function]
    cls.add_method('GetDlMapElements', 
                   'std::list< ns3::OfdmDlMapIe >', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): ns3::TypeId ns3::DlMap::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## dl-mac-messages.h (module 'wimax'): std::string ns3::DlMap::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## dl-mac-messages.h (module 'wimax'): uint32_t ns3::DlMap::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## dl-mac-messages.h (module 'wimax'): static ns3::TypeId ns3::DlMap::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## dl-mac-messages.h (module 'wimax'): void ns3::DlMap::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## dl-mac-messages.h (module 'wimax'): void ns3::DlMap::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## dl-mac-messages.h (module 'wimax'): void ns3::DlMap::SetBaseStationId(ns3::Mac48Address baseStationID) [member function]
    cls.add_method('SetBaseStationId', 
                   'void', 
                   [param('ns3::Mac48Address', 'baseStationID')])
    ## dl-mac-messages.h (module 'wimax'): void ns3::DlMap::SetDcdCount(uint8_t dcdCount) [member function]
    cls.add_method('SetDcdCount', 
                   'void', 
                   [param('uint8_t', 'dcdCount')])
    return

def register_Ns3DsaAck_methods(root_module, cls):
    ## mac-messages.h (module 'wimax'): ns3::DsaAck::DsaAck(ns3::DsaAck const & arg0) [constructor]
    cls.add_constructor([param('ns3::DsaAck const &', 'arg0')])
    ## mac-messages.h (module 'wimax'): ns3::DsaAck::DsaAck() [constructor]
    cls.add_constructor([])
    ## mac-messages.h (module 'wimax'): uint32_t ns3::DsaAck::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mac-messages.h (module 'wimax'): uint16_t ns3::DsaAck::GetConfirmationCode() const [member function]
    cls.add_method('GetConfirmationCode', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): ns3::TypeId ns3::DsaAck::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): std::string ns3::DsaAck::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::DsaAck::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): uint16_t ns3::DsaAck::GetTransactionId() const [member function]
    cls.add_method('GetTransactionId', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): static ns3::TypeId ns3::DsaAck::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mac-messages.h (module 'wimax'): void ns3::DsaAck::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::DsaAck::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::DsaAck::SetConfirmationCode(uint16_t confirmationCode) [member function]
    cls.add_method('SetConfirmationCode', 
                   'void', 
                   [param('uint16_t', 'confirmationCode')])
    ## mac-messages.h (module 'wimax'): void ns3::DsaAck::SetTransactionId(uint16_t transactionId) [member function]
    cls.add_method('SetTransactionId', 
                   'void', 
                   [param('uint16_t', 'transactionId')])
    return

def register_Ns3DsaReq_methods(root_module, cls):
    ## mac-messages.h (module 'wimax'): ns3::DsaReq::DsaReq(ns3::DsaReq const & arg0) [constructor]
    cls.add_constructor([param('ns3::DsaReq const &', 'arg0')])
    ## mac-messages.h (module 'wimax'): ns3::DsaReq::DsaReq() [constructor]
    cls.add_constructor([])
    ## mac-messages.h (module 'wimax'): ns3::DsaReq::DsaReq(ns3::ServiceFlow sf) [constructor]
    cls.add_constructor([param('ns3::ServiceFlow', 'sf')])
    ## mac-messages.h (module 'wimax'): uint32_t ns3::DsaReq::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mac-messages.h (module 'wimax'): ns3::Cid ns3::DsaReq::GetCid() const [member function]
    cls.add_method('GetCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): ns3::TypeId ns3::DsaReq::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): std::string ns3::DsaReq::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::DsaReq::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): ns3::ServiceFlow ns3::DsaReq::GetServiceFlow() const [member function]
    cls.add_method('GetServiceFlow', 
                   'ns3::ServiceFlow', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::DsaReq::GetSfid() const [member function]
    cls.add_method('GetSfid', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint16_t ns3::DsaReq::GetTransactionId() const [member function]
    cls.add_method('GetTransactionId', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): static ns3::TypeId ns3::DsaReq::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mac-messages.h (module 'wimax'): void ns3::DsaReq::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::DsaReq::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::DsaReq::SetCid(ns3::Cid cid) [member function]
    cls.add_method('SetCid', 
                   'void', 
                   [param('ns3::Cid', 'cid')])
    ## mac-messages.h (module 'wimax'): void ns3::DsaReq::SetServiceFlow(ns3::ServiceFlow sf) [member function]
    cls.add_method('SetServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow', 'sf')])
    ## mac-messages.h (module 'wimax'): void ns3::DsaReq::SetSfid(uint32_t sfid) [member function]
    cls.add_method('SetSfid', 
                   'void', 
                   [param('uint32_t', 'sfid')])
    ## mac-messages.h (module 'wimax'): void ns3::DsaReq::SetTransactionId(uint16_t transactionId) [member function]
    cls.add_method('SetTransactionId', 
                   'void', 
                   [param('uint16_t', 'transactionId')])
    return

def register_Ns3DsaRsp_methods(root_module, cls):
    ## mac-messages.h (module 'wimax'): ns3::DsaRsp::DsaRsp(ns3::DsaRsp const & arg0) [constructor]
    cls.add_constructor([param('ns3::DsaRsp const &', 'arg0')])
    ## mac-messages.h (module 'wimax'): ns3::DsaRsp::DsaRsp() [constructor]
    cls.add_constructor([])
    ## mac-messages.h (module 'wimax'): uint32_t ns3::DsaRsp::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## mac-messages.h (module 'wimax'): ns3::Cid ns3::DsaRsp::GetCid() const [member function]
    cls.add_method('GetCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint16_t ns3::DsaRsp::GetConfirmationCode() const [member function]
    cls.add_method('GetConfirmationCode', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): ns3::TypeId ns3::DsaRsp::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): std::string ns3::DsaRsp::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::DsaRsp::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): ns3::ServiceFlow ns3::DsaRsp::GetServiceFlow() const [member function]
    cls.add_method('GetServiceFlow', 
                   'ns3::ServiceFlow', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint32_t ns3::DsaRsp::GetSfid() const [member function]
    cls.add_method('GetSfid', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): uint16_t ns3::DsaRsp::GetTransactionId() const [member function]
    cls.add_method('GetTransactionId', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## mac-messages.h (module 'wimax'): static ns3::TypeId ns3::DsaRsp::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## mac-messages.h (module 'wimax'): void ns3::DsaRsp::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::DsaRsp::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## mac-messages.h (module 'wimax'): void ns3::DsaRsp::SetCid(ns3::Cid cid) [member function]
    cls.add_method('SetCid', 
                   'void', 
                   [param('ns3::Cid', 'cid')])
    ## mac-messages.h (module 'wimax'): void ns3::DsaRsp::SetConfirmationCode(uint16_t confirmationCode) [member function]
    cls.add_method('SetConfirmationCode', 
                   'void', 
                   [param('uint16_t', 'confirmationCode')])
    ## mac-messages.h (module 'wimax'): void ns3::DsaRsp::SetServiceFlow(ns3::ServiceFlow sf) [member function]
    cls.add_method('SetServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow', 'sf')])
    ## mac-messages.h (module 'wimax'): void ns3::DsaRsp::SetSfid(uint32_t sfid) [member function]
    cls.add_method('SetSfid', 
                   'void', 
                   [param('uint32_t', 'sfid')])
    ## mac-messages.h (module 'wimax'): void ns3::DsaRsp::SetTransactionId(uint16_t transactionId) [member function]
    cls.add_method('SetTransactionId', 
                   'void', 
                   [param('uint16_t', 'transactionId')])
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

def register_Ns3FixedRssLossModel_methods(root_module, cls):
    ## propagation-loss-model.h (module 'propagation'): static ns3::TypeId ns3::FixedRssLossModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## propagation-loss-model.h (module 'propagation'): ns3::FixedRssLossModel::FixedRssLossModel() [constructor]
    cls.add_constructor([])
    ## propagation-loss-model.h (module 'propagation'): void ns3::FixedRssLossModel::SetRss(double rss) [member function]
    cls.add_method('SetRss', 
                   'void', 
                   [param('double', 'rss')])
    ## propagation-loss-model.h (module 'propagation'): double ns3::FixedRssLossModel::DoCalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('DoCalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::FixedRssLossModel::DoAssignStreams(int64_t stream) [member function]
    cls.add_method('DoAssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3FragmentationSubheader_methods(root_module, cls):
    ## wimax-mac-header.h (module 'wimax'): ns3::FragmentationSubheader::FragmentationSubheader(ns3::FragmentationSubheader const & arg0) [constructor]
    cls.add_constructor([param('ns3::FragmentationSubheader const &', 'arg0')])
    ## wimax-mac-header.h (module 'wimax'): ns3::FragmentationSubheader::FragmentationSubheader() [constructor]
    cls.add_constructor([])
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::FragmentationSubheader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::FragmentationSubheader::GetFc() const [member function]
    cls.add_method('GetFc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::FragmentationSubheader::GetFsn() const [member function]
    cls.add_method('GetFsn', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): ns3::TypeId ns3::FragmentationSubheader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): std::string ns3::FragmentationSubheader::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::FragmentationSubheader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): static ns3::TypeId ns3::FragmentationSubheader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::FragmentationSubheader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::FragmentationSubheader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::FragmentationSubheader::SetFc(uint8_t fc) [member function]
    cls.add_method('SetFc', 
                   'void', 
                   [param('uint8_t', 'fc')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::FragmentationSubheader::SetFsn(uint8_t fsn) [member function]
    cls.add_method('SetFsn', 
                   'void', 
                   [param('uint8_t', 'fsn')])
    return

def register_Ns3FriisPropagationLossModel_methods(root_module, cls):
    ## propagation-loss-model.h (module 'propagation'): static ns3::TypeId ns3::FriisPropagationLossModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## propagation-loss-model.h (module 'propagation'): ns3::FriisPropagationLossModel::FriisPropagationLossModel() [constructor]
    cls.add_constructor([])
    ## propagation-loss-model.h (module 'propagation'): void ns3::FriisPropagationLossModel::SetFrequency(double frequency) [member function]
    cls.add_method('SetFrequency', 
                   'void', 
                   [param('double', 'frequency')])
    ## propagation-loss-model.h (module 'propagation'): void ns3::FriisPropagationLossModel::SetSystemLoss(double systemLoss) [member function]
    cls.add_method('SetSystemLoss', 
                   'void', 
                   [param('double', 'systemLoss')])
    ## propagation-loss-model.h (module 'propagation'): void ns3::FriisPropagationLossModel::SetMinLoss(double minLoss) [member function]
    cls.add_method('SetMinLoss', 
                   'void', 
                   [param('double', 'minLoss')])
    ## propagation-loss-model.h (module 'propagation'): double ns3::FriisPropagationLossModel::GetMinLoss() const [member function]
    cls.add_method('GetMinLoss', 
                   'double', 
                   [], 
                   is_const=True)
    ## propagation-loss-model.h (module 'propagation'): double ns3::FriisPropagationLossModel::GetFrequency() const [member function]
    cls.add_method('GetFrequency', 
                   'double', 
                   [], 
                   is_const=True)
    ## propagation-loss-model.h (module 'propagation'): double ns3::FriisPropagationLossModel::GetSystemLoss() const [member function]
    cls.add_method('GetSystemLoss', 
                   'double', 
                   [], 
                   is_const=True)
    ## propagation-loss-model.h (module 'propagation'): double ns3::FriisPropagationLossModel::DoCalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('DoCalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::FriisPropagationLossModel::DoAssignStreams(int64_t stream) [member function]
    cls.add_method('DoAssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True, visibility='private')
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

def register_Ns3GenericMacHeader_methods(root_module, cls):
    ## wimax-mac-header.h (module 'wimax'): ns3::GenericMacHeader::GenericMacHeader(ns3::GenericMacHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::GenericMacHeader const &', 'arg0')])
    ## wimax-mac-header.h (module 'wimax'): ns3::GenericMacHeader::GenericMacHeader() [constructor]
    cls.add_constructor([])
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::GenericMacHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::GenericMacHeader::GetCi() const [member function]
    cls.add_method('GetCi', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): ns3::Cid ns3::GenericMacHeader::GetCid() const [member function]
    cls.add_method('GetCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::GenericMacHeader::GetEc() const [member function]
    cls.add_method('GetEc', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::GenericMacHeader::GetEks() const [member function]
    cls.add_method('GetEks', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::GenericMacHeader::GetHcs() const [member function]
    cls.add_method('GetHcs', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::GenericMacHeader::GetHt() const [member function]
    cls.add_method('GetHt', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): ns3::TypeId ns3::GenericMacHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): uint16_t ns3::GenericMacHeader::GetLen() const [member function]
    cls.add_method('GetLen', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): std::string ns3::GenericMacHeader::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::GenericMacHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::GenericMacHeader::GetType() const [member function]
    cls.add_method('GetType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): static ns3::TypeId ns3::GenericMacHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::GenericMacHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::GenericMacHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::GenericMacHeader::SetCi(uint8_t ci) [member function]
    cls.add_method('SetCi', 
                   'void', 
                   [param('uint8_t', 'ci')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::GenericMacHeader::SetCid(ns3::Cid cid) [member function]
    cls.add_method('SetCid', 
                   'void', 
                   [param('ns3::Cid', 'cid')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::GenericMacHeader::SetEc(uint8_t ec) [member function]
    cls.add_method('SetEc', 
                   'void', 
                   [param('uint8_t', 'ec')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::GenericMacHeader::SetEks(uint8_t eks) [member function]
    cls.add_method('SetEks', 
                   'void', 
                   [param('uint8_t', 'eks')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::GenericMacHeader::SetHcs(uint8_t hcs) [member function]
    cls.add_method('SetHcs', 
                   'void', 
                   [param('uint8_t', 'hcs')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::GenericMacHeader::SetHt(uint8_t ht) [member function]
    cls.add_method('SetHt', 
                   'void', 
                   [param('uint8_t', 'ht')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::GenericMacHeader::SetLen(uint16_t len) [member function]
    cls.add_method('SetLen', 
                   'void', 
                   [param('uint16_t', 'len')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::GenericMacHeader::SetType(uint8_t type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('uint8_t', 'type')])
    ## wimax-mac-header.h (module 'wimax'): bool ns3::GenericMacHeader::check_hcs() const [member function]
    cls.add_method('check_hcs', 
                   'bool', 
                   [], 
                   is_const=True)
    return

def register_Ns3GrantManagementSubheader_methods(root_module, cls):
    ## wimax-mac-header.h (module 'wimax'): ns3::GrantManagementSubheader::GrantManagementSubheader(ns3::GrantManagementSubheader const & arg0) [constructor]
    cls.add_constructor([param('ns3::GrantManagementSubheader const &', 'arg0')])
    ## wimax-mac-header.h (module 'wimax'): ns3::GrantManagementSubheader::GrantManagementSubheader() [constructor]
    cls.add_constructor([])
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::GrantManagementSubheader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): ns3::TypeId ns3::GrantManagementSubheader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): std::string ns3::GrantManagementSubheader::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint16_t ns3::GrantManagementSubheader::GetPbr() const [member function]
    cls.add_method('GetPbr', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::GrantManagementSubheader::GetPm() const [member function]
    cls.add_method('GetPm', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): uint32_t ns3::GrantManagementSubheader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): uint8_t ns3::GrantManagementSubheader::GetSi() const [member function]
    cls.add_method('GetSi', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-mac-header.h (module 'wimax'): static ns3::TypeId ns3::GrantManagementSubheader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::GrantManagementSubheader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::GrantManagementSubheader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## wimax-mac-header.h (module 'wimax'): void ns3::GrantManagementSubheader::SetPbr(uint16_t pbr) [member function]
    cls.add_method('SetPbr', 
                   'void', 
                   [param('uint16_t', 'pbr')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::GrantManagementSubheader::SetPm(uint8_t pm) [member function]
    cls.add_method('SetPm', 
                   'void', 
                   [param('uint8_t', 'pm')])
    ## wimax-mac-header.h (module 'wimax'): void ns3::GrantManagementSubheader::SetSi(uint8_t si) [member function]
    cls.add_method('SetSi', 
                   'void', 
                   [param('uint8_t', 'si')])
    return

def register_Ns3IpcsClassifier_methods(root_module, cls):
    ## ipcs-classifier.h (module 'wimax'): ns3::IpcsClassifier::IpcsClassifier(ns3::IpcsClassifier const & arg0) [constructor]
    cls.add_constructor([param('ns3::IpcsClassifier const &', 'arg0')])
    ## ipcs-classifier.h (module 'wimax'): ns3::IpcsClassifier::IpcsClassifier() [constructor]
    cls.add_constructor([])
    ## ipcs-classifier.h (module 'wimax'): ns3::ServiceFlow * ns3::IpcsClassifier::Classify(ns3::Ptr<const ns3::Packet> packet, ns3::Ptr<ns3::ServiceFlowManager> sfm, ns3::ServiceFlow::Direction dir) [member function]
    cls.add_method('Classify', 
                   'ns3::ServiceFlow *', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet'), param('ns3::Ptr< ns3::ServiceFlowManager >', 'sfm'), param('ns3::ServiceFlow::Direction', 'dir')])
    ## ipcs-classifier.h (module 'wimax'): static ns3::TypeId ns3::IpcsClassifier::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
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

def register_Ns3LogDistancePropagationLossModel_methods(root_module, cls):
    ## propagation-loss-model.h (module 'propagation'): static ns3::TypeId ns3::LogDistancePropagationLossModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## propagation-loss-model.h (module 'propagation'): ns3::LogDistancePropagationLossModel::LogDistancePropagationLossModel() [constructor]
    cls.add_constructor([])
    ## propagation-loss-model.h (module 'propagation'): void ns3::LogDistancePropagationLossModel::SetPathLossExponent(double n) [member function]
    cls.add_method('SetPathLossExponent', 
                   'void', 
                   [param('double', 'n')])
    ## propagation-loss-model.h (module 'propagation'): double ns3::LogDistancePropagationLossModel::GetPathLossExponent() const [member function]
    cls.add_method('GetPathLossExponent', 
                   'double', 
                   [], 
                   is_const=True)
    ## propagation-loss-model.h (module 'propagation'): void ns3::LogDistancePropagationLossModel::SetReference(double referenceDistance, double referenceLoss) [member function]
    cls.add_method('SetReference', 
                   'void', 
                   [param('double', 'referenceDistance'), param('double', 'referenceLoss')])
    ## propagation-loss-model.h (module 'propagation'): double ns3::LogDistancePropagationLossModel::DoCalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('DoCalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::LogDistancePropagationLossModel::DoAssignStreams(int64_t stream) [member function]
    cls.add_method('DoAssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True, visibility='private')
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

def register_Ns3MatrixPropagationLossModel_methods(root_module, cls):
    ## propagation-loss-model.h (module 'propagation'): static ns3::TypeId ns3::MatrixPropagationLossModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## propagation-loss-model.h (module 'propagation'): ns3::MatrixPropagationLossModel::MatrixPropagationLossModel() [constructor]
    cls.add_constructor([])
    ## propagation-loss-model.h (module 'propagation'): void ns3::MatrixPropagationLossModel::SetLoss(ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b, double loss, bool symmetric=true) [member function]
    cls.add_method('SetLoss', 
                   'void', 
                   [param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b'), param('double', 'loss'), param('bool', 'symmetric', default_value='true')])
    ## propagation-loss-model.h (module 'propagation'): void ns3::MatrixPropagationLossModel::SetDefaultLoss(double defaultLoss) [member function]
    cls.add_method('SetDefaultLoss', 
                   'void', 
                   [param('double', 'defaultLoss')])
    ## propagation-loss-model.h (module 'propagation'): double ns3::MatrixPropagationLossModel::DoCalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('DoCalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::MatrixPropagationLossModel::DoAssignStreams(int64_t stream) [member function]
    cls.add_method('DoAssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3NakagamiPropagationLossModel_methods(root_module, cls):
    ## propagation-loss-model.h (module 'propagation'): static ns3::TypeId ns3::NakagamiPropagationLossModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## propagation-loss-model.h (module 'propagation'): ns3::NakagamiPropagationLossModel::NakagamiPropagationLossModel() [constructor]
    cls.add_constructor([])
    ## propagation-loss-model.h (module 'propagation'): double ns3::NakagamiPropagationLossModel::DoCalcRxPower(double txPowerDbm, ns3::Ptr<ns3::MobilityModel> a, ns3::Ptr<ns3::MobilityModel> b) const [member function]
    cls.add_method('DoCalcRxPower', 
                   'double', 
                   [param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::MobilityModel >', 'a'), param('ns3::Ptr< ns3::MobilityModel >', 'b')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## propagation-loss-model.h (module 'propagation'): int64_t ns3::NakagamiPropagationLossModel::DoAssignStreams(int64_t stream) [member function]
    cls.add_method('DoAssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True, visibility='private')
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

def register_Ns3OutputStreamWrapper_methods(root_module, cls):
    ## output-stream-wrapper.h (module 'network'): ns3::OutputStreamWrapper::OutputStreamWrapper(ns3::OutputStreamWrapper const & arg0) [constructor]
    cls.add_constructor([param('ns3::OutputStreamWrapper const &', 'arg0')])
    ## output-stream-wrapper.h (module 'network'): ns3::OutputStreamWrapper::OutputStreamWrapper(std::string filename, std::ios_base::openmode filemode) [constructor]
    cls.add_constructor([param('std::string', 'filename'), param('std::ios_base::openmode', 'filemode')])
    ## output-stream-wrapper.h (module 'network'): ns3::OutputStreamWrapper::OutputStreamWrapper(std::ostream * os) [constructor]
    cls.add_constructor([param('std::ostream *', 'os')])
    ## output-stream-wrapper.h (module 'network'): std::ostream * ns3::OutputStreamWrapper::GetStream() [member function]
    cls.add_method('GetStream', 
                   'std::ostream *', 
                   [])
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

def register_Ns3SimpleOfdmWimaxPhy_methods(root_module, cls):
    ## simple-ofdm-wimax-phy.h (module 'wimax'): ns3::SimpleOfdmWimaxPhy::SimpleOfdmWimaxPhy(ns3::SimpleOfdmWimaxPhy const & arg0) [constructor]
    cls.add_constructor([param('ns3::SimpleOfdmWimaxPhy const &', 'arg0')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): ns3::SimpleOfdmWimaxPhy::SimpleOfdmWimaxPhy() [constructor]
    cls.add_constructor([])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): ns3::SimpleOfdmWimaxPhy::SimpleOfdmWimaxPhy(char * tracesPath) [constructor]
    cls.add_constructor([param('char *', 'tracesPath')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::ActivateLoss(bool loss) [member function]
    cls.add_method('ActivateLoss', 
                   'void', 
                   [param('bool', 'loss')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): int64_t ns3::SimpleOfdmWimaxPhy::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True)
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::DoAttach(ns3::Ptr<ns3::WimaxChannel> channel) [member function]
    cls.add_method('DoAttach', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxChannel >', 'channel')], 
                   is_virtual=True)
    ## simple-ofdm-wimax-phy.h (module 'wimax'): uint32_t ns3::SimpleOfdmWimaxPhy::GetBandwidth() const [member function]
    cls.add_method('GetBandwidth', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## simple-ofdm-wimax-phy.h (module 'wimax'): double ns3::SimpleOfdmWimaxPhy::GetNoiseFigure() const [member function]
    cls.add_method('GetNoiseFigure', 
                   'double', 
                   [], 
                   is_const=True)
    ## simple-ofdm-wimax-phy.h (module 'wimax'): ns3::WimaxPhy::PhyType ns3::SimpleOfdmWimaxPhy::GetPhyType() const [member function]
    cls.add_method('GetPhyType', 
                   'ns3::WimaxPhy::PhyType', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-ofdm-wimax-phy.h (module 'wimax'): double ns3::SimpleOfdmWimaxPhy::GetTxPower() const [member function]
    cls.add_method('GetTxPower', 
                   'double', 
                   [], 
                   is_const=True)
    ## simple-ofdm-wimax-phy.h (module 'wimax'): static ns3::TypeId ns3::SimpleOfdmWimaxPhy::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::NotifyRxBegin(ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('NotifyRxBegin', 
                   'void', 
                   [param('ns3::Ptr< ns3::PacketBurst >', 'burst')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::NotifyRxDrop(ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('NotifyRxDrop', 
                   'void', 
                   [param('ns3::Ptr< ns3::PacketBurst >', 'burst')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::NotifyRxEnd(ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('NotifyRxEnd', 
                   'void', 
                   [param('ns3::Ptr< ns3::PacketBurst >', 'burst')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::NotifyTxBegin(ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('NotifyTxBegin', 
                   'void', 
                   [param('ns3::Ptr< ns3::PacketBurst >', 'burst')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::NotifyTxDrop(ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('NotifyTxDrop', 
                   'void', 
                   [param('ns3::Ptr< ns3::PacketBurst >', 'burst')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::NotifyTxEnd(ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('NotifyTxEnd', 
                   'void', 
                   [param('ns3::Ptr< ns3::PacketBurst >', 'burst')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::Send(ns3::Ptr<ns3::PacketBurst> burst, ns3::WimaxPhy::ModulationType modulationType, uint8_t direction) [member function]
    cls.add_method('Send', 
                   'void', 
                   [param('ns3::Ptr< ns3::PacketBurst >', 'burst'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('uint8_t', 'direction')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::Send(ns3::SendParams * params) [member function]
    cls.add_method('Send', 
                   'void', 
                   [param('ns3::SendParams *', 'params')], 
                   is_virtual=True)
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::SetBandwidth(uint32_t BW) [member function]
    cls.add_method('SetBandwidth', 
                   'void', 
                   [param('uint32_t', 'BW')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::SetNoiseFigure(double nf) [member function]
    cls.add_method('SetNoiseFigure', 
                   'void', 
                   [param('double', 'nf')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::SetReceiveCallback(ns3::Callback<void, ns3::Ptr<ns3::PacketBurst>, ns3::Ptr<ns3::WimaxConnection>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> callback) [member function]
    cls.add_method('SetReceiveCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::PacketBurst >, ns3::Ptr< ns3::WimaxConnection >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'callback')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::SetSNRToBlockErrorRateTracesPath(char * tracesPath) [member function]
    cls.add_method('SetSNRToBlockErrorRateTracesPath', 
                   'void', 
                   [param('char *', 'tracesPath')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::SetTxPower(double txPower) [member function]
    cls.add_method('SetTxPower', 
                   'void', 
                   [param('double', 'txPower')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::StartReceive(uint32_t burstSize, bool isFirstBlock, uint64_t frequency, ns3::WimaxPhy::ModulationType modulationType, uint8_t direction, double rxPower, ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('StartReceive', 
                   'void', 
                   [param('uint32_t', 'burstSize'), param('bool', 'isFirstBlock'), param('uint64_t', 'frequency'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('uint8_t', 'direction'), param('double', 'rxPower'), param('ns3::Ptr< ns3::PacketBurst >', 'burst')])
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): uint32_t ns3::SimpleOfdmWimaxPhy::DoGetDataRate(ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('DoGetDataRate', 
                   'uint32_t', 
                   [param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): ns3::Time ns3::SimpleOfdmWimaxPhy::DoGetFrameDuration(uint8_t frameDurationCode) const [member function]
    cls.add_method('DoGetFrameDuration', 
                   'ns3::Time', 
                   [param('uint8_t', 'frameDurationCode')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): uint8_t ns3::SimpleOfdmWimaxPhy::DoGetFrameDurationCode() const [member function]
    cls.add_method('DoGetFrameDurationCode', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): double ns3::SimpleOfdmWimaxPhy::DoGetGValue() const [member function]
    cls.add_method('DoGetGValue', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): uint16_t ns3::SimpleOfdmWimaxPhy::DoGetNfft() const [member function]
    cls.add_method('DoGetNfft', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): uint64_t ns3::SimpleOfdmWimaxPhy::DoGetNrBytes(uint32_t symbols, ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('DoGetNrBytes', 
                   'uint64_t', 
                   [param('uint32_t', 'symbols'), param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): uint64_t ns3::SimpleOfdmWimaxPhy::DoGetNrSymbols(uint32_t size, ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('DoGetNrSymbols', 
                   'uint64_t', 
                   [param('uint32_t', 'size'), param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): uint16_t ns3::SimpleOfdmWimaxPhy::DoGetRtg() const [member function]
    cls.add_method('DoGetRtg', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): double ns3::SimpleOfdmWimaxPhy::DoGetSamplingFactor() const [member function]
    cls.add_method('DoGetSamplingFactor', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): double ns3::SimpleOfdmWimaxPhy::DoGetSamplingFrequency() const [member function]
    cls.add_method('DoGetSamplingFrequency', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): ns3::Time ns3::SimpleOfdmWimaxPhy::DoGetTransmissionTime(uint32_t size, ns3::WimaxPhy::ModulationType modulationType) const [member function]
    cls.add_method('DoGetTransmissionTime', 
                   'ns3::Time', 
                   [param('uint32_t', 'size'), param('ns3::WimaxPhy::ModulationType', 'modulationType')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): uint16_t ns3::SimpleOfdmWimaxPhy::DoGetTtg() const [member function]
    cls.add_method('DoGetTtg', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::DoSetDataRates() [member function]
    cls.add_method('DoSetDataRates', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-phy.h (module 'wimax'): void ns3::SimpleOfdmWimaxPhy::DoSetPhyParameters() [member function]
    cls.add_method('DoSetPhyParameters', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
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

def register_Ns3WimaxChannel_methods(root_module, cls):
    ## wimax-channel.h (module 'wimax'): ns3::WimaxChannel::WimaxChannel(ns3::WimaxChannel const & arg0) [constructor]
    cls.add_constructor([param('ns3::WimaxChannel const &', 'arg0')])
    ## wimax-channel.h (module 'wimax'): ns3::WimaxChannel::WimaxChannel() [constructor]
    cls.add_constructor([])
    ## wimax-channel.h (module 'wimax'): int64_t ns3::WimaxChannel::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_pure_virtual=True, is_virtual=True)
    ## wimax-channel.h (module 'wimax'): void ns3::WimaxChannel::Attach(ns3::Ptr<ns3::WimaxPhy> phy) [member function]
    cls.add_method('Attach', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxPhy >', 'phy')])
    ## wimax-channel.h (module 'wimax'): ns3::Ptr<ns3::NetDevice> ns3::WimaxChannel::GetDevice(std::size_t i) const [member function]
    cls.add_method('GetDevice', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [param('std::size_t', 'i')], 
                   is_const=True, is_virtual=True)
    ## wimax-channel.h (module 'wimax'): std::size_t ns3::WimaxChannel::GetNDevices() const [member function]
    cls.add_method('GetNDevices', 
                   'std::size_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-channel.h (module 'wimax'): static ns3::TypeId ns3::WimaxChannel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-channel.h (module 'wimax'): void ns3::WimaxChannel::DoAttach(ns3::Ptr<ns3::WimaxPhy> phy) [member function]
    cls.add_method('DoAttach', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxPhy >', 'phy')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-channel.h (module 'wimax'): ns3::Ptr<ns3::NetDevice> ns3::WimaxChannel::DoGetDevice(std::size_t i) const [member function]
    cls.add_method('DoGetDevice', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [param('std::size_t', 'i')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-channel.h (module 'wimax'): std::size_t ns3::WimaxChannel::DoGetNDevices() const [member function]
    cls.add_method('DoGetNDevices', 
                   'std::size_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='private')
    return

def register_Ns3WimaxNetDevice_methods(root_module, cls):
    ## wimax-net-device.h (module 'wimax'): static ns3::TypeId ns3::WimaxNetDevice::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## wimax-net-device.h (module 'wimax'): ns3::WimaxNetDevice::WimaxNetDevice() [constructor]
    cls.add_constructor([])
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetTtg(uint16_t ttg) [member function]
    cls.add_method('SetTtg', 
                   'void', 
                   [param('uint16_t', 'ttg')])
    ## wimax-net-device.h (module 'wimax'): uint16_t ns3::WimaxNetDevice::GetTtg() const [member function]
    cls.add_method('GetTtg', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetRtg(uint16_t rtg) [member function]
    cls.add_method('SetRtg', 
                   'void', 
                   [param('uint16_t', 'rtg')])
    ## wimax-net-device.h (module 'wimax'): uint16_t ns3::WimaxNetDevice::GetRtg() const [member function]
    cls.add_method('GetRtg', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::Attach(ns3::Ptr<ns3::WimaxChannel> channel) [member function]
    cls.add_method('Attach', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxChannel >', 'channel')])
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetPhy(ns3::Ptr<ns3::WimaxPhy> phy) [member function]
    cls.add_method('SetPhy', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxPhy >', 'phy')])
    ## wimax-net-device.h (module 'wimax'): ns3::Ptr<ns3::WimaxPhy> ns3::WimaxNetDevice::GetPhy() const [member function]
    cls.add_method('GetPhy', 
                   'ns3::Ptr< ns3::WimaxPhy >', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetChannel(ns3::Ptr<ns3::WimaxChannel> wimaxChannel) [member function]
    cls.add_method('SetChannel', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxChannel >', 'wimaxChannel')])
    ## wimax-net-device.h (module 'wimax'): uint64_t ns3::WimaxNetDevice::GetChannel(uint8_t index) const [member function]
    cls.add_method('GetChannel', 
                   'uint64_t', 
                   [param('uint8_t', 'index')], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetNrFrames(uint32_t nrFrames) [member function]
    cls.add_method('SetNrFrames', 
                   'void', 
                   [param('uint32_t', 'nrFrames')])
    ## wimax-net-device.h (module 'wimax'): uint32_t ns3::WimaxNetDevice::GetNrFrames() const [member function]
    cls.add_method('GetNrFrames', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetMacAddress(ns3::Mac48Address address) [member function]
    cls.add_method('SetMacAddress', 
                   'void', 
                   [param('ns3::Mac48Address', 'address')])
    ## wimax-net-device.h (module 'wimax'): ns3::Mac48Address ns3::WimaxNetDevice::GetMacAddress() const [member function]
    cls.add_method('GetMacAddress', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetState(uint8_t state) [member function]
    cls.add_method('SetState', 
                   'void', 
                   [param('uint8_t', 'state')])
    ## wimax-net-device.h (module 'wimax'): uint8_t ns3::WimaxNetDevice::GetState() const [member function]
    cls.add_method('GetState', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Ptr<ns3::WimaxConnection> ns3::WimaxNetDevice::GetInitialRangingConnection() const [member function]
    cls.add_method('GetInitialRangingConnection', 
                   'ns3::Ptr< ns3::WimaxConnection >', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Ptr<ns3::WimaxConnection> ns3::WimaxNetDevice::GetBroadcastConnection() const [member function]
    cls.add_method('GetBroadcastConnection', 
                   'ns3::Ptr< ns3::WimaxConnection >', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetCurrentDcd(ns3::Dcd dcd) [member function]
    cls.add_method('SetCurrentDcd', 
                   'void', 
                   [param('ns3::Dcd', 'dcd')])
    ## wimax-net-device.h (module 'wimax'): ns3::Dcd ns3::WimaxNetDevice::GetCurrentDcd() const [member function]
    cls.add_method('GetCurrentDcd', 
                   'ns3::Dcd', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetCurrentUcd(ns3::Ucd ucd) [member function]
    cls.add_method('SetCurrentUcd', 
                   'void', 
                   [param('ns3::Ucd', 'ucd')])
    ## wimax-net-device.h (module 'wimax'): ns3::Ucd ns3::WimaxNetDevice::GetCurrentUcd() const [member function]
    cls.add_method('GetCurrentUcd', 
                   'ns3::Ucd', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Ptr<ns3::ConnectionManager> ns3::WimaxNetDevice::GetConnectionManager() const [member function]
    cls.add_method('GetConnectionManager', 
                   'ns3::Ptr< ns3::ConnectionManager >', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetConnectionManager(ns3::Ptr<ns3::ConnectionManager> connectionManager) [member function]
    cls.add_method('SetConnectionManager', 
                   'void', 
                   [param('ns3::Ptr< ns3::ConnectionManager >', 'connectionManager')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Ptr<ns3::BurstProfileManager> ns3::WimaxNetDevice::GetBurstProfileManager() const [member function]
    cls.add_method('GetBurstProfileManager', 
                   'ns3::Ptr< ns3::BurstProfileManager >', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetBurstProfileManager(ns3::Ptr<ns3::BurstProfileManager> burstProfileManager) [member function]
    cls.add_method('SetBurstProfileManager', 
                   'void', 
                   [param('ns3::Ptr< ns3::BurstProfileManager >', 'burstProfileManager')])
    ## wimax-net-device.h (module 'wimax'): ns3::Ptr<ns3::BandwidthManager> ns3::WimaxNetDevice::GetBandwidthManager() const [member function]
    cls.add_method('GetBandwidthManager', 
                   'ns3::Ptr< ns3::BandwidthManager >', 
                   [], 
                   is_const=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetBandwidthManager(ns3::Ptr<ns3::BandwidthManager> bandwidthManager) [member function]
    cls.add_method('SetBandwidthManager', 
                   'void', 
                   [param('ns3::Ptr< ns3::BandwidthManager >', 'bandwidthManager')])
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::CreateDefaultConnections() [member function]
    cls.add_method('CreateDefaultConnections', 
                   'void', 
                   [])
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::Start() [member function]
    cls.add_method('Start', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::Stop() [member function]
    cls.add_method('Stop', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetReceiveCallback() [member function]
    cls.add_method('SetReceiveCallback', 
                   'void', 
                   [])
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::ForwardUp(ns3::Ptr<ns3::Packet> packet, ns3::Mac48Address const & source, ns3::Mac48Address const & dest) [member function]
    cls.add_method('ForwardUp', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Mac48Address const &', 'source'), param('ns3::Mac48Address const &', 'dest')])
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::Enqueue(ns3::Ptr<ns3::Packet> packet, ns3::MacHeaderType const & hdrType, ns3::Ptr<ns3::WimaxConnection> connection) [member function]
    cls.add_method('Enqueue', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::MacHeaderType const &', 'hdrType'), param('ns3::Ptr< ns3::WimaxConnection >', 'connection')], 
                   is_pure_virtual=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::ForwardDown(ns3::Ptr<ns3::PacketBurst> burst, ns3::WimaxPhy::ModulationType modulationType) [member function]
    cls.add_method('ForwardDown', 
                   'void', 
                   [param('ns3::Ptr< ns3::PacketBurst >', 'burst'), param('ns3::WimaxPhy::ModulationType', 'modulationType')])
    ## wimax-net-device.h (module 'wimax'): ns3::WimaxNetDevice::m_direction [variable]
    cls.add_static_attribute('m_direction', 'uint8_t', is_const=False)
    ## wimax-net-device.h (module 'wimax'): ns3::WimaxNetDevice::m_frameStartTime [variable]
    cls.add_static_attribute('m_frameStartTime', 'ns3::Time', is_const=False)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetName(std::string const name) [member function]
    cls.add_method('SetName', 
                   'void', 
                   [param('std::string const', 'name')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): std::string ns3::WimaxNetDevice::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetIfIndex(uint32_t const index) [member function]
    cls.add_method('SetIfIndex', 
                   'void', 
                   [param('uint32_t const', 'index')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): uint32_t ns3::WimaxNetDevice::GetIfIndex() const [member function]
    cls.add_method('GetIfIndex', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Ptr<ns3::Channel> ns3::WimaxNetDevice::GetPhyChannel() const [member function]
    cls.add_method('GetPhyChannel', 
                   'ns3::Ptr< ns3::Channel >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Ptr<ns3::Channel> ns3::WimaxNetDevice::GetChannel() const [member function]
    cls.add_method('GetChannel', 
                   'ns3::Ptr< ns3::Channel >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetAddress(ns3::Address address) [member function]
    cls.add_method('SetAddress', 
                   'void', 
                   [param('ns3::Address', 'address')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Address ns3::WimaxNetDevice::GetAddress() const [member function]
    cls.add_method('GetAddress', 
                   'ns3::Address', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::SetMtu(uint16_t const mtu) [member function]
    cls.add_method('SetMtu', 
                   'bool', 
                   [param('uint16_t const', 'mtu')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): uint16_t ns3::WimaxNetDevice::GetMtu() const [member function]
    cls.add_method('GetMtu', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::IsLinkUp() const [member function]
    cls.add_method('IsLinkUp', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetLinkChangeCallback(ns3::Callback<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> callback) [member function]
    cls.add_method('SetLinkChangeCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'callback')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::IsBroadcast() const [member function]
    cls.add_method('IsBroadcast', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Address ns3::WimaxNetDevice::GetBroadcast() const [member function]
    cls.add_method('GetBroadcast', 
                   'ns3::Address', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::IsMulticast() const [member function]
    cls.add_method('IsMulticast', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Address ns3::WimaxNetDevice::GetMulticast() const [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Address', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Address ns3::WimaxNetDevice::MakeMulticastAddress(ns3::Ipv4Address multicastGroup) const [member function]
    cls.add_method('MakeMulticastAddress', 
                   'ns3::Address', 
                   [param('ns3::Ipv4Address', 'multicastGroup')], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::IsPointToPoint() const [member function]
    cls.add_method('IsPointToPoint', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::Send(ns3::Ptr<ns3::Packet> packet, ns3::Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('Send', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetNode(ns3::Ptr<ns3::Node> node) [member function]
    cls.add_method('SetNode', 
                   'void', 
                   [param('ns3::Ptr< ns3::Node >', 'node')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Ptr<ns3::Node> ns3::WimaxNetDevice::GetNode() const [member function]
    cls.add_method('GetNode', 
                   'ns3::Ptr< ns3::Node >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::NeedsArp() const [member function]
    cls.add_method('NeedsArp', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetReceiveCallback(ns3::NetDevice::ReceiveCallback cb) [member function]
    cls.add_method('SetReceiveCallback', 
                   'void', 
                   [param('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::AddLinkChangeCallback(ns3::Callback<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> callback) [member function]
    cls.add_method('AddLinkChangeCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'callback')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::SendFrom(ns3::Ptr<ns3::Packet> packet, ns3::Address const & source, ns3::Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('SendFrom', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Address const &', 'source'), param('ns3::Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::SetPromiscReceiveCallback(ns3::NetDevice::PromiscReceiveCallback cb) [member function]
    cls.add_method('SetPromiscReceiveCallback', 
                   'void', 
                   [param('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >', 'cb')], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::NetDevice::PromiscReceiveCallback ns3::WimaxNetDevice::GetPromiscReceiveCallback() [member function]
    cls.add_method('GetPromiscReceiveCallback', 
                   'ns3::NetDevice::PromiscReceiveCallback', 
                   [])
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::SupportsSendFrom() const [member function]
    cls.add_method('SupportsSendFrom', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::WimaxNetDevice::m_traceRx [variable]
    cls.add_instance_attribute('m_traceRx', 'ns3::TracedCallback< ns3::Ptr< ns3::Packet const >, ns3::Mac48Address const & >', is_const=False)
    ## wimax-net-device.h (module 'wimax'): ns3::WimaxNetDevice::m_traceTx [variable]
    cls.add_instance_attribute('m_traceTx', 'ns3::TracedCallback< ns3::Ptr< ns3::Packet const >, ns3::Mac48Address const & >', is_const=False)
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Address ns3::WimaxNetDevice::GetMulticast(ns3::Ipv6Address addr) const [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Address', 
                   [param('ns3::Ipv6Address', 'addr')], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): ns3::Address ns3::WimaxNetDevice::GetMulticast(ns3::Ipv4Address multicastGroup) const [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Address', 
                   [param('ns3::Ipv4Address', 'multicastGroup')], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::IsBridge() const [member function]
    cls.add_method('IsBridge', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::IsPromisc() [member function]
    cls.add_method('IsPromisc', 
                   'bool', 
                   [])
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::NotifyPromiscTrace(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('NotifyPromiscTrace', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')])
    ## wimax-net-device.h (module 'wimax'): bool ns3::WimaxNetDevice::DoSend(ns3::Ptr<ns3::Packet> packet, ns3::Mac48Address const & source, ns3::Mac48Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('DoSend', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Mac48Address const &', 'source'), param('ns3::Mac48Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-net-device.h (module 'wimax'): void ns3::WimaxNetDevice::DoReceive(ns3::Ptr<ns3::Packet> packet) [member function]
    cls.add_method('DoReceive', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## wimax-net-device.h (module 'wimax'): ns3::Ptr<ns3::WimaxChannel> ns3::WimaxNetDevice::DoGetChannel() const [member function]
    cls.add_method('DoGetChannel', 
                   'ns3::Ptr< ns3::WimaxChannel >', 
                   [], 
                   is_const=True, is_virtual=True, visibility='private')
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

def register_Ns3BaseStationNetDevice_methods(root_module, cls):
    ## bs-net-device.h (module 'wimax'): static ns3::TypeId ns3::BaseStationNetDevice::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## bs-net-device.h (module 'wimax'): ns3::BaseStationNetDevice::BaseStationNetDevice() [constructor]
    cls.add_constructor([])
    ## bs-net-device.h (module 'wimax'): ns3::BaseStationNetDevice::BaseStationNetDevice(ns3::Ptr<ns3::Node> node, ns3::Ptr<ns3::WimaxPhy> phy) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::Node >', 'node'), param('ns3::Ptr< ns3::WimaxPhy >', 'phy')])
    ## bs-net-device.h (module 'wimax'): ns3::BaseStationNetDevice::BaseStationNetDevice(ns3::Ptr<ns3::Node> node, ns3::Ptr<ns3::WimaxPhy> phy, ns3::Ptr<ns3::UplinkScheduler> uplinkScheduler, ns3::Ptr<ns3::BSScheduler> bsScheduler) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::Node >', 'node'), param('ns3::Ptr< ns3::WimaxPhy >', 'phy'), param('ns3::Ptr< ns3::UplinkScheduler >', 'uplinkScheduler'), param('ns3::Ptr< ns3::BSScheduler >', 'bsScheduler')])
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetInitialRangingInterval(ns3::Time initialRangInterval) [member function]
    cls.add_method('SetInitialRangingInterval', 
                   'void', 
                   [param('ns3::Time', 'initialRangInterval')])
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::InitBaseStationNetDevice() [member function]
    cls.add_method('InitBaseStationNetDevice', 
                   'void', 
                   [])
    ## bs-net-device.h (module 'wimax'): ns3::Time ns3::BaseStationNetDevice::GetInitialRangingInterval() const [member function]
    cls.add_method('GetInitialRangingInterval', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetDcdInterval(ns3::Time dcdInterval) [member function]
    cls.add_method('SetDcdInterval', 
                   'void', 
                   [param('ns3::Time', 'dcdInterval')])
    ## bs-net-device.h (module 'wimax'): ns3::Time ns3::BaseStationNetDevice::GetDcdInterval() const [member function]
    cls.add_method('GetDcdInterval', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetUcdInterval(ns3::Time ucdInterval) [member function]
    cls.add_method('SetUcdInterval', 
                   'void', 
                   [param('ns3::Time', 'ucdInterval')])
    ## bs-net-device.h (module 'wimax'): ns3::Time ns3::BaseStationNetDevice::GetUcdInterval() const [member function]
    cls.add_method('GetUcdInterval', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetIntervalT8(ns3::Time interval) [member function]
    cls.add_method('SetIntervalT8', 
                   'void', 
                   [param('ns3::Time', 'interval')])
    ## bs-net-device.h (module 'wimax'): ns3::Time ns3::BaseStationNetDevice::GetIntervalT8() const [member function]
    cls.add_method('GetIntervalT8', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetMaxRangingCorrectionRetries(uint8_t maxRangCorrectionRetries) [member function]
    cls.add_method('SetMaxRangingCorrectionRetries', 
                   'void', 
                   [param('uint8_t', 'maxRangCorrectionRetries')])
    ## bs-net-device.h (module 'wimax'): uint8_t ns3::BaseStationNetDevice::GetMaxRangingCorrectionRetries() const [member function]
    cls.add_method('GetMaxRangingCorrectionRetries', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetMaxInvitedRangRetries(uint8_t maxInvitedRangRetries) [member function]
    cls.add_method('SetMaxInvitedRangRetries', 
                   'void', 
                   [param('uint8_t', 'maxInvitedRangRetries')])
    ## bs-net-device.h (module 'wimax'): uint8_t ns3::BaseStationNetDevice::GetMaxInvitedRangRetries() const [member function]
    cls.add_method('GetMaxInvitedRangRetries', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetRangReqOppSize(uint8_t rangReqOppSize) [member function]
    cls.add_method('SetRangReqOppSize', 
                   'void', 
                   [param('uint8_t', 'rangReqOppSize')])
    ## bs-net-device.h (module 'wimax'): uint8_t ns3::BaseStationNetDevice::GetRangReqOppSize() const [member function]
    cls.add_method('GetRangReqOppSize', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetBwReqOppSize(uint8_t bwReqOppSize) [member function]
    cls.add_method('SetBwReqOppSize', 
                   'void', 
                   [param('uint8_t', 'bwReqOppSize')])
    ## bs-net-device.h (module 'wimax'): uint8_t ns3::BaseStationNetDevice::GetBwReqOppSize() const [member function]
    cls.add_method('GetBwReqOppSize', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetNrDlSymbols(uint32_t dlSymbols) [member function]
    cls.add_method('SetNrDlSymbols', 
                   'void', 
                   [param('uint32_t', 'dlSymbols')])
    ## bs-net-device.h (module 'wimax'): uint32_t ns3::BaseStationNetDevice::GetNrDlSymbols() const [member function]
    cls.add_method('GetNrDlSymbols', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetNrUlSymbols(uint32_t ulSymbols) [member function]
    cls.add_method('SetNrUlSymbols', 
                   'void', 
                   [param('uint32_t', 'ulSymbols')])
    ## bs-net-device.h (module 'wimax'): uint32_t ns3::BaseStationNetDevice::GetNrUlSymbols() const [member function]
    cls.add_method('GetNrUlSymbols', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): uint32_t ns3::BaseStationNetDevice::GetNrDcdSent() const [member function]
    cls.add_method('GetNrDcdSent', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): uint32_t ns3::BaseStationNetDevice::GetNrUcdSent() const [member function]
    cls.add_method('GetNrUcdSent', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): ns3::Time ns3::BaseStationNetDevice::GetDlSubframeStartTime() const [member function]
    cls.add_method('GetDlSubframeStartTime', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): ns3::Time ns3::BaseStationNetDevice::GetUlSubframeStartTime() const [member function]
    cls.add_method('GetUlSubframeStartTime', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): uint8_t ns3::BaseStationNetDevice::GetRangingOppNumber() const [member function]
    cls.add_method('GetRangingOppNumber', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): ns3::Ptr<ns3::SSManager> ns3::BaseStationNetDevice::GetSSManager() const [member function]
    cls.add_method('GetSSManager', 
                   'ns3::Ptr< ns3::SSManager >', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetSSManager(ns3::Ptr<ns3::SSManager> ssManager) [member function]
    cls.add_method('SetSSManager', 
                   'void', 
                   [param('ns3::Ptr< ns3::SSManager >', 'ssManager')])
    ## bs-net-device.h (module 'wimax'): ns3::Ptr<ns3::UplinkScheduler> ns3::BaseStationNetDevice::GetUplinkScheduler() const [member function]
    cls.add_method('GetUplinkScheduler', 
                   'ns3::Ptr< ns3::UplinkScheduler >', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetUplinkScheduler(ns3::Ptr<ns3::UplinkScheduler> ulScheduler) [member function]
    cls.add_method('SetUplinkScheduler', 
                   'void', 
                   [param('ns3::Ptr< ns3::UplinkScheduler >', 'ulScheduler')])
    ## bs-net-device.h (module 'wimax'): ns3::Ptr<ns3::BSLinkManager> ns3::BaseStationNetDevice::GetLinkManager() const [member function]
    cls.add_method('GetLinkManager', 
                   'ns3::Ptr< ns3::BSLinkManager >', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetBSScheduler(ns3::Ptr<ns3::BSScheduler> bsSchedule) [member function]
    cls.add_method('SetBSScheduler', 
                   'void', 
                   [param('ns3::Ptr< ns3::BSScheduler >', 'bsSchedule')])
    ## bs-net-device.h (module 'wimax'): ns3::Ptr<ns3::BSScheduler> ns3::BaseStationNetDevice::GetBSScheduler() const [member function]
    cls.add_method('GetBSScheduler', 
                   'ns3::Ptr< ns3::BSScheduler >', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetLinkManager(ns3::Ptr<ns3::BSLinkManager> linkManager) [member function]
    cls.add_method('SetLinkManager', 
                   'void', 
                   [param('ns3::Ptr< ns3::BSLinkManager >', 'linkManager')])
    ## bs-net-device.h (module 'wimax'): ns3::Ptr<ns3::IpcsClassifier> ns3::BaseStationNetDevice::GetBsClassifier() const [member function]
    cls.add_method('GetBsClassifier', 
                   'ns3::Ptr< ns3::IpcsClassifier >', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetBsClassifier(ns3::Ptr<ns3::IpcsClassifier> classifier) [member function]
    cls.add_method('SetBsClassifier', 
                   'void', 
                   [param('ns3::Ptr< ns3::IpcsClassifier >', 'classifier')])
    ## bs-net-device.h (module 'wimax'): ns3::Time ns3::BaseStationNetDevice::GetPsDuration() const [member function]
    cls.add_method('GetPsDuration', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): ns3::Time ns3::BaseStationNetDevice::GetSymbolDuration() const [member function]
    cls.add_method('GetSymbolDuration', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::Start() [member function]
    cls.add_method('Start', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::Stop() [member function]
    cls.add_method('Stop', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## bs-net-device.h (module 'wimax'): bool ns3::BaseStationNetDevice::Enqueue(ns3::Ptr<ns3::Packet> packet, ns3::MacHeaderType const & hdrType, ns3::Ptr<ns3::WimaxConnection> connection) [member function]
    cls.add_method('Enqueue', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::MacHeaderType const &', 'hdrType'), param('ns3::Ptr< ns3::WimaxConnection >', 'connection')], 
                   is_virtual=True)
    ## bs-net-device.h (module 'wimax'): ns3::Ptr<ns3::WimaxConnection> ns3::BaseStationNetDevice::GetConnection(ns3::Cid cid) [member function]
    cls.add_method('GetConnection', 
                   'ns3::Ptr< ns3::WimaxConnection >', 
                   [param('ns3::Cid', 'cid')])
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::MarkUplinkAllocations() [member function]
    cls.add_method('MarkUplinkAllocations', 
                   'void', 
                   [])
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::MarkRangingOppStart(ns3::Time rangingOppStartTime) [member function]
    cls.add_method('MarkRangingOppStart', 
                   'void', 
                   [param('ns3::Time', 'rangingOppStartTime')])
    ## bs-net-device.h (module 'wimax'): ns3::Ptr<ns3::BsServiceFlowManager> ns3::BaseStationNetDevice::GetServiceFlowManager() const [member function]
    cls.add_method('GetServiceFlowManager', 
                   'ns3::Ptr< ns3::BsServiceFlowManager >', 
                   [], 
                   is_const=True)
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::SetServiceFlowManager(ns3::Ptr<ns3::BsServiceFlowManager> sfm) [member function]
    cls.add_method('SetServiceFlowManager', 
                   'void', 
                   [param('ns3::Ptr< ns3::BsServiceFlowManager >', 'sfm')])
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    ## bs-net-device.h (module 'wimax'): bool ns3::BaseStationNetDevice::DoSend(ns3::Ptr<ns3::Packet> packet, ns3::Mac48Address const & source, ns3::Mac48Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('DoSend', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Mac48Address const &', 'source'), param('ns3::Mac48Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_virtual=True, visibility='private')
    ## bs-net-device.h (module 'wimax'): void ns3::BaseStationNetDevice::DoReceive(ns3::Ptr<ns3::Packet> packet) [member function]
    cls.add_method('DoReceive', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet')], 
                   is_virtual=True, visibility='private')
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

def register_Ns3CallbackImpl__Void_Bool_Unsigned_long_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, bool, unsigned long long, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(bool arg0, long unsigned int arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('bool', 'arg0'), param('long unsigned int', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Const_ns3Mac48Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Mac48Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Mac48Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Mac48Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::Packet const >, ns3::Mac48Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Mac48Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Mac48Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Mac48Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<const ns3::Packet> arg0, ns3::Mac48Address const & arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'arg0'), param('ns3::Mac48Address const &', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Ns3Mac48Address_Const_ns3Cid___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, const ns3::Cid &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, const ns3::Cid &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, const ns3::Cid &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::Packet const >, ns3::Mac48Address, ns3::Cid const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, const ns3::Cid &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, const ns3::Cid &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, const ns3::Cid &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<const ns3::Packet> arg0, ns3::Mac48Address arg1, ns3::Cid const & arg2) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'arg0'), param('ns3::Mac48Address', 'arg1'), param('ns3::Cid const &', 'arg2')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Ns3Mac48Address_Ns3Cid_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, ns3::Cid, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, ns3::Cid, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, ns3::Cid, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::Packet const >, ns3::Mac48Address, ns3::Cid, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, ns3::Cid, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, ns3::Cid, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::Mac48Address, ns3::Cid, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<const ns3::Packet> arg0, ns3::Mac48Address arg1, ns3::Cid arg2) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'arg0'), param('ns3::Mac48Address', 'arg1'), param('ns3::Cid', 'arg2')], 
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

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3PacketBurst__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<const ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::PacketBurst const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<const ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<const ns3::PacketBurst> arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::PacketBurst const >', 'arg0')], 
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

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3PacketBurst__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::PacketBurst >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<ns3::PacketBurst>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<ns3::PacketBurst> arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::PacketBurst >', 'arg0')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3SimpleOfdmWimaxChannel_methods(root_module, cls):
    ## simple-ofdm-wimax-channel.h (module 'wimax'): ns3::SimpleOfdmWimaxChannel::SimpleOfdmWimaxChannel(ns3::SimpleOfdmWimaxChannel const & arg0) [constructor]
    cls.add_constructor([param('ns3::SimpleOfdmWimaxChannel const &', 'arg0')])
    ## simple-ofdm-wimax-channel.h (module 'wimax'): ns3::SimpleOfdmWimaxChannel::SimpleOfdmWimaxChannel() [constructor]
    cls.add_constructor([])
    ## simple-ofdm-wimax-channel.h (module 'wimax'): ns3::SimpleOfdmWimaxChannel::SimpleOfdmWimaxChannel(ns3::SimpleOfdmWimaxChannel::PropModel propModel) [constructor]
    cls.add_constructor([param('ns3::SimpleOfdmWimaxChannel::PropModel', 'propModel')])
    ## simple-ofdm-wimax-channel.h (module 'wimax'): int64_t ns3::SimpleOfdmWimaxChannel::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')], 
                   is_virtual=True)
    ## simple-ofdm-wimax-channel.h (module 'wimax'): static ns3::TypeId ns3::SimpleOfdmWimaxChannel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## simple-ofdm-wimax-channel.h (module 'wimax'): void ns3::SimpleOfdmWimaxChannel::Send(ns3::Time BlockTime, uint32_t burstSize, ns3::Ptr<ns3::WimaxPhy> phy, bool isFirstBlock, bool isLastBlock, uint64_t frequency, ns3::WimaxPhy::ModulationType modulationType, uint8_t direction, double txPowerDbm, ns3::Ptr<ns3::PacketBurst> burst) [member function]
    cls.add_method('Send', 
                   'void', 
                   [param('ns3::Time', 'BlockTime'), param('uint32_t', 'burstSize'), param('ns3::Ptr< ns3::WimaxPhy >', 'phy'), param('bool', 'isFirstBlock'), param('bool', 'isLastBlock'), param('uint64_t', 'frequency'), param('ns3::WimaxPhy::ModulationType', 'modulationType'), param('uint8_t', 'direction'), param('double', 'txPowerDbm'), param('ns3::Ptr< ns3::PacketBurst >', 'burst')])
    ## simple-ofdm-wimax-channel.h (module 'wimax'): void ns3::SimpleOfdmWimaxChannel::SetPropagationModel(ns3::SimpleOfdmWimaxChannel::PropModel propModel) [member function]
    cls.add_method('SetPropagationModel', 
                   'void', 
                   [param('ns3::SimpleOfdmWimaxChannel::PropModel', 'propModel')])
    ## simple-ofdm-wimax-channel.h (module 'wimax'): void ns3::SimpleOfdmWimaxChannel::DoAttach(ns3::Ptr<ns3::WimaxPhy> phy) [member function]
    cls.add_method('DoAttach', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxPhy >', 'phy')], 
                   is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-channel.h (module 'wimax'): ns3::Ptr<ns3::NetDevice> ns3::SimpleOfdmWimaxChannel::DoGetDevice(std::size_t i) const [member function]
    cls.add_method('DoGetDevice', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [param('std::size_t', 'i')], 
                   is_const=True, is_virtual=True, visibility='private')
    ## simple-ofdm-wimax-channel.h (module 'wimax'): std::size_t ns3::SimpleOfdmWimaxChannel::DoGetNDevices() const [member function]
    cls.add_method('DoGetNDevices', 
                   'std::size_t', 
                   [], 
                   is_const=True, is_virtual=True, visibility='private')
    return

def register_Ns3SubscriberStationNetDevice_methods(root_module, cls):
    ## ss-net-device.h (module 'wimax'): static ns3::TypeId ns3::SubscriberStationNetDevice::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## ss-net-device.h (module 'wimax'): ns3::SubscriberStationNetDevice::SubscriberStationNetDevice() [constructor]
    cls.add_constructor([])
    ## ss-net-device.h (module 'wimax'): ns3::SubscriberStationNetDevice::SubscriberStationNetDevice(ns3::Ptr<ns3::Node> node, ns3::Ptr<ns3::WimaxPhy> phy) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::Node >', 'node'), param('ns3::Ptr< ns3::WimaxPhy >', 'phy')])
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::InitSubscriberStationNetDevice() [member function]
    cls.add_method('InitSubscriberStationNetDevice', 
                   'void', 
                   [])
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetLostDlMapInterval(ns3::Time lostDlMapInterval) [member function]
    cls.add_method('SetLostDlMapInterval', 
                   'void', 
                   [param('ns3::Time', 'lostDlMapInterval')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetLostDlMapInterval() const [member function]
    cls.add_method('GetLostDlMapInterval', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetLostUlMapInterval(ns3::Time lostUlMapInterval) [member function]
    cls.add_method('SetLostUlMapInterval', 
                   'void', 
                   [param('ns3::Time', 'lostUlMapInterval')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetLostUlMapInterval() const [member function]
    cls.add_method('GetLostUlMapInterval', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetMaxDcdInterval(ns3::Time maxDcdInterval) [member function]
    cls.add_method('SetMaxDcdInterval', 
                   'void', 
                   [param('ns3::Time', 'maxDcdInterval')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetMaxDcdInterval() const [member function]
    cls.add_method('GetMaxDcdInterval', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetMaxUcdInterval(ns3::Time maxUcdInterval) [member function]
    cls.add_method('SetMaxUcdInterval', 
                   'void', 
                   [param('ns3::Time', 'maxUcdInterval')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetMaxUcdInterval() const [member function]
    cls.add_method('GetMaxUcdInterval', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetIntervalT1(ns3::Time interval1) [member function]
    cls.add_method('SetIntervalT1', 
                   'void', 
                   [param('ns3::Time', 'interval1')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetIntervalT1() const [member function]
    cls.add_method('GetIntervalT1', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetIntervalT2(ns3::Time interval2) [member function]
    cls.add_method('SetIntervalT2', 
                   'void', 
                   [param('ns3::Time', 'interval2')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetIntervalT2() const [member function]
    cls.add_method('GetIntervalT2', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetIntervalT3(ns3::Time interval3) [member function]
    cls.add_method('SetIntervalT3', 
                   'void', 
                   [param('ns3::Time', 'interval3')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetIntervalT3() const [member function]
    cls.add_method('GetIntervalT3', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetIntervalT7(ns3::Time interval7) [member function]
    cls.add_method('SetIntervalT7', 
                   'void', 
                   [param('ns3::Time', 'interval7')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetIntervalT7() const [member function]
    cls.add_method('GetIntervalT7', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetIntervalT12(ns3::Time interval12) [member function]
    cls.add_method('SetIntervalT12', 
                   'void', 
                   [param('ns3::Time', 'interval12')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetIntervalT12() const [member function]
    cls.add_method('GetIntervalT12', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetIntervalT20(ns3::Time interval20) [member function]
    cls.add_method('SetIntervalT20', 
                   'void', 
                   [param('ns3::Time', 'interval20')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetIntervalT20() const [member function]
    cls.add_method('GetIntervalT20', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetIntervalT21(ns3::Time interval21) [member function]
    cls.add_method('SetIntervalT21', 
                   'void', 
                   [param('ns3::Time', 'interval21')])
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetIntervalT21() const [member function]
    cls.add_method('GetIntervalT21', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetMaxContentionRangingRetries(uint8_t maxContentionRangingRetries) [member function]
    cls.add_method('SetMaxContentionRangingRetries', 
                   'void', 
                   [param('uint8_t', 'maxContentionRangingRetries')])
    ## ss-net-device.h (module 'wimax'): uint8_t ns3::SubscriberStationNetDevice::GetMaxContentionRangingRetries() const [member function]
    cls.add_method('GetMaxContentionRangingRetries', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetBasicConnection(ns3::Ptr<ns3::WimaxConnection> basicConnection) [member function]
    cls.add_method('SetBasicConnection', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxConnection >', 'basicConnection')])
    ## ss-net-device.h (module 'wimax'): ns3::Ptr<ns3::WimaxConnection> ns3::SubscriberStationNetDevice::GetBasicConnection() const [member function]
    cls.add_method('GetBasicConnection', 
                   'ns3::Ptr< ns3::WimaxConnection >', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetPrimaryConnection(ns3::Ptr<ns3::WimaxConnection> primaryConnection) [member function]
    cls.add_method('SetPrimaryConnection', 
                   'void', 
                   [param('ns3::Ptr< ns3::WimaxConnection >', 'primaryConnection')])
    ## ss-net-device.h (module 'wimax'): ns3::Ptr<ns3::WimaxConnection> ns3::SubscriberStationNetDevice::GetPrimaryConnection() const [member function]
    cls.add_method('GetPrimaryConnection', 
                   'ns3::Ptr< ns3::WimaxConnection >', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): ns3::Cid ns3::SubscriberStationNetDevice::GetBasicCid() const [member function]
    cls.add_method('GetBasicCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): ns3::Cid ns3::SubscriberStationNetDevice::GetPrimaryCid() const [member function]
    cls.add_method('GetPrimaryCid', 
                   'ns3::Cid', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetModulationType(ns3::WimaxPhy::ModulationType modulationType) [member function]
    cls.add_method('SetModulationType', 
                   'void', 
                   [param('ns3::WimaxPhy::ModulationType', 'modulationType')])
    ## ss-net-device.h (module 'wimax'): ns3::WimaxPhy::ModulationType ns3::SubscriberStationNetDevice::GetModulationType() const [member function]
    cls.add_method('GetModulationType', 
                   'ns3::WimaxPhy::ModulationType', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetAreManagementConnectionsAllocated(bool areManagementConnectionsAllocated) [member function]
    cls.add_method('SetAreManagementConnectionsAllocated', 
                   'void', 
                   [param('bool', 'areManagementConnectionsAllocated')])
    ## ss-net-device.h (module 'wimax'): bool ns3::SubscriberStationNetDevice::GetAreManagementConnectionsAllocated() const [member function]
    cls.add_method('GetAreManagementConnectionsAllocated', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetAreServiceFlowsAllocated(bool areServiceFlowsAllocated) [member function]
    cls.add_method('SetAreServiceFlowsAllocated', 
                   'void', 
                   [param('bool', 'areServiceFlowsAllocated')])
    ## ss-net-device.h (module 'wimax'): bool ns3::SubscriberStationNetDevice::GetAreServiceFlowsAllocated() const [member function]
    cls.add_method('GetAreServiceFlowsAllocated', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): ns3::Ptr<ns3::SSScheduler> ns3::SubscriberStationNetDevice::GetScheduler() const [member function]
    cls.add_method('GetScheduler', 
                   'ns3::Ptr< ns3::SSScheduler >', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetScheduler(ns3::Ptr<ns3::SSScheduler> ssScheduler) [member function]
    cls.add_method('SetScheduler', 
                   'void', 
                   [param('ns3::Ptr< ns3::SSScheduler >', 'ssScheduler')])
    ## ss-net-device.h (module 'wimax'): bool ns3::SubscriberStationNetDevice::HasServiceFlows() const [member function]
    cls.add_method('HasServiceFlows', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): bool ns3::SubscriberStationNetDevice::Enqueue(ns3::Ptr<ns3::Packet> packet, ns3::MacHeaderType const & hdrType, ns3::Ptr<ns3::WimaxConnection> connection) [member function]
    cls.add_method('Enqueue', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::MacHeaderType const &', 'hdrType'), param('ns3::Ptr< ns3::WimaxConnection >', 'connection')], 
                   is_virtual=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SendBurst(uint8_t uiuc, uint16_t nrSymbols, ns3::Ptr<ns3::WimaxConnection> connection, ns3::MacHeaderType::HeaderType packetType=::ns3::MacHeaderType::HeaderType::HEADER_TYPE_GENERIC) [member function]
    cls.add_method('SendBurst', 
                   'void', 
                   [param('uint8_t', 'uiuc'), param('uint16_t', 'nrSymbols'), param('ns3::Ptr< ns3::WimaxConnection >', 'connection'), param('ns3::MacHeaderType::HeaderType', 'packetType', default_value='::ns3::MacHeaderType::HeaderType::HEADER_TYPE_GENERIC')])
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::Start() [member function]
    cls.add_method('Start', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::Stop() [member function]
    cls.add_method('Stop', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::AddServiceFlow(ns3::ServiceFlow * sf) [member function]
    cls.add_method('AddServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow *', 'sf')])
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::AddServiceFlow(ns3::ServiceFlow sf) [member function]
    cls.add_method('AddServiceFlow', 
                   'void', 
                   [param('ns3::ServiceFlow', 'sf')])
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetTimer(ns3::EventId eventId, ns3::EventId & event) [member function]
    cls.add_method('SetTimer', 
                   'void', 
                   [param('ns3::EventId', 'eventId'), param('ns3::EventId &', 'event')])
    ## ss-net-device.h (module 'wimax'): bool ns3::SubscriberStationNetDevice::IsRegistered() const [member function]
    cls.add_method('IsRegistered', 
                   'bool', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): ns3::Time ns3::SubscriberStationNetDevice::GetTimeToAllocation(ns3::Time deferTime) [member function]
    cls.add_method('GetTimeToAllocation', 
                   'ns3::Time', 
                   [param('ns3::Time', 'deferTime')])
    ## ss-net-device.h (module 'wimax'): ns3::SubscriberStationNetDevice::m_linkManager [variable]
    cls.add_instance_attribute('m_linkManager', 'ns3::Ptr< ns3::SSLinkManager >', is_const=False)
    ## ss-net-device.h (module 'wimax'): ns3::Ptr<ns3::IpcsClassifier> ns3::SubscriberStationNetDevice::GetIpcsClassifier() const [member function]
    cls.add_method('GetIpcsClassifier', 
                   'ns3::Ptr< ns3::IpcsClassifier >', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetIpcsPacketClassifier(ns3::Ptr<ns3::IpcsClassifier> classifier) [member function]
    cls.add_method('SetIpcsPacketClassifier', 
                   'void', 
                   [param('ns3::Ptr< ns3::IpcsClassifier >', 'classifier')])
    ## ss-net-device.h (module 'wimax'): ns3::Ptr<ns3::SSLinkManager> ns3::SubscriberStationNetDevice::GetLinkManager() const [member function]
    cls.add_method('GetLinkManager', 
                   'ns3::Ptr< ns3::SSLinkManager >', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetLinkManager(ns3::Ptr<ns3::SSLinkManager> linkManager) [member function]
    cls.add_method('SetLinkManager', 
                   'void', 
                   [param('ns3::Ptr< ns3::SSLinkManager >', 'linkManager')])
    ## ss-net-device.h (module 'wimax'): ns3::Ptr<ns3::SsServiceFlowManager> ns3::SubscriberStationNetDevice::GetServiceFlowManager() const [member function]
    cls.add_method('GetServiceFlowManager', 
                   'ns3::Ptr< ns3::SsServiceFlowManager >', 
                   [], 
                   is_const=True)
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetServiceFlowManager(ns3::Ptr<ns3::SsServiceFlowManager> sfm) [member function]
    cls.add_method('SetServiceFlowManager', 
                   'void', 
                   [param('ns3::Ptr< ns3::SsServiceFlowManager >', 'sfm')])
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetAsciiTxQueueEnqueueCallback(ns3::SubscriberStationNetDevice::AsciiTraceCallback cb) [member function]
    cls.add_method('SetAsciiTxQueueEnqueueCallback', 
                   'void', 
                   [param('ns3::Callback< void, std::basic_string< char >, ns3::Ptr< ns3::Packet const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')])
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetAsciiTxQueueDequeueCallback(ns3::SubscriberStationNetDevice::AsciiTraceCallback cb) [member function]
    cls.add_method('SetAsciiTxQueueDequeueCallback', 
                   'void', 
                   [param('ns3::Callback< void, std::basic_string< char >, ns3::Ptr< ns3::Packet const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')])
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::SetAsciiTxQueueDropCallback(ns3::SubscriberStationNetDevice::AsciiTraceCallback cb) [member function]
    cls.add_method('SetAsciiTxQueueDropCallback', 
                   'void', 
                   [param('ns3::Callback< void, std::basic_string< char >, ns3::Ptr< ns3::Packet const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')])
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    ## ss-net-device.h (module 'wimax'): bool ns3::SubscriberStationNetDevice::DoSend(ns3::Ptr<ns3::Packet> packet, ns3::Mac48Address const & source, ns3::Mac48Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('DoSend', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Mac48Address const &', 'source'), param('ns3::Mac48Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_virtual=True, visibility='private')
    ## ss-net-device.h (module 'wimax'): void ns3::SubscriberStationNetDevice::DoReceive(ns3::Ptr<ns3::Packet> packet) [member function]
    cls.add_method('DoReceive', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet')], 
                   is_virtual=True, visibility='private')
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

def register_functions(root_module):
    module = root_module
    ## crc8.h (module 'wimax'): uint8_t ns3::CRC8Calculate(uint8_t const * data, int length) [free function]
    module.add_function('CRC8Calculate', 
                        'uint8_t', 
                        [param('uint8_t const *', 'data'), param('int', 'length')])
    register_functions_ns3_FatalImpl(module.add_cpp_namespace('FatalImpl'), root_module)
    register_functions_ns3_Hash(module.add_cpp_namespace('Hash'), root_module)
    register_functions_ns3_TracedValueCallback(module.add_cpp_namespace('TracedValueCallback'), root_module)
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

