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
    root_module = Module('ns.network', cpp_namespace='::ns3')
    return root_module

def register_types(module):
    root_module = module.get_root()
    
    ## packetbb.h (module 'network'): ns3::PbbAddressLength [enumeration]
    module.add_enum('PbbAddressLength', ['IPV4', 'IPV6'])
    ## ethernet-header.h (module 'network'): ns3::ethernet_header_t [enumeration]
    module.add_enum('ethernet_header_t', ['LENGTH', 'VLAN', 'QINQ'])
    ## queue-size.h (module 'network'): ns3::QueueSizeUnit [enumeration]
    module.add_enum('QueueSizeUnit', ['PACKETS', 'BYTES'])
    ## log.h (module 'core'): ns3::LogLevel [enumeration]
    module.add_enum('LogLevel', ['LOG_NONE', 'LOG_ERROR', 'LOG_LEVEL_ERROR', 'LOG_WARN', 'LOG_LEVEL_WARN', 'LOG_DEBUG', 'LOG_LEVEL_DEBUG', 'LOG_INFO', 'LOG_LEVEL_INFO', 'LOG_FUNCTION', 'LOG_LEVEL_FUNCTION', 'LOG_LOGIC', 'LOG_LEVEL_LOGIC', 'LOG_ALL', 'LOG_LEVEL_ALL', 'LOG_PREFIX_FUNC', 'LOG_PREFIX_TIME', 'LOG_PREFIX_NODE', 'LOG_PREFIX_LEVEL', 'LOG_PREFIX_ALL'], import_from_module='ns.core')
    ## address.h (module 'network'): ns3::Address [class]
    module.add_class('Address')
    ## address.h (module 'network'): ns3::Address::MaxSize_e [enumeration]
    module.add_enum('MaxSize_e', ['MAX_SIZE'], outer_class=root_module['ns3::Address'])
    ## application-container.h (module 'network'): ns3::ApplicationContainer [class]
    module.add_class('ApplicationContainer')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Application > > const_iterator', 'ns3::ApplicationContainer::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Application > > const_iterator*', 'ns3::ApplicationContainer::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Application > > const_iterator&', 'ns3::ApplicationContainer::Iterator&')
    ## trace-helper.h (module 'network'): ns3::AsciiTraceHelper [class]
    module.add_class('AsciiTraceHelper')
    ## trace-helper.h (module 'network'): ns3::AsciiTraceHelperForDevice [class]
    module.add_class('AsciiTraceHelperForDevice', allow_subclassing=True)
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList [class]
    module.add_class('AttributeConstructionList', import_from_module='ns.core')
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::Item [struct]
    module.add_class('Item', import_from_module='ns.core', outer_class=root_module['ns3::AttributeConstructionList'])
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator', 'ns3::AttributeConstructionList::CIterator')
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator*', 'ns3::AttributeConstructionList::CIterator*')
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator&', 'ns3::AttributeConstructionList::CIterator&')
    ## bit-deserializer.h (module 'network'): ns3::BitDeserializer [class]
    module.add_class('BitDeserializer')
    ## bit-serializer.h (module 'network'): ns3::BitSerializer [class]
    module.add_class('BitSerializer')
    ## buffer.h (module 'network'): ns3::Buffer [class]
    module.add_class('Buffer')
    ## buffer.h (module 'network'): ns3::Buffer::Iterator [class]
    module.add_class('Iterator', outer_class=root_module['ns3::Buffer'])
    ## packet.h (module 'network'): ns3::ByteTagIterator [class]
    module.add_class('ByteTagIterator')
    ## packet.h (module 'network'): ns3::ByteTagIterator::Item [class]
    module.add_class('Item', outer_class=root_module['ns3::ByteTagIterator'])
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList [class]
    module.add_class('ByteTagList')
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator [class]
    module.add_class('Iterator', outer_class=root_module['ns3::ByteTagList'])
    ## byte-tag-list.h (module 'network'): ns3::ByteTagList::Iterator::Item [struct]
    module.add_class('Item', outer_class=root_module['ns3::ByteTagList::Iterator'])
    ## callback.h (module 'core'): ns3::CallbackBase [class]
    module.add_class('CallbackBase', import_from_module='ns.core')
    ## channel-list.h (module 'network'): ns3::ChannelList [class]
    module.add_class('ChannelList')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Channel > > const_iterator', 'ns3::ChannelList::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Channel > > const_iterator*', 'ns3::ChannelList::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Channel > > const_iterator&', 'ns3::ChannelList::Iterator&')
    ## data-output-interface.h (module 'stats'): ns3::DataOutputCallback [class]
    module.add_class('DataOutputCallback', allow_subclassing=True, import_from_module='ns.stats')
    ## data-rate.h (module 'network'): ns3::DataRate [class]
    module.add_class('DataRate')
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
    module.add_class('DefaultDeleter', template_parameters=['ns3::NixVector'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::OutputStreamWrapper> [struct]
    module.add_class('DefaultDeleter', template_parameters=['ns3::OutputStreamWrapper'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::Packet> [struct]
    module.add_class('DefaultDeleter', template_parameters=['ns3::Packet'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::PbbAddressBlock> [struct]
    module.add_class('DefaultDeleter', template_parameters=['ns3::PbbAddressBlock'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::PbbMessage> [struct]
    module.add_class('DefaultDeleter', template_parameters=['ns3::PbbMessage'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::PbbTlv> [struct]
    module.add_class('DefaultDeleter', template_parameters=['ns3::PbbTlv'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::QueueItem> [struct]
    module.add_class('DefaultDeleter', template_parameters=['ns3::QueueItem'])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::TraceSourceAccessor> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::TraceSourceAccessor'])
    ## delay-jitter-estimation.h (module 'network'): ns3::DelayJitterEstimation [class]
    module.add_class('DelayJitterEstimation')
    ## event-id.h (module 'core'): ns3::EventId [class]
    module.add_class('EventId', import_from_module='ns.core')
    ## event-id.h (module 'core'): ns3::EventId::UID [enumeration]
    module.add_enum('UID', ['INVALID', 'NOW', 'DESTROY', 'RESERVED', 'VALID'], outer_class=root_module['ns3::EventId'], import_from_module='ns.core')
    ## hash.h (module 'core'): ns3::Hasher [class]
    module.add_class('Hasher', import_from_module='ns.core')
    ## inet6-socket-address.h (module 'network'): ns3::Inet6SocketAddress [class]
    module.add_class('Inet6SocketAddress')
    ## inet6-socket-address.h (module 'network'): ns3::Inet6SocketAddress [class]
    root_module['ns3::Inet6SocketAddress'].implicitly_converts_to(root_module['ns3::Address'])
    ## inet-socket-address.h (module 'network'): ns3::InetSocketAddress [class]
    module.add_class('InetSocketAddress')
    ## inet-socket-address.h (module 'network'): ns3::InetSocketAddress [class]
    root_module['ns3::InetSocketAddress'].implicitly_converts_to(root_module['ns3::Address'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address [class]
    module.add_class('Ipv4Address')
    ## ipv4-address.h (module 'network'): ns3::Ipv4Address [class]
    root_module['ns3::Ipv4Address'].implicitly_converts_to(root_module['ns3::Address'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressHash [class]
    module.add_class('Ipv4AddressHash')
    ## ipv4-address.h (module 'network'): ns3::Ipv4Mask [class]
    module.add_class('Ipv4Mask')
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address [class]
    module.add_class('Ipv6Address')
    ## ipv6-address.h (module 'network'): ns3::Ipv6Address [class]
    root_module['ns3::Ipv6Address'].implicitly_converts_to(root_module['ns3::Address'])
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressHash [class]
    module.add_class('Ipv6AddressHash')
    ## ipv6-address.h (module 'network'): ns3::Ipv6Prefix [class]
    module.add_class('Ipv6Prefix')
    ## log.h (module 'core'): ns3::LogComponent [class]
    module.add_class('LogComponent', import_from_module='ns.core')
    typehandlers.add_type_alias('std::map< std::string, ns3::LogComponent * >', 'ns3::LogComponent::ComponentList')
    typehandlers.add_type_alias('std::map< std::string, ns3::LogComponent * >*', 'ns3::LogComponent::ComponentList*')
    typehandlers.add_type_alias('std::map< std::string, ns3::LogComponent * >&', 'ns3::LogComponent::ComponentList&')
    ## mac16-address.h (module 'network'): ns3::Mac16Address [class]
    module.add_class('Mac16Address')
    ## mac16-address.h (module 'network'): ns3::Mac16Address [class]
    root_module['ns3::Mac16Address'].implicitly_converts_to(root_module['ns3::Address'])
    ## mac48-address.h (module 'network'): ns3::Mac48Address [class]
    module.add_class('Mac48Address')
    typehandlers.add_type_alias('void ( * ) ( ns3::Mac48Address )', 'ns3::Mac48Address::TracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Mac48Address )*', 'ns3::Mac48Address::TracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Mac48Address )&', 'ns3::Mac48Address::TracedCallback&')
    ## mac48-address.h (module 'network'): ns3::Mac48Address [class]
    root_module['ns3::Mac48Address'].implicitly_converts_to(root_module['ns3::Address'])
    ## mac64-address.h (module 'network'): ns3::Mac64Address [class]
    module.add_class('Mac64Address')
    ## mac64-address.h (module 'network'): ns3::Mac64Address [class]
    root_module['ns3::Mac64Address'].implicitly_converts_to(root_module['ns3::Address'])
    ## mac8-address.h (module 'network'): ns3::Mac8Address [class]
    module.add_class('Mac8Address')
    ## mac8-address.h (module 'network'): ns3::Mac8Address [class]
    root_module['ns3::Mac8Address'].implicitly_converts_to(root_module['ns3::Address'])
    ## net-device-container.h (module 'network'): ns3::NetDeviceContainer [class]
    module.add_class('NetDeviceContainer')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::NetDevice > > const_iterator', 'ns3::NetDeviceContainer::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::NetDevice > > const_iterator*', 'ns3::NetDeviceContainer::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::NetDevice > > const_iterator&', 'ns3::NetDeviceContainer::Iterator&')
    ## node-container.h (module 'network'): ns3::NodeContainer [class]
    module.add_class('NodeContainer')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Node > > const_iterator', 'ns3::NodeContainer::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Node > > const_iterator*', 'ns3::NodeContainer::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Node > > const_iterator&', 'ns3::NodeContainer::Iterator&')
    ## node-list.h (module 'network'): ns3::NodeList [class]
    module.add_class('NodeList')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Node > > const_iterator', 'ns3::NodeList::Iterator')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Node > > const_iterator*', 'ns3::NodeList::Iterator*')
    typehandlers.add_type_alias('std::vector< ns3::Ptr< ns3::Node > > const_iterator&', 'ns3::NodeList::Iterator&')
    ## object-base.h (module 'core'): ns3::ObjectBase [class]
    module.add_class('ObjectBase', allow_subclassing=True, import_from_module='ns.core')
    ## object.h (module 'core'): ns3::ObjectDeleter [struct]
    module.add_class('ObjectDeleter', import_from_module='ns.core')
    ## object-factory.h (module 'core'): ns3::ObjectFactory [class]
    module.add_class('ObjectFactory', import_from_module='ns.core')
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata [class]
    module.add_class('PacketMetadata')
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item [struct]
    module.add_class('Item', outer_class=root_module['ns3::PacketMetadata'])
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::Item::ItemType [enumeration]
    module.add_enum('ItemType', ['PAYLOAD', 'HEADER', 'TRAILER'], outer_class=root_module['ns3::PacketMetadata::Item'])
    ## packet-metadata.h (module 'network'): ns3::PacketMetadata::ItemIterator [class]
    module.add_class('ItemIterator', outer_class=root_module['ns3::PacketMetadata'])
    ## packet-socket-address.h (module 'network'): ns3::PacketSocketAddress [class]
    module.add_class('PacketSocketAddress')
    ## packet-socket-address.h (module 'network'): ns3::PacketSocketAddress [class]
    root_module['ns3::PacketSocketAddress'].implicitly_converts_to(root_module['ns3::Address'])
    ## packet-socket-helper.h (module 'network'): ns3::PacketSocketHelper [class]
    module.add_class('PacketSocketHelper')
    ## packet.h (module 'network'): ns3::PacketTagIterator [class]
    module.add_class('PacketTagIterator')
    ## packet.h (module 'network'): ns3::PacketTagIterator::Item [class]
    module.add_class('Item', outer_class=root_module['ns3::PacketTagIterator'])
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList [class]
    module.add_class('PacketTagList')
    ## packet-tag-list.h (module 'network'): ns3::PacketTagList::TagData [struct]
    module.add_class('TagData', outer_class=root_module['ns3::PacketTagList'])
    ## log.h (module 'core'): ns3::ParameterLogger [class]
    module.add_class('ParameterLogger', import_from_module='ns.core')
    ## packetbb.h (module 'network'): ns3::PbbAddressTlvBlock [class]
    module.add_class('PbbAddressTlvBlock')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressTlv > > iterator', 'ns3::PbbAddressTlvBlock::Iterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressTlv > > iterator*', 'ns3::PbbAddressTlvBlock::Iterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressTlv > > iterator&', 'ns3::PbbAddressTlvBlock::Iterator&')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressTlv > > const_iterator', 'ns3::PbbAddressTlvBlock::ConstIterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressTlv > > const_iterator*', 'ns3::PbbAddressTlvBlock::ConstIterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressTlv > > const_iterator&', 'ns3::PbbAddressTlvBlock::ConstIterator&')
    ## packetbb.h (module 'network'): ns3::PbbTlvBlock [class]
    module.add_class('PbbTlvBlock')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'ns3::PbbTlvBlock::Iterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > iterator*', 'ns3::PbbTlvBlock::Iterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > iterator&', 'ns3::PbbTlvBlock::Iterator&')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > const_iterator', 'ns3::PbbTlvBlock::ConstIterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > const_iterator*', 'ns3::PbbTlvBlock::ConstIterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > const_iterator&', 'ns3::PbbTlvBlock::ConstIterator&')
    ## pcap-file.h (module 'network'): ns3::PcapFile [class]
    module.add_class('PcapFile')
    ## trace-helper.h (module 'network'): ns3::PcapHelper [class]
    module.add_class('PcapHelper')
    ## trace-helper.h (module 'network'): ns3::PcapHelper::DataLinkType [enumeration]
    module.add_enum('DataLinkType', ['DLT_NULL', 'DLT_EN10MB', 'DLT_PPP', 'DLT_RAW', 'DLT_IEEE802_11', 'DLT_LINUX_SLL', 'DLT_PRISM_HEADER', 'DLT_IEEE802_11_RADIO', 'DLT_IEEE802_15_4', 'DLT_NETLINK', 'DLT_LORATAP'], outer_class=root_module['ns3::PcapHelper'])
    ## trace-helper.h (module 'network'): ns3::PcapHelperForDevice [class]
    module.add_class('PcapHelperForDevice', allow_subclassing=True)
    ## queue-size.h (module 'network'): ns3::QueueSize [class]
    module.add_class('QueueSize')
    ## sequence-number.h (module 'network'): ns3::SequenceNumber<unsigned int, int> [class]
    module.add_class('SequenceNumber32')
    ## sequence-number.h (module 'network'): ns3::SequenceNumber<unsigned short, short> [class]
    module.add_class('SequenceNumber16')
    ## simple-net-device-helper.h (module 'network'): ns3::SimpleNetDeviceHelper [class]
    module.add_class('SimpleNetDeviceHelper')
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::ObjectBase'], template_parameters=['ns3::Object', 'ns3::ObjectBase', 'ns3::ObjectDeleter'])
    ## simulator.h (module 'core'): ns3::Simulator [class]
    module.add_class('Simulator', destructor_visibility='private', import_from_module='ns.core')
    ## simulator.h (module 'core'): ns3::Simulator [enumeration]
    module.add_enum('', ['NO_CONTEXT'], outer_class=root_module['ns3::Simulator'], import_from_module='ns.core')
    ## data-calculator.h (module 'stats'): ns3::StatisticalSummary [class]
    module.add_class('StatisticalSummary', allow_subclassing=True, import_from_module='ns.stats')
    ## system-wall-clock-ms.h (module 'core'): ns3::SystemWallClockMs [class]
    module.add_class('SystemWallClockMs', import_from_module='ns.core')
    ## tag.h (module 'network'): ns3::Tag [class]
    module.add_class('Tag', parent=root_module['ns3::ObjectBase'])
    ## tag-buffer.h (module 'network'): ns3::TagBuffer [class]
    module.add_class('TagBuffer')
    ## nstime.h (module 'core'): ns3::Time [class]
    module.add_class('Time', import_from_module='ns.core')
    ## nstime.h (module 'core'): ns3::Time::Unit [enumeration]
    module.add_enum('Unit', ['Y', 'D', 'H', 'MIN', 'S', 'MS', 'US', 'NS', 'PS', 'FS', 'LAST', 'AUTO'], outer_class=root_module['ns3::Time'], import_from_module='ns.core')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time )', 'ns3::Time::TracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time )*', 'ns3::Time::TracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time )&', 'ns3::Time::TracedCallback&')
    ## nstime.h (module 'core'): ns3::TimeWithUnit [class]
    module.add_class('TimeWithUnit', import_from_module='ns.core')
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned int> [class]
    module.add_class('TracedValue', import_from_module='ns.core', template_parameters=['unsigned int'])
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
    ## empty.h (module 'core'): ns3::empty [class]
    module.add_class('empty', import_from_module='ns.core')
    ## int64x64-128.h (module 'core'): ns3::int64x64_t [class]
    module.add_class('int64x64_t', import_from_module='ns.core')
    ## int64x64-128.h (module 'core'): ns3::int64x64_t::impl_type [enumeration]
    module.add_enum('impl_type', ['int128_impl', 'cairo_impl', 'ld_impl'], outer_class=root_module['ns3::int64x64_t'], import_from_module='ns.core')
    ## chunk.h (module 'network'): ns3::Chunk [class]
    module.add_class('Chunk', parent=root_module['ns3::ObjectBase'])
    ## packet-socket.h (module 'network'): ns3::DeviceNameTag [class]
    module.add_class('DeviceNameTag', parent=root_module['ns3::Tag'])
    ## flow-id-tag.h (module 'network'): ns3::FlowIdTag [class]
    module.add_class('FlowIdTag', parent=root_module['ns3::Tag'])
    ## header.h (module 'network'): ns3::Header [class]
    module.add_class('Header', parent=root_module['ns3::Chunk'])
    ## llc-snap-header.h (module 'network'): ns3::LlcSnapHeader [class]
    module.add_class('LlcSnapHeader', parent=root_module['ns3::Header'])
    ## object.h (module 'core'): ns3::Object [class]
    module.add_class('Object', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter >'])
    ## object.h (module 'core'): ns3::Object::AggregateIterator [class]
    module.add_class('AggregateIterator', import_from_module='ns.core', outer_class=root_module['ns3::Object'])
    ## packet-burst.h (module 'network'): ns3::PacketBurst [class]
    module.add_class('PacketBurst', parent=root_module['ns3::Object'])
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::PacketBurst const > )', 'ns3::PacketBurst::TracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::PacketBurst const > )*', 'ns3::PacketBurst::TracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::PacketBurst const > )&', 'ns3::PacketBurst::TracedCallback&')
    ## packet-socket.h (module 'network'): ns3::PacketSocketTag [class]
    module.add_class('PacketSocketTag', parent=root_module['ns3::Tag'])
    ## pcap-file-wrapper.h (module 'network'): ns3::PcapFileWrapper [class]
    module.add_class('PcapFileWrapper', parent=root_module['ns3::Object'])
    ## queue.h (module 'network'): ns3::QueueBase [class]
    module.add_class('QueueBase', parent=root_module['ns3::Object'])
    ## queue-limits.h (module 'network'): ns3::QueueLimits [class]
    module.add_class('QueueLimits', parent=root_module['ns3::Object'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader [class]
    module.add_class('RadiotapHeader', parent=root_module['ns3::Header'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::FrameFlag [enumeration]
    module.add_enum('FrameFlag', ['FRAME_FLAG_NONE', 'FRAME_FLAG_CFP', 'FRAME_FLAG_SHORT_PREAMBLE', 'FRAME_FLAG_WEP', 'FRAME_FLAG_FRAGMENTED', 'FRAME_FLAG_FCS_INCLUDED', 'FRAME_FLAG_DATA_PADDING', 'FRAME_FLAG_BAD_FCS', 'FRAME_FLAG_SHORT_GUARD'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::ChannelFlags [enumeration]
    module.add_enum('ChannelFlags', ['CHANNEL_FLAG_NONE', 'CHANNEL_FLAG_TURBO', 'CHANNEL_FLAG_CCK', 'CHANNEL_FLAG_OFDM', 'CHANNEL_FLAG_SPECTRUM_2GHZ', 'CHANNEL_FLAG_SPECTRUM_5GHZ', 'CHANNEL_FLAG_PASSIVE', 'CHANNEL_FLAG_DYNAMIC', 'CHANNEL_FLAG_GFSK'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::McsKnown [enumeration]
    module.add_enum('McsKnown', ['MCS_KNOWN_NONE', 'MCS_KNOWN_BANDWIDTH', 'MCS_KNOWN_INDEX', 'MCS_KNOWN_GUARD_INTERVAL', 'MCS_KNOWN_HT_FORMAT', 'MCS_KNOWN_FEC_TYPE', 'MCS_KNOWN_STBC', 'MCS_KNOWN_NESS', 'MCS_KNOWN_NESS_BIT_1'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::McsFlags [enumeration]
    module.add_enum('McsFlags', ['MCS_FLAGS_NONE', 'MCS_FLAGS_BANDWIDTH_40', 'MCS_FLAGS_BANDWIDTH_20L', 'MCS_FLAGS_BANDWIDTH_20U', 'MCS_FLAGS_GUARD_INTERVAL', 'MCS_FLAGS_HT_GREENFIELD', 'MCS_FLAGS_FEC_TYPE', 'MCS_FLAGS_STBC_STREAMS', 'MCS_FLAGS_NESS_BIT_0'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::AmpduFlags [enumeration]
    module.add_enum('AmpduFlags', ['A_MPDU_STATUS_NONE', 'A_MPDU_STATUS_REPORT_ZERO_LENGTH', 'A_MPDU_STATUS_IS_ZERO_LENGTH', 'A_MPDU_STATUS_LAST_KNOWN', 'A_MPDU_STATUS_LAST', 'A_MPDU_STATUS_DELIMITER_CRC_ERROR', 'A_MPDU_STATUS_DELIMITER_CRC_KNOWN'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::VhtKnown [enumeration]
    module.add_enum('VhtKnown', ['VHT_KNOWN_NONE', 'VHT_KNOWN_STBC', 'VHT_KNOWN_TXOP_PS_NOT_ALLOWED', 'VHT_KNOWN_GUARD_INTERVAL', 'VHT_KNOWN_SHORT_GI_NSYM_DISAMBIGUATION', 'VHT_KNOWN_LDPC_EXTRA_OFDM_SYMBOL', 'VHT_KNOWN_BEAMFORMED', 'VHT_KNOWN_BANDWIDTH', 'VHT_KNOWN_GROUP_ID', 'VHT_KNOWN_PARTIAL_AID'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::VhtFlags [enumeration]
    module.add_enum('VhtFlags', ['VHT_FLAGS_NONE', 'VHT_FLAGS_STBC', 'VHT_FLAGS_TXOP_PS_NOT_ALLOWED', 'VHT_FLAGS_GUARD_INTERVAL', 'VHT_FLAGS_SHORT_GI_NSYM_DISAMBIGUATION', 'VHT_FLAGS_LDPC_EXTRA_OFDM_SYMBOL', 'VHT_FLAGS_BEAMFORMED'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::HeData1 [enumeration]
    module.add_enum('HeData1', ['HE_DATA1_FORMAT_EXT_SU', 'HE_DATA1_FORMAT_MU', 'HE_DATA1_FORMAT_TRIG', 'HE_DATA1_BSS_COLOR_KNOWN', 'HE_DATA1_BEAM_CHANGE_KNOWN', 'HE_DATA1_UL_DL_KNOWN', 'HE_DATA1_DATA_MCS_KNOWN', 'HE_DATA1_DATA_DCM_KNOWN', 'HE_DATA1_CODING_KNOWN', 'HE_DATA1_LDPC_XSYMSEG_KNOWN', 'HE_DATA1_STBC_KNOWN', 'HE_DATA1_SPTL_REUSE_KNOWN', 'HE_DATA1_SPTL_REUSE2_KNOWN', 'HE_DATA1_SPTL_REUSE3_KNOWN', 'HE_DATA1_SPTL_REUSE4_KNOWN', 'HE_DATA1_BW_RU_ALLOC_KNOWN', 'HE_DATA1_DOPPLER_KNOWN'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::HeData2 [enumeration]
    module.add_enum('HeData2', ['HE_DATA2_PRISEC_80_KNOWN', 'HE_DATA2_GI_KNOWN', 'HE_DATA2_NUM_LTF_SYMS_KNOWN', 'HE_DATA2_PRE_FEC_PAD_KNOWN', 'HE_DATA2_TXBF_KNOWN', 'HE_DATA2_PE_DISAMBIG_KNOWN', 'HE_DATA2_TXOP_KNOWN', 'HE_DATA2_MIDAMBLE_KNOWN', 'HE_DATA2_RU_OFFSET', 'HE_DATA2_RU_OFFSET_KNOWN', 'HE_DATA2_PRISEC_80_SEC'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::HeData5 [enumeration]
    module.add_enum('HeData5', ['HE_DATA5_DATA_BW_RU_ALLOC_40MHZ', 'HE_DATA5_DATA_BW_RU_ALLOC_80MHZ', 'HE_DATA5_DATA_BW_RU_ALLOC_160MHZ', 'HE_DATA5_DATA_BW_RU_ALLOC_26T', 'HE_DATA5_DATA_BW_RU_ALLOC_52T', 'HE_DATA5_DATA_BW_RU_ALLOC_106T', 'HE_DATA5_DATA_BW_RU_ALLOC_242T', 'HE_DATA5_DATA_BW_RU_ALLOC_484T', 'HE_DATA5_DATA_BW_RU_ALLOC_996T', 'HE_DATA5_DATA_BW_RU_ALLOC_2x996T', 'HE_DATA5_GI_1_6', 'HE_DATA5_GI_3_2', 'HE_DATA5_LTF_SYM_SIZE', 'HE_DATA5_NUM_LTF_SYMS', 'HE_DATA5_PRE_FEC_PAD', 'HE_DATA5_TXBF', 'HE_DATA5_PE_DISAMBIG'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::HeMuFlags1 [enumeration]
    module.add_enum('HeMuFlags1', ['HE_MU_FLAGS1_SIGB_MCS', 'HE_MU_FLAGS1_SIGB_MCS_KNOWN', 'HE_MU_FLAGS1_SIGB_DCM', 'HE_MU_FLAGS1_SIGB_DCM_KNOWN', 'HE_MU_FLAGS1_CH2_CENTER_26T_RU_KNOWN', 'HE_MU_FLAGS1_CH1_RUS_KNOWN', 'HE_MU_FLAGS1_CH2_RUS_KNOWN', 'HE_MU_FLAGS1_CH1_CENTER_26T_RU_KNOWN', 'HE_MU_FLAGS1_CH1_CENTER_26T_RU', 'HE_MU_FLAGS1_SIGB_COMPRESSION_KNOWN', 'HE_MU_FLAGS1_NUM_SIGB_SYMBOLS_KNOWN'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::HeMuFlags2 [enumeration]
    module.add_enum('HeMuFlags2', ['HE_MU_FLAGS2_BW_FROM_SIGA', 'HE_MU_FLAGS2_BW_FROM_SIGA_KNOWN', 'HE_MU_FLAGS2_SIGB_COMPRESSION_FROM_SIGA', 'HE_MU_FLAGS2_NUM_SIGB_SYMBOLS_FROM_SIGA', 'HE_MU_FLAGS2_PREAMBLE_PUNCTURING_FROM_SIGA_BW_FIELD', 'HE_MU_FLAGS2_PREAMBLE_PUNCTURING_FROM_SIGA_BW_FIELD_KNOWN', 'HE_MU_FLAGS2_CH2_CENTER_26T_RU'], outer_class=root_module['ns3::RadiotapHeader'])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::HeMuPerUserKnown [enumeration]
    module.add_enum('HeMuPerUserKnown', ['HE_MU_PER_USER_POSITION_KNOWN', 'HE_MU_PER_USER_STA_ID_KNOWN', 'HE_MU_PER_USER_NSTS_KNOWN', 'HE_MU_PER_USER_TX_BF_KNOWN', 'HE_MU_PER_USER_SPATIAL_CONFIGURATION_KNOWN', 'HE_MU_PER_USER_MCS_KNOWN', 'HE_MU_PER_USER_DCM_KNOWN', 'HE_MU_PER_USER_CODING_KNOWN'], outer_class=root_module['ns3::RadiotapHeader'])
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
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::NixVector', 'ns3::empty', 'ns3::DefaultDeleter<ns3::NixVector>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::OutputStreamWrapper, ns3::empty, ns3::DefaultDeleter<ns3::OutputStreamWrapper>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::OutputStreamWrapper', 'ns3::empty', 'ns3::DefaultDeleter<ns3::OutputStreamWrapper>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::Packet', 'ns3::empty', 'ns3::DefaultDeleter<ns3::Packet>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbAddressBlock, ns3::empty, ns3::DefaultDeleter<ns3::PbbAddressBlock>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::PbbAddressBlock', 'ns3::empty', 'ns3::DefaultDeleter<ns3::PbbAddressBlock>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbMessage, ns3::empty, ns3::DefaultDeleter<ns3::PbbMessage>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::PbbMessage', 'ns3::empty', 'ns3::DefaultDeleter<ns3::PbbMessage>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbPacket, ns3::Header, ns3::DefaultDeleter<ns3::PbbPacket>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::Header'], template_parameters=['ns3::PbbPacket', 'ns3::Header', 'ns3::DefaultDeleter<ns3::PbbPacket>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbTlv, ns3::empty, ns3::DefaultDeleter<ns3::PbbTlv>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::PbbTlv', 'ns3::empty', 'ns3::DefaultDeleter<ns3::PbbTlv>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::QueueItem, ns3::empty, ns3::DefaultDeleter<ns3::QueueItem>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::QueueItem', 'ns3::empty', 'ns3::DefaultDeleter<ns3::QueueItem>'])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::TraceSourceAccessor', 'ns3::empty', 'ns3::DefaultDeleter<ns3::TraceSourceAccessor>'])
    ## sll-header.h (module 'network'): ns3::SllHeader [class]
    module.add_class('SllHeader', parent=root_module['ns3::Header'])
    ## sll-header.h (module 'network'): ns3::SllHeader::PacketType [enumeration]
    module.add_enum('PacketType', ['UNICAST_FROM_PEER_TO_ME', 'BROADCAST_BY_PEER', 'MULTICAST_BY_PEER', 'INTERCEPTED_PACKET', 'SENT_BY_US'], outer_class=root_module['ns3::SllHeader'])
    ## socket.h (module 'network'): ns3::Socket [class]
    module.add_class('Socket', parent=root_module['ns3::Object'])
    ## socket.h (module 'network'): ns3::Socket::SocketErrno [enumeration]
    module.add_enum('SocketErrno', ['ERROR_NOTERROR', 'ERROR_ISCONN', 'ERROR_NOTCONN', 'ERROR_MSGSIZE', 'ERROR_AGAIN', 'ERROR_SHUTDOWN', 'ERROR_OPNOTSUPP', 'ERROR_AFNOSUPPORT', 'ERROR_INVAL', 'ERROR_BADF', 'ERROR_NOROUTETOHOST', 'ERROR_NODEV', 'ERROR_ADDRNOTAVAIL', 'ERROR_ADDRINUSE', 'SOCKET_ERRNO_LAST'], outer_class=root_module['ns3::Socket'])
    ## socket.h (module 'network'): ns3::Socket::SocketType [enumeration]
    module.add_enum('SocketType', ['NS3_SOCK_STREAM', 'NS3_SOCK_SEQPACKET', 'NS3_SOCK_DGRAM', 'NS3_SOCK_RAW'], outer_class=root_module['ns3::Socket'])
    ## socket.h (module 'network'): ns3::Socket::SocketPriority [enumeration]
    module.add_enum('SocketPriority', ['NS3_PRIO_BESTEFFORT', 'NS3_PRIO_FILLER', 'NS3_PRIO_BULK', 'NS3_PRIO_INTERACTIVE_BULK', 'NS3_PRIO_INTERACTIVE', 'NS3_PRIO_CONTROL'], outer_class=root_module['ns3::Socket'])
    ## socket.h (module 'network'): ns3::Socket::Ipv6MulticastFilterMode [enumeration]
    module.add_enum('Ipv6MulticastFilterMode', ['INCLUDE', 'EXCLUDE'], outer_class=root_module['ns3::Socket'])
    ## socket-factory.h (module 'network'): ns3::SocketFactory [class]
    module.add_class('SocketFactory', parent=root_module['ns3::Object'])
    ## socket.h (module 'network'): ns3::SocketIpTosTag [class]
    module.add_class('SocketIpTosTag', parent=root_module['ns3::Tag'])
    ## socket.h (module 'network'): ns3::SocketIpTtlTag [class]
    module.add_class('SocketIpTtlTag', parent=root_module['ns3::Tag'])
    ## socket.h (module 'network'): ns3::SocketIpv6HopLimitTag [class]
    module.add_class('SocketIpv6HopLimitTag', parent=root_module['ns3::Tag'])
    ## socket.h (module 'network'): ns3::SocketIpv6TclassTag [class]
    module.add_class('SocketIpv6TclassTag', parent=root_module['ns3::Tag'])
    ## socket.h (module 'network'): ns3::SocketPriorityTag [class]
    module.add_class('SocketPriorityTag', parent=root_module['ns3::Tag'])
    ## socket.h (module 'network'): ns3::SocketSetDontFragmentTag [class]
    module.add_class('SocketSetDontFragmentTag', parent=root_module['ns3::Tag'])
    ## trace-source-accessor.h (module 'core'): ns3::TraceSourceAccessor [class]
    module.add_class('TraceSourceAccessor', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor> >'])
    ## trailer.h (module 'network'): ns3::Trailer [class]
    module.add_class('Trailer', parent=root_module['ns3::Chunk'])
    ## random-variable-stream.h (module 'core'): ns3::TriangularRandomVariable [class]
    module.add_class('TriangularRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## random-variable-stream.h (module 'core'): ns3::UniformRandomVariable [class]
    module.add_class('UniformRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## random-variable-stream.h (module 'core'): ns3::WeibullRandomVariable [class]
    module.add_class('WeibullRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## random-variable-stream.h (module 'core'): ns3::ZetaRandomVariable [class]
    module.add_class('ZetaRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## random-variable-stream.h (module 'core'): ns3::ZipfRandomVariable [class]
    module.add_class('ZipfRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## application.h (module 'network'): ns3::Application [class]
    module.add_class('Application', parent=root_module['ns3::Object'])
    typehandlers.add_type_alias('void ( * ) ( ns3::Time const &, ns3::Address const & )', 'ns3::Application::DelayAddressCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time const &, ns3::Address const & )*', 'ns3::Application::DelayAddressCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time const &, ns3::Address const & )&', 'ns3::Application::DelayAddressCallback&')
    typehandlers.add_type_alias('void ( * ) ( std::string const &, std::string const & )', 'ns3::Application::StateTransitionCallback')
    typehandlers.add_type_alias('void ( * ) ( std::string const &, std::string const & )*', 'ns3::Application::StateTransitionCallback*')
    typehandlers.add_type_alias('void ( * ) ( std::string const &, std::string const & )&', 'ns3::Application::StateTransitionCallback&')
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
    module.add_class('Channel', parent=root_module['ns3::Object'])
    ## random-variable-stream.h (module 'core'): ns3::ConstantRandomVariable [class]
    module.add_class('ConstantRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## data-calculator.h (module 'stats'): ns3::DataCalculator [class]
    module.add_class('DataCalculator', import_from_module='ns.stats', parent=root_module['ns3::Object'])
    ## data-collection-object.h (module 'stats'): ns3::DataCollectionObject [class]
    module.add_class('DataCollectionObject', import_from_module='ns.stats', parent=root_module['ns3::Object'])
    ## data-output-interface.h (module 'stats'): ns3::DataOutputInterface [class]
    module.add_class('DataOutputInterface', import_from_module='ns.stats', parent=root_module['ns3::Object'])
    ## data-rate.h (module 'network'): ns3::DataRateChecker [class]
    module.add_class('DataRateChecker', parent=root_module['ns3::AttributeChecker'])
    ## data-rate.h (module 'network'): ns3::DataRateValue [class]
    module.add_class('DataRateValue', parent=root_module['ns3::AttributeValue'])
    ## random-variable-stream.h (module 'core'): ns3::DeterministicRandomVariable [class]
    module.add_class('DeterministicRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## double.h (module 'core'): ns3::DoubleValue [class]
    module.add_class('DoubleValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## dynamic-queue-limits.h (module 'network'): ns3::DynamicQueueLimits [class]
    module.add_class('DynamicQueueLimits', parent=root_module['ns3::QueueLimits'])
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
    ## error-model.h (module 'network'): ns3::ErrorModel [class]
    module.add_class('ErrorModel', parent=root_module['ns3::Object'])
    ## ethernet-header.h (module 'network'): ns3::EthernetHeader [class]
    module.add_class('EthernetHeader', parent=root_module['ns3::Header'])
    ## ethernet-trailer.h (module 'network'): ns3::EthernetTrailer [class]
    module.add_class('EthernetTrailer', parent=root_module['ns3::Trailer'])
    ## event-impl.h (module 'core'): ns3::EventImpl [class]
    module.add_class('EventImpl', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl> >'])
    ## random-variable-stream.h (module 'core'): ns3::ExponentialRandomVariable [class]
    module.add_class('ExponentialRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## random-variable-stream.h (module 'core'): ns3::GammaRandomVariable [class]
    module.add_class('GammaRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## integer.h (module 'core'): ns3::IntegerValue [class]
    module.add_class('IntegerValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressChecker [class]
    module.add_class('Ipv4AddressChecker', parent=root_module['ns3::AttributeChecker'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4AddressValue [class]
    module.add_class('Ipv4AddressValue', parent=root_module['ns3::AttributeValue'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4MaskChecker [class]
    module.add_class('Ipv4MaskChecker', parent=root_module['ns3::AttributeChecker'])
    ## ipv4-address.h (module 'network'): ns3::Ipv4MaskValue [class]
    module.add_class('Ipv4MaskValue', parent=root_module['ns3::AttributeValue'])
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressChecker [class]
    module.add_class('Ipv6AddressChecker', parent=root_module['ns3::AttributeChecker'])
    ## ipv6-address.h (module 'network'): ns3::Ipv6AddressValue [class]
    module.add_class('Ipv6AddressValue', parent=root_module['ns3::AttributeValue'])
    ## ipv6-address.h (module 'network'): ns3::Ipv6PrefixChecker [class]
    module.add_class('Ipv6PrefixChecker', parent=root_module['ns3::AttributeChecker'])
    ## ipv6-address.h (module 'network'): ns3::Ipv6PrefixValue [class]
    module.add_class('Ipv6PrefixValue', parent=root_module['ns3::AttributeValue'])
    ## error-model.h (module 'network'): ns3::ListErrorModel [class]
    module.add_class('ListErrorModel', parent=root_module['ns3::ErrorModel'])
    ## random-variable-stream.h (module 'core'): ns3::LogNormalRandomVariable [class]
    module.add_class('LogNormalRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## mac16-address.h (module 'network'): ns3::Mac16AddressChecker [class]
    module.add_class('Mac16AddressChecker', parent=root_module['ns3::AttributeChecker'])
    ## mac16-address.h (module 'network'): ns3::Mac16AddressValue [class]
    module.add_class('Mac16AddressValue', parent=root_module['ns3::AttributeValue'])
    ## mac48-address.h (module 'network'): ns3::Mac48AddressChecker [class]
    module.add_class('Mac48AddressChecker', parent=root_module['ns3::AttributeChecker'])
    ## mac48-address.h (module 'network'): ns3::Mac48AddressValue [class]
    module.add_class('Mac48AddressValue', parent=root_module['ns3::AttributeValue'])
    ## mac64-address.h (module 'network'): ns3::Mac64AddressChecker [class]
    module.add_class('Mac64AddressChecker', parent=root_module['ns3::AttributeChecker'])
    ## mac64-address.h (module 'network'): ns3::Mac64AddressValue [class]
    module.add_class('Mac64AddressValue', parent=root_module['ns3::AttributeValue'])
    ## basic-data-calculators.h (module 'stats'): ns3::MinMaxAvgTotalCalculator<unsigned int> [class]
    module.add_class('MinMaxAvgTotalCalculator', import_from_module='ns.stats', parent=[root_module['ns3::DataCalculator'], root_module['ns3::StatisticalSummary']], template_parameters=['unsigned int'])
    ## net-device.h (module 'network'): ns3::NetDevice [class]
    module.add_class('NetDevice', parent=root_module['ns3::Object'])
    ## net-device.h (module 'network'): ns3::NetDevice::PacketType [enumeration]
    module.add_enum('PacketType', ['PACKET_HOST', 'NS3_PACKET_HOST', 'PACKET_BROADCAST', 'NS3_PACKET_BROADCAST', 'PACKET_MULTICAST', 'NS3_PACKET_MULTICAST', 'PACKET_OTHERHOST', 'NS3_PACKET_OTHERHOST'], outer_class=root_module['ns3::NetDevice'])
    typehandlers.add_type_alias('void ( * ) (  )', 'ns3::NetDevice::LinkChangeTracedCallback')
    typehandlers.add_type_alias('void ( * ) (  )*', 'ns3::NetDevice::LinkChangeTracedCallback*')
    typehandlers.add_type_alias('void ( * ) (  )&', 'ns3::NetDevice::LinkChangeTracedCallback&')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::NetDevice::ReceiveCallback')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::NetDevice::ReceiveCallback*')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::NetDevice::ReceiveCallback&')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >', 'ns3::NetDevice::PromiscReceiveCallback')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::NetDevice::PromiscReceiveCallback*')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::NetDevice::PromiscReceiveCallback&')
    ## net-device-queue-interface.h (module 'network'): ns3::NetDeviceQueue [class]
    module.add_class('NetDeviceQueue', parent=root_module['ns3::Object'])
    typehandlers.add_type_alias('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::NetDeviceQueue::WakeCallback')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::NetDeviceQueue::WakeCallback*')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::NetDeviceQueue::WakeCallback&')
    ## net-device-queue-interface.h (module 'network'): ns3::NetDeviceQueueInterface [class]
    module.add_class('NetDeviceQueueInterface', parent=root_module['ns3::Object'])
    typehandlers.add_type_alias('std::function< unsigned long long ( ns3::Ptr< ns3::QueueItem > ) >', 'ns3::NetDeviceQueueInterface::SelectQueueCallback')
    typehandlers.add_type_alias('std::function< unsigned long long ( ns3::Ptr< ns3::QueueItem > ) >*', 'ns3::NetDeviceQueueInterface::SelectQueueCallback*')
    typehandlers.add_type_alias('std::function< unsigned long long ( ns3::Ptr< ns3::QueueItem > ) >&', 'ns3::NetDeviceQueueInterface::SelectQueueCallback&')
    ## nix-vector.h (module 'network'): ns3::NixVector [class]
    module.add_class('NixVector', parent=root_module['ns3::SimpleRefCount< ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector> >'])
    ## node.h (module 'network'): ns3::Node [class]
    module.add_class('Node', parent=root_module['ns3::Object'])
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
    module.add_class('OutputStreamWrapper', parent=root_module['ns3::SimpleRefCount< ns3::OutputStreamWrapper, ns3::empty, ns3::DefaultDeleter<ns3::OutputStreamWrapper> >'])
    ## packet.h (module 'network'): ns3::Packet [class]
    module.add_class('Packet', parent=root_module['ns3::SimpleRefCount< ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet> >'])
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
    ## packet-data-calculators.h (module 'network'): ns3::PacketSizeMinMaxAvgTotalCalculator [class]
    module.add_class('PacketSizeMinMaxAvgTotalCalculator', parent=root_module['ns3::MinMaxAvgTotalCalculator< unsigned int >'])
    ## packet-socket.h (module 'network'): ns3::PacketSocket [class]
    module.add_class('PacketSocket', parent=root_module['ns3::Socket'])
    ## packet-socket-client.h (module 'network'): ns3::PacketSocketClient [class]
    module.add_class('PacketSocketClient', parent=root_module['ns3::Application'])
    ## packet-socket-factory.h (module 'network'): ns3::PacketSocketFactory [class]
    module.add_class('PacketSocketFactory', parent=root_module['ns3::SocketFactory'])
    ## packet-socket-server.h (module 'network'): ns3::PacketSocketServer [class]
    module.add_class('PacketSocketServer', parent=root_module['ns3::Application'])
    ## random-variable-stream.h (module 'core'): ns3::ParetoRandomVariable [class]
    module.add_class('ParetoRandomVariable', import_from_module='ns.core', parent=root_module['ns3::RandomVariableStream'])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock [class]
    module.add_class('PbbAddressBlock', parent=root_module['ns3::SimpleRefCount< ns3::PbbAddressBlock, ns3::empty, ns3::DefaultDeleter<ns3::PbbAddressBlock> >'])
    typehandlers.add_type_alias('std::list< ns3::Address > iterator', 'ns3::PbbAddressBlock::AddressIterator')
    typehandlers.add_type_alias('std::list< ns3::Address > iterator*', 'ns3::PbbAddressBlock::AddressIterator*')
    typehandlers.add_type_alias('std::list< ns3::Address > iterator&', 'ns3::PbbAddressBlock::AddressIterator&')
    typehandlers.add_type_alias('std::list< ns3::Address > const_iterator', 'ns3::PbbAddressBlock::ConstAddressIterator')
    typehandlers.add_type_alias('std::list< ns3::Address > const_iterator*', 'ns3::PbbAddressBlock::ConstAddressIterator*')
    typehandlers.add_type_alias('std::list< ns3::Address > const_iterator&', 'ns3::PbbAddressBlock::ConstAddressIterator&')
    typehandlers.add_type_alias('std::list< unsigned char > iterator', 'ns3::PbbAddressBlock::PrefixIterator')
    typehandlers.add_type_alias('std::list< unsigned char > iterator*', 'ns3::PbbAddressBlock::PrefixIterator*')
    typehandlers.add_type_alias('std::list< unsigned char > iterator&', 'ns3::PbbAddressBlock::PrefixIterator&')
    typehandlers.add_type_alias('std::list< unsigned char > const_iterator', 'ns3::PbbAddressBlock::ConstPrefixIterator')
    typehandlers.add_type_alias('std::list< unsigned char > const_iterator*', 'ns3::PbbAddressBlock::ConstPrefixIterator*')
    typehandlers.add_type_alias('std::list< unsigned char > const_iterator&', 'ns3::PbbAddressBlock::ConstPrefixIterator&')
    typehandlers.add_type_alias('ns3::PbbAddressTlvBlock::Iterator', 'ns3::PbbAddressBlock::TlvIterator')
    typehandlers.add_type_alias('ns3::PbbAddressTlvBlock::Iterator*', 'ns3::PbbAddressBlock::TlvIterator*')
    typehandlers.add_type_alias('ns3::PbbAddressTlvBlock::Iterator&', 'ns3::PbbAddressBlock::TlvIterator&')
    typehandlers.add_type_alias('ns3::PbbAddressTlvBlock::ConstIterator', 'ns3::PbbAddressBlock::ConstTlvIterator')
    typehandlers.add_type_alias('ns3::PbbAddressTlvBlock::ConstIterator*', 'ns3::PbbAddressBlock::ConstTlvIterator*')
    typehandlers.add_type_alias('ns3::PbbAddressTlvBlock::ConstIterator&', 'ns3::PbbAddressBlock::ConstTlvIterator&')
    ## packetbb.h (module 'network'): ns3::PbbAddressBlockIpv4 [class]
    module.add_class('PbbAddressBlockIpv4', parent=root_module['ns3::PbbAddressBlock'])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlockIpv6 [class]
    module.add_class('PbbAddressBlockIpv6', parent=root_module['ns3::PbbAddressBlock'])
    ## packetbb.h (module 'network'): ns3::PbbMessage [class]
    module.add_class('PbbMessage', parent=root_module['ns3::SimpleRefCount< ns3::PbbMessage, ns3::empty, ns3::DefaultDeleter<ns3::PbbMessage> >'])
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'ns3::PbbMessage::TlvIterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > iterator*', 'ns3::PbbMessage::TlvIterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > iterator&', 'ns3::PbbMessage::TlvIterator&')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > const_iterator', 'ns3::PbbMessage::ConstTlvIterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > const_iterator*', 'ns3::PbbMessage::ConstTlvIterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > const_iterator&', 'ns3::PbbMessage::ConstTlvIterator&')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressBlock > > iterator', 'ns3::PbbMessage::AddressBlockIterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressBlock > > iterator*', 'ns3::PbbMessage::AddressBlockIterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressBlock > > iterator&', 'ns3::PbbMessage::AddressBlockIterator&')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressBlock > > const_iterator', 'ns3::PbbMessage::ConstAddressBlockIterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressBlock > > const_iterator*', 'ns3::PbbMessage::ConstAddressBlockIterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbAddressBlock > > const_iterator&', 'ns3::PbbMessage::ConstAddressBlockIterator&')
    ## packetbb.h (module 'network'): ns3::PbbMessageIpv4 [class]
    module.add_class('PbbMessageIpv4', parent=root_module['ns3::PbbMessage'])
    ## packetbb.h (module 'network'): ns3::PbbMessageIpv6 [class]
    module.add_class('PbbMessageIpv6', parent=root_module['ns3::PbbMessage'])
    ## packetbb.h (module 'network'): ns3::PbbPacket [class]
    module.add_class('PbbPacket', parent=root_module['ns3::SimpleRefCount< ns3::PbbPacket, ns3::Header, ns3::DefaultDeleter<ns3::PbbPacket> >'])
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'ns3::PbbPacket::TlvIterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > iterator*', 'ns3::PbbPacket::TlvIterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > iterator&', 'ns3::PbbPacket::TlvIterator&')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > const_iterator', 'ns3::PbbPacket::ConstTlvIterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > const_iterator*', 'ns3::PbbPacket::ConstTlvIterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbTlv > > const_iterator&', 'ns3::PbbPacket::ConstTlvIterator&')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbMessage > > iterator', 'ns3::PbbPacket::MessageIterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbMessage > > iterator*', 'ns3::PbbPacket::MessageIterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbMessage > > iterator&', 'ns3::PbbPacket::MessageIterator&')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbMessage > > const_iterator', 'ns3::PbbPacket::ConstMessageIterator')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbMessage > > const_iterator*', 'ns3::PbbPacket::ConstMessageIterator*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::PbbMessage > > const_iterator&', 'ns3::PbbPacket::ConstMessageIterator&')
    ## packetbb.h (module 'network'): ns3::PbbTlv [class]
    module.add_class('PbbTlv', parent=root_module['ns3::SimpleRefCount< ns3::PbbTlv, ns3::empty, ns3::DefaultDeleter<ns3::PbbTlv> >'])
    ## probe.h (module 'stats'): ns3::Probe [class]
    module.add_class('Probe', import_from_module='ns.stats', parent=root_module['ns3::DataCollectionObject'])
    ## queue.h (module 'network'): ns3::Queue<ns3::Packet> [class]
    module.add_class('Queue', parent=root_module['ns3::QueueBase'], template_parameters=['ns3::Packet'])
    typehandlers.add_type_alias('ns3::Packet', 'ns3::Queue< ns3::Packet > ItemType')
    typehandlers.add_type_alias('ns3::Packet*', 'ns3::Queue< ns3::Packet > ItemType*')
    typehandlers.add_type_alias('ns3::Packet&', 'ns3::Queue< ns3::Packet > ItemType&')
    module.add_typedef(root_module['ns3::Packet'], 'ItemType')
    ## queue.h (module 'network'): ns3::Queue<ns3::QueueDiscItem> [class]
    module.add_class('Queue', parent=root_module['ns3::QueueBase'], template_parameters=['ns3::QueueDiscItem'])
    typehandlers.add_type_alias('ns3::QueueDiscItem', 'ns3::Queue< ns3::QueueDiscItem > ItemType')
    typehandlers.add_type_alias('ns3::QueueDiscItem*', 'ns3::Queue< ns3::QueueDiscItem > ItemType*')
    typehandlers.add_type_alias('ns3::QueueDiscItem&', 'ns3::Queue< ns3::QueueDiscItem > ItemType&')
    ## queue-item.h (module 'network'): ns3::QueueItem [class]
    module.add_class('QueueItem', parent=root_module['ns3::SimpleRefCount< ns3::QueueItem, ns3::empty, ns3::DefaultDeleter<ns3::QueueItem> >'])
    ## queue-item.h (module 'network'): ns3::QueueItem::Uint8Values [enumeration]
    module.add_enum('Uint8Values', ['IP_DSFIELD'], outer_class=root_module['ns3::QueueItem'])
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::QueueItem const > )', 'ns3::QueueItem::TracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::QueueItem const > )*', 'ns3::QueueItem::TracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Ptr< ns3::QueueItem const > )&', 'ns3::QueueItem::TracedCallback&')
    ## queue-size.h (module 'network'): ns3::QueueSizeChecker [class]
    module.add_class('QueueSizeChecker', parent=root_module['ns3::AttributeChecker'])
    ## queue-size.h (module 'network'): ns3::QueueSizeValue [class]
    module.add_class('QueueSizeValue', parent=root_module['ns3::AttributeValue'])
    ## error-model.h (module 'network'): ns3::RateErrorModel [class]
    module.add_class('RateErrorModel', parent=root_module['ns3::ErrorModel'])
    ## error-model.h (module 'network'): ns3::RateErrorModel::ErrorUnit [enumeration]
    module.add_enum('ErrorUnit', ['ERROR_UNIT_BIT', 'ERROR_UNIT_BYTE', 'ERROR_UNIT_PACKET'], outer_class=root_module['ns3::RateErrorModel'])
    ## error-model.h (module 'network'): ns3::ReceiveListErrorModel [class]
    module.add_class('ReceiveListErrorModel', parent=root_module['ns3::ErrorModel'])
    ## simple-channel.h (module 'network'): ns3::SimpleChannel [class]
    module.add_class('SimpleChannel', parent=root_module['ns3::Channel'])
    ## simple-net-device.h (module 'network'): ns3::SimpleNetDevice [class]
    module.add_class('SimpleNetDevice', parent=root_module['ns3::NetDevice'])
    ## nstime.h (module 'core'): ns3::TimeValue [class]
    module.add_class('TimeValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## type-id.h (module 'core'): ns3::TypeIdChecker [class]
    module.add_class('TypeIdChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## type-id.h (module 'core'): ns3::TypeIdValue [class]
    module.add_class('TypeIdValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## uinteger.h (module 'core'): ns3::UintegerValue [class]
    module.add_class('UintegerValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## address.h (module 'network'): ns3::AddressChecker [class]
    module.add_class('AddressChecker', parent=root_module['ns3::AttributeChecker'])
    ## address.h (module 'network'): ns3::AddressValue [class]
    module.add_class('AddressValue', parent=root_module['ns3::AttributeValue'])
    ## error-model.h (module 'network'): ns3::BinaryErrorModel [class]
    module.add_class('BinaryErrorModel', parent=root_module['ns3::ErrorModel'])
    ## error-model.h (module 'network'): ns3::BurstErrorModel [class]
    module.add_class('BurstErrorModel', parent=root_module['ns3::ErrorModel'])
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', parent=root_module['ns3::CallbackImplBase'], template_parameters=['bool', 'ns3::Ptr<ns3::NetDevice>', 'ns3::Ptr<const ns3::Packet>', 'unsigned short', 'const ns3::Address &', 'const ns3::Address &', 'ns3::NetDevice::PacketType', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['bool', 'ns3::Ptr<ns3::NetDevice>', 'ns3::Ptr<const ns3::Packet>', 'unsigned short', 'const ns3::Address &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['bool', 'ns3::Ptr<ns3::Socket>', 'const ns3::Address &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['ns3::ObjectBase *', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<const ns3::Packet>', 'const ns3::Address &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<const ns3::Packet>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::QueueDiscItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<const ns3::QueueDiscItem>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<ns3::NetDevice>', 'ns3::Ptr<const ns3::Packet>', 'unsigned short', 'const ns3::Address &', 'const ns3::Address &', 'ns3::NetDevice::PacketType', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<ns3::NetDevice>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<ns3::Socket>', 'const ns3::Address &', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<ns3::Socket>', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::Ptr<ns3::Socket>', 'unsigned int', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'unsigned int', 'unsigned int', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## basic-data-calculators.h (module 'stats'): ns3::CounterCalculator<unsigned int> [class]
    module.add_class('CounterCalculator', import_from_module='ns.stats', parent=root_module['ns3::DataCalculator'], template_parameters=['unsigned int'])
    ## drop-tail-queue.h (module 'network'): ns3::DropTailQueue<ns3::Packet> [class]
    module.add_class('DropTailQueue', parent=root_module['ns3::Queue< ns3::Packet >'], template_parameters=['ns3::Packet'])
    ## drop-tail-queue.h (module 'network'): ns3::DropTailQueue<ns3::QueueDiscItem> [class]
    module.add_class('DropTailQueue', parent=root_module['ns3::Queue< ns3::QueueDiscItem >'], template_parameters=['ns3::QueueDiscItem'])
    ## error-channel.h (module 'network'): ns3::ErrorChannel [class]
    module.add_class('ErrorChannel', parent=root_module['ns3::SimpleChannel'])
    ## packet-data-calculators.h (module 'network'): ns3::PacketCounterCalculator [class]
    module.add_class('PacketCounterCalculator', parent=root_module['ns3::CounterCalculator< unsigned int >'])
    ## packet-probe.h (module 'network'): ns3::PacketProbe [class]
    module.add_class('PacketProbe', parent=root_module['ns3::Probe'])
    ## packetbb.h (module 'network'): ns3::PbbAddressTlv [class]
    module.add_class('PbbAddressTlv', parent=root_module['ns3::PbbTlv'])
    ## queue-item.h (module 'network'): ns3::QueueDiscItem [class]
    module.add_class('QueueDiscItem', parent=root_module['ns3::QueueItem'])
    module.add_container('std::vector< unsigned char >', 'unsigned char', container_type='vector')
    module.add_container('std::map< std::string, ns3::LogComponent * >', ('std::basic_string< char, std::char_traits< char >, std::allocator< char > >', 'ns3::LogComponent *'), container_type='map')
    module.add_container('std::list< ns3::Ptr< ns3::Packet > >', 'ns3::Ptr< ns3::Packet >', container_type='list')
    module.add_container('std::vector< ns3::Ipv6Address >', 'ns3::Ipv6Address', container_type='vector')
    module.add_container('std::list< unsigned int >', 'unsigned int', container_type='list')
    typehandlers.add_type_alias('ns3::SequenceNumber< unsigned int, int >', 'ns3::SequenceNumber32')
    typehandlers.add_type_alias('ns3::SequenceNumber< unsigned int, int >*', 'ns3::SequenceNumber32*')
    typehandlers.add_type_alias('ns3::SequenceNumber< unsigned int, int >&', 'ns3::SequenceNumber32&')
    typehandlers.add_type_alias('ns3::SequenceNumber< unsigned short, short >', 'ns3::SequenceNumber16')
    typehandlers.add_type_alias('ns3::SequenceNumber< unsigned short, short >*', 'ns3::SequenceNumber16*')
    typehandlers.add_type_alias('ns3::SequenceNumber< unsigned short, short >&', 'ns3::SequenceNumber16&')
    typehandlers.add_type_alias('ns3::SequenceNumber< unsigned char, signed char >', 'ns3::SequenceNumber8')
    typehandlers.add_type_alias('ns3::SequenceNumber< unsigned char, signed char >*', 'ns3::SequenceNumber8*')
    typehandlers.add_type_alias('ns3::SequenceNumber< unsigned char, signed char >&', 'ns3::SequenceNumber8&')
    typehandlers.add_type_alias('ns3::LollipopCounter< unsigned char >', 'ns3::LollipopCounter8')
    typehandlers.add_type_alias('ns3::LollipopCounter< unsigned char >*', 'ns3::LollipopCounter8*')
    typehandlers.add_type_alias('ns3::LollipopCounter< unsigned char >&', 'ns3::LollipopCounter8&')
    typehandlers.add_type_alias('ns3::LollipopCounter< unsigned short >', 'ns3::LollipopCounter16')
    typehandlers.add_type_alias('ns3::LollipopCounter< unsigned short >*', 'ns3::LollipopCounter16*')
    typehandlers.add_type_alias('ns3::LollipopCounter< unsigned short >&', 'ns3::LollipopCounter16&')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::Packet >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::GenericPhyTxStartCallback')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::Packet >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::GenericPhyTxStartCallback*')
    typehandlers.add_type_alias('ns3::Callback< bool, ns3::Ptr< ns3::Packet >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::GenericPhyTxStartCallback&')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::Packet const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::GenericPhyTxEndCallback')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::Packet const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::GenericPhyTxEndCallback*')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::Packet const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::GenericPhyTxEndCallback&')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::GenericPhyRxStartCallback')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::GenericPhyRxStartCallback*')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::GenericPhyRxStartCallback&')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::GenericPhyRxEndErrorCallback')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::GenericPhyRxEndErrorCallback*')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::GenericPhyRxEndErrorCallback&')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::Packet >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'ns3::GenericPhyRxEndOkCallback')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::Packet >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >*', 'ns3::GenericPhyRxEndOkCallback*')
    typehandlers.add_type_alias('ns3::Callback< void, ns3::Ptr< ns3::Packet >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >&', 'ns3::GenericPhyRxEndOkCallback&')
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
    
    
    ## Register a nested module for the namespace addressUtils
    
    nested_module = module.add_cpp_namespace('addressUtils')
    register_types_ns3_addressUtils(nested_module)
    
    
    ## Register a nested module for the namespace internal
    
    nested_module = module.add_cpp_namespace('internal')
    register_types_ns3_internal(nested_module)
    
    
    ## Register a nested module for the namespace tests
    
    nested_module = module.add_cpp_namespace('tests')
    register_types_ns3_tests(nested_module)
    

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
    typehandlers.add_type_alias('void ( * ) ( ns3::SequenceNumber32, ns3::SequenceNumber32 )', 'ns3::TracedValueCallback::SequenceNumber32')
    typehandlers.add_type_alias('void ( * ) ( ns3::SequenceNumber32, ns3::SequenceNumber32 )*', 'ns3::TracedValueCallback::SequenceNumber32*')
    typehandlers.add_type_alias('void ( * ) ( ns3::SequenceNumber32, ns3::SequenceNumber32 )&', 'ns3::TracedValueCallback::SequenceNumber32&')
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
    typehandlers.add_type_alias('void ( * ) ( ns3::DataRate, ns3::DataRate )', 'ns3::TracedValueCallback::DataRate')
    typehandlers.add_type_alias('void ( * ) ( ns3::DataRate, ns3::DataRate )*', 'ns3::TracedValueCallback::DataRate*')
    typehandlers.add_type_alias('void ( * ) ( ns3::DataRate, ns3::DataRate )&', 'ns3::TracedValueCallback::DataRate&')

def register_types_ns3_addressUtils(module):
    root_module = module.get_root()
    

def register_types_ns3_internal(module):
    root_module = module.get_root()
    

def register_types_ns3_tests(module):
    root_module = module.get_root()
    

def register_methods(root_module):
    register_Ns3Address_methods(root_module, root_module['ns3::Address'])
    register_Ns3ApplicationContainer_methods(root_module, root_module['ns3::ApplicationContainer'])
    register_Ns3AsciiTraceHelper_methods(root_module, root_module['ns3::AsciiTraceHelper'])
    register_Ns3AsciiTraceHelperForDevice_methods(root_module, root_module['ns3::AsciiTraceHelperForDevice'])
    register_Ns3AttributeConstructionList_methods(root_module, root_module['ns3::AttributeConstructionList'])
    register_Ns3AttributeConstructionListItem_methods(root_module, root_module['ns3::AttributeConstructionList::Item'])
    register_Ns3BitDeserializer_methods(root_module, root_module['ns3::BitDeserializer'])
    register_Ns3BitSerializer_methods(root_module, root_module['ns3::BitSerializer'])
    register_Ns3Buffer_methods(root_module, root_module['ns3::Buffer'])
    register_Ns3BufferIterator_methods(root_module, root_module['ns3::Buffer::Iterator'])
    register_Ns3ByteTagIterator_methods(root_module, root_module['ns3::ByteTagIterator'])
    register_Ns3ByteTagIteratorItem_methods(root_module, root_module['ns3::ByteTagIterator::Item'])
    register_Ns3ByteTagList_methods(root_module, root_module['ns3::ByteTagList'])
    register_Ns3ByteTagListIterator_methods(root_module, root_module['ns3::ByteTagList::Iterator'])
    register_Ns3ByteTagListIteratorItem_methods(root_module, root_module['ns3::ByteTagList::Iterator::Item'])
    register_Ns3CallbackBase_methods(root_module, root_module['ns3::CallbackBase'])
    register_Ns3ChannelList_methods(root_module, root_module['ns3::ChannelList'])
    register_Ns3DataOutputCallback_methods(root_module, root_module['ns3::DataOutputCallback'])
    register_Ns3DataRate_methods(root_module, root_module['ns3::DataRate'])
    register_Ns3DefaultDeleter__Ns3AttributeAccessor_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeAccessor >'])
    register_Ns3DefaultDeleter__Ns3AttributeChecker_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeChecker >'])
    register_Ns3DefaultDeleter__Ns3AttributeValue_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeValue >'])
    register_Ns3DefaultDeleter__Ns3CallbackImplBase_methods(root_module, root_module['ns3::DefaultDeleter< ns3::CallbackImplBase >'])
    register_Ns3DefaultDeleter__Ns3EventImpl_methods(root_module, root_module['ns3::DefaultDeleter< ns3::EventImpl >'])
    register_Ns3DefaultDeleter__Ns3HashImplementation_methods(root_module, root_module['ns3::DefaultDeleter< ns3::Hash::Implementation >'])
    register_Ns3DefaultDeleter__Ns3NixVector_methods(root_module, root_module['ns3::DefaultDeleter< ns3::NixVector >'])
    register_Ns3DefaultDeleter__Ns3OutputStreamWrapper_methods(root_module, root_module['ns3::DefaultDeleter< ns3::OutputStreamWrapper >'])
    register_Ns3DefaultDeleter__Ns3Packet_methods(root_module, root_module['ns3::DefaultDeleter< ns3::Packet >'])
    register_Ns3DefaultDeleter__Ns3PbbAddressBlock_methods(root_module, root_module['ns3::DefaultDeleter< ns3::PbbAddressBlock >'])
    register_Ns3DefaultDeleter__Ns3PbbMessage_methods(root_module, root_module['ns3::DefaultDeleter< ns3::PbbMessage >'])
    register_Ns3DefaultDeleter__Ns3PbbTlv_methods(root_module, root_module['ns3::DefaultDeleter< ns3::PbbTlv >'])
    register_Ns3DefaultDeleter__Ns3QueueItem_methods(root_module, root_module['ns3::DefaultDeleter< ns3::QueueItem >'])
    register_Ns3DefaultDeleter__Ns3TraceSourceAccessor_methods(root_module, root_module['ns3::DefaultDeleter< ns3::TraceSourceAccessor >'])
    register_Ns3DelayJitterEstimation_methods(root_module, root_module['ns3::DelayJitterEstimation'])
    register_Ns3EventId_methods(root_module, root_module['ns3::EventId'])
    register_Ns3Hasher_methods(root_module, root_module['ns3::Hasher'])
    register_Ns3Inet6SocketAddress_methods(root_module, root_module['ns3::Inet6SocketAddress'])
    register_Ns3InetSocketAddress_methods(root_module, root_module['ns3::InetSocketAddress'])
    register_Ns3Ipv4Address_methods(root_module, root_module['ns3::Ipv4Address'])
    register_Ns3Ipv4AddressHash_methods(root_module, root_module['ns3::Ipv4AddressHash'])
    register_Ns3Ipv4Mask_methods(root_module, root_module['ns3::Ipv4Mask'])
    register_Ns3Ipv6Address_methods(root_module, root_module['ns3::Ipv6Address'])
    register_Ns3Ipv6AddressHash_methods(root_module, root_module['ns3::Ipv6AddressHash'])
    register_Ns3Ipv6Prefix_methods(root_module, root_module['ns3::Ipv6Prefix'])
    register_Ns3LogComponent_methods(root_module, root_module['ns3::LogComponent'])
    register_Ns3Mac16Address_methods(root_module, root_module['ns3::Mac16Address'])
    register_Ns3Mac48Address_methods(root_module, root_module['ns3::Mac48Address'])
    register_Ns3Mac64Address_methods(root_module, root_module['ns3::Mac64Address'])
    register_Ns3Mac8Address_methods(root_module, root_module['ns3::Mac8Address'])
    register_Ns3NetDeviceContainer_methods(root_module, root_module['ns3::NetDeviceContainer'])
    register_Ns3NodeContainer_methods(root_module, root_module['ns3::NodeContainer'])
    register_Ns3NodeList_methods(root_module, root_module['ns3::NodeList'])
    register_Ns3ObjectBase_methods(root_module, root_module['ns3::ObjectBase'])
    register_Ns3ObjectDeleter_methods(root_module, root_module['ns3::ObjectDeleter'])
    register_Ns3ObjectFactory_methods(root_module, root_module['ns3::ObjectFactory'])
    register_Ns3PacketMetadata_methods(root_module, root_module['ns3::PacketMetadata'])
    register_Ns3PacketMetadataItem_methods(root_module, root_module['ns3::PacketMetadata::Item'])
    register_Ns3PacketMetadataItemIterator_methods(root_module, root_module['ns3::PacketMetadata::ItemIterator'])
    register_Ns3PacketSocketAddress_methods(root_module, root_module['ns3::PacketSocketAddress'])
    register_Ns3PacketSocketHelper_methods(root_module, root_module['ns3::PacketSocketHelper'])
    register_Ns3PacketTagIterator_methods(root_module, root_module['ns3::PacketTagIterator'])
    register_Ns3PacketTagIteratorItem_methods(root_module, root_module['ns3::PacketTagIterator::Item'])
    register_Ns3PacketTagList_methods(root_module, root_module['ns3::PacketTagList'])
    register_Ns3PacketTagListTagData_methods(root_module, root_module['ns3::PacketTagList::TagData'])
    register_Ns3ParameterLogger_methods(root_module, root_module['ns3::ParameterLogger'])
    register_Ns3PbbAddressTlvBlock_methods(root_module, root_module['ns3::PbbAddressTlvBlock'])
    register_Ns3PbbTlvBlock_methods(root_module, root_module['ns3::PbbTlvBlock'])
    register_Ns3PcapFile_methods(root_module, root_module['ns3::PcapFile'])
    register_Ns3PcapHelper_methods(root_module, root_module['ns3::PcapHelper'])
    register_Ns3PcapHelperForDevice_methods(root_module, root_module['ns3::PcapHelperForDevice'])
    register_Ns3QueueSize_methods(root_module, root_module['ns3::QueueSize'])
    register_Ns3SequenceNumber32_methods(root_module, root_module['ns3::SequenceNumber32'])
    register_Ns3SequenceNumber16_methods(root_module, root_module['ns3::SequenceNumber16'])
    register_Ns3SimpleNetDeviceHelper_methods(root_module, root_module['ns3::SimpleNetDeviceHelper'])
    register_Ns3SimpleRefCount__Ns3Object_Ns3ObjectBase_Ns3ObjectDeleter_methods(root_module, root_module['ns3::SimpleRefCount< ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter >'])
    register_Ns3Simulator_methods(root_module, root_module['ns3::Simulator'])
    register_Ns3StatisticalSummary_methods(root_module, root_module['ns3::StatisticalSummary'])
    register_Ns3SystemWallClockMs_methods(root_module, root_module['ns3::SystemWallClockMs'])
    register_Ns3Tag_methods(root_module, root_module['ns3::Tag'])
    register_Ns3TagBuffer_methods(root_module, root_module['ns3::TagBuffer'])
    register_Ns3Time_methods(root_module, root_module['ns3::Time'])
    register_Ns3TimeWithUnit_methods(root_module, root_module['ns3::TimeWithUnit'])
    register_Ns3TracedValue__Unsigned_int_methods(root_module, root_module['ns3::TracedValue< unsigned int >'])
    register_Ns3TypeId_methods(root_module, root_module['ns3::TypeId'])
    register_Ns3TypeIdAttributeInformation_methods(root_module, root_module['ns3::TypeId::AttributeInformation'])
    register_Ns3TypeIdTraceSourceInformation_methods(root_module, root_module['ns3::TypeId::TraceSourceInformation'])
    register_Ns3Empty_methods(root_module, root_module['ns3::empty'])
    register_Ns3Int64x64_t_methods(root_module, root_module['ns3::int64x64_t'])
    register_Ns3Chunk_methods(root_module, root_module['ns3::Chunk'])
    register_Ns3DeviceNameTag_methods(root_module, root_module['ns3::DeviceNameTag'])
    register_Ns3FlowIdTag_methods(root_module, root_module['ns3::FlowIdTag'])
    register_Ns3Header_methods(root_module, root_module['ns3::Header'])
    register_Ns3LlcSnapHeader_methods(root_module, root_module['ns3::LlcSnapHeader'])
    register_Ns3Object_methods(root_module, root_module['ns3::Object'])
    register_Ns3ObjectAggregateIterator_methods(root_module, root_module['ns3::Object::AggregateIterator'])
    register_Ns3PacketBurst_methods(root_module, root_module['ns3::PacketBurst'])
    register_Ns3PacketSocketTag_methods(root_module, root_module['ns3::PacketSocketTag'])
    register_Ns3PcapFileWrapper_methods(root_module, root_module['ns3::PcapFileWrapper'])
    register_Ns3QueueBase_methods(root_module, root_module['ns3::QueueBase'])
    register_Ns3QueueLimits_methods(root_module, root_module['ns3::QueueLimits'])
    register_Ns3RadiotapHeader_methods(root_module, root_module['ns3::RadiotapHeader'])
    register_Ns3RandomVariableStream_methods(root_module, root_module['ns3::RandomVariableStream'])
    register_Ns3SequentialRandomVariable_methods(root_module, root_module['ns3::SequentialRandomVariable'])
    register_Ns3SimpleRefCount__Ns3AttributeAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeAccessor__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeAccessor, ns3::empty, ns3::DefaultDeleter<ns3::AttributeAccessor> >'])
    register_Ns3SimpleRefCount__Ns3AttributeChecker_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeChecker__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeChecker, ns3::empty, ns3::DefaultDeleter<ns3::AttributeChecker> >'])
    register_Ns3SimpleRefCount__Ns3AttributeValue_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeValue__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeValue, ns3::empty, ns3::DefaultDeleter<ns3::AttributeValue> >'])
    register_Ns3SimpleRefCount__Ns3CallbackImplBase_Ns3Empty_Ns3DefaultDeleter__lt__ns3CallbackImplBase__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter<ns3::CallbackImplBase> >'])
    register_Ns3SimpleRefCount__Ns3EventImpl_Ns3Empty_Ns3DefaultDeleter__lt__ns3EventImpl__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl> >'])
    register_Ns3SimpleRefCount__Ns3HashImplementation_Ns3Empty_Ns3DefaultDeleter__lt__ns3HashImplementation__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::Hash::Implementation, ns3::empty, ns3::DefaultDeleter<ns3::Hash::Implementation> >'])
    register_Ns3SimpleRefCount__Ns3NixVector_Ns3Empty_Ns3DefaultDeleter__lt__ns3NixVector__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::NixVector, ns3::empty, ns3::DefaultDeleter<ns3::NixVector> >'])
    register_Ns3SimpleRefCount__Ns3OutputStreamWrapper_Ns3Empty_Ns3DefaultDeleter__lt__ns3OutputStreamWrapper__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::OutputStreamWrapper, ns3::empty, ns3::DefaultDeleter<ns3::OutputStreamWrapper> >'])
    register_Ns3SimpleRefCount__Ns3Packet_Ns3Empty_Ns3DefaultDeleter__lt__ns3Packet__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::Packet, ns3::empty, ns3::DefaultDeleter<ns3::Packet> >'])
    register_Ns3SimpleRefCount__Ns3PbbAddressBlock_Ns3Empty_Ns3DefaultDeleter__lt__ns3PbbAddressBlock__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::PbbAddressBlock, ns3::empty, ns3::DefaultDeleter<ns3::PbbAddressBlock> >'])
    register_Ns3SimpleRefCount__Ns3PbbMessage_Ns3Empty_Ns3DefaultDeleter__lt__ns3PbbMessage__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::PbbMessage, ns3::empty, ns3::DefaultDeleter<ns3::PbbMessage> >'])
    register_Ns3SimpleRefCount__Ns3PbbPacket_Ns3Header_Ns3DefaultDeleter__lt__ns3PbbPacket__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::PbbPacket, ns3::Header, ns3::DefaultDeleter<ns3::PbbPacket> >'])
    register_Ns3SimpleRefCount__Ns3PbbTlv_Ns3Empty_Ns3DefaultDeleter__lt__ns3PbbTlv__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::PbbTlv, ns3::empty, ns3::DefaultDeleter<ns3::PbbTlv> >'])
    register_Ns3SimpleRefCount__Ns3QueueItem_Ns3Empty_Ns3DefaultDeleter__lt__ns3QueueItem__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::QueueItem, ns3::empty, ns3::DefaultDeleter<ns3::QueueItem> >'])
    register_Ns3SimpleRefCount__Ns3TraceSourceAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3TraceSourceAccessor__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor> >'])
    register_Ns3SllHeader_methods(root_module, root_module['ns3::SllHeader'])
    register_Ns3Socket_methods(root_module, root_module['ns3::Socket'])
    register_Ns3SocketFactory_methods(root_module, root_module['ns3::SocketFactory'])
    register_Ns3SocketIpTosTag_methods(root_module, root_module['ns3::SocketIpTosTag'])
    register_Ns3SocketIpTtlTag_methods(root_module, root_module['ns3::SocketIpTtlTag'])
    register_Ns3SocketIpv6HopLimitTag_methods(root_module, root_module['ns3::SocketIpv6HopLimitTag'])
    register_Ns3SocketIpv6TclassTag_methods(root_module, root_module['ns3::SocketIpv6TclassTag'])
    register_Ns3SocketPriorityTag_methods(root_module, root_module['ns3::SocketPriorityTag'])
    register_Ns3SocketSetDontFragmentTag_methods(root_module, root_module['ns3::SocketSetDontFragmentTag'])
    register_Ns3TraceSourceAccessor_methods(root_module, root_module['ns3::TraceSourceAccessor'])
    register_Ns3Trailer_methods(root_module, root_module['ns3::Trailer'])
    register_Ns3TriangularRandomVariable_methods(root_module, root_module['ns3::TriangularRandomVariable'])
    register_Ns3UniformRandomVariable_methods(root_module, root_module['ns3::UniformRandomVariable'])
    register_Ns3WeibullRandomVariable_methods(root_module, root_module['ns3::WeibullRandomVariable'])
    register_Ns3ZetaRandomVariable_methods(root_module, root_module['ns3::ZetaRandomVariable'])
    register_Ns3ZipfRandomVariable_methods(root_module, root_module['ns3::ZipfRandomVariable'])
    register_Ns3Application_methods(root_module, root_module['ns3::Application'])
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
    register_Ns3DataCalculator_methods(root_module, root_module['ns3::DataCalculator'])
    register_Ns3DataCollectionObject_methods(root_module, root_module['ns3::DataCollectionObject'])
    register_Ns3DataOutputInterface_methods(root_module, root_module['ns3::DataOutputInterface'])
    register_Ns3DataRateChecker_methods(root_module, root_module['ns3::DataRateChecker'])
    register_Ns3DataRateValue_methods(root_module, root_module['ns3::DataRateValue'])
    register_Ns3DeterministicRandomVariable_methods(root_module, root_module['ns3::DeterministicRandomVariable'])
    register_Ns3DoubleValue_methods(root_module, root_module['ns3::DoubleValue'])
    register_Ns3DynamicQueueLimits_methods(root_module, root_module['ns3::DynamicQueueLimits'])
    register_Ns3EmpiricalRandomVariable_methods(root_module, root_module['ns3::EmpiricalRandomVariable'])
    register_Ns3EmptyAttributeAccessor_methods(root_module, root_module['ns3::EmptyAttributeAccessor'])
    register_Ns3EmptyAttributeChecker_methods(root_module, root_module['ns3::EmptyAttributeChecker'])
    register_Ns3EmptyAttributeValue_methods(root_module, root_module['ns3::EmptyAttributeValue'])
    register_Ns3EnumChecker_methods(root_module, root_module['ns3::EnumChecker'])
    register_Ns3EnumValue_methods(root_module, root_module['ns3::EnumValue'])
    register_Ns3ErlangRandomVariable_methods(root_module, root_module['ns3::ErlangRandomVariable'])
    register_Ns3ErrorModel_methods(root_module, root_module['ns3::ErrorModel'])
    register_Ns3EthernetHeader_methods(root_module, root_module['ns3::EthernetHeader'])
    register_Ns3EthernetTrailer_methods(root_module, root_module['ns3::EthernetTrailer'])
    register_Ns3EventImpl_methods(root_module, root_module['ns3::EventImpl'])
    register_Ns3ExponentialRandomVariable_methods(root_module, root_module['ns3::ExponentialRandomVariable'])
    register_Ns3GammaRandomVariable_methods(root_module, root_module['ns3::GammaRandomVariable'])
    register_Ns3IntegerValue_methods(root_module, root_module['ns3::IntegerValue'])
    register_Ns3Ipv4AddressChecker_methods(root_module, root_module['ns3::Ipv4AddressChecker'])
    register_Ns3Ipv4AddressValue_methods(root_module, root_module['ns3::Ipv4AddressValue'])
    register_Ns3Ipv4MaskChecker_methods(root_module, root_module['ns3::Ipv4MaskChecker'])
    register_Ns3Ipv4MaskValue_methods(root_module, root_module['ns3::Ipv4MaskValue'])
    register_Ns3Ipv6AddressChecker_methods(root_module, root_module['ns3::Ipv6AddressChecker'])
    register_Ns3Ipv6AddressValue_methods(root_module, root_module['ns3::Ipv6AddressValue'])
    register_Ns3Ipv6PrefixChecker_methods(root_module, root_module['ns3::Ipv6PrefixChecker'])
    register_Ns3Ipv6PrefixValue_methods(root_module, root_module['ns3::Ipv6PrefixValue'])
    register_Ns3ListErrorModel_methods(root_module, root_module['ns3::ListErrorModel'])
    register_Ns3LogNormalRandomVariable_methods(root_module, root_module['ns3::LogNormalRandomVariable'])
    register_Ns3Mac16AddressChecker_methods(root_module, root_module['ns3::Mac16AddressChecker'])
    register_Ns3Mac16AddressValue_methods(root_module, root_module['ns3::Mac16AddressValue'])
    register_Ns3Mac48AddressChecker_methods(root_module, root_module['ns3::Mac48AddressChecker'])
    register_Ns3Mac48AddressValue_methods(root_module, root_module['ns3::Mac48AddressValue'])
    register_Ns3Mac64AddressChecker_methods(root_module, root_module['ns3::Mac64AddressChecker'])
    register_Ns3Mac64AddressValue_methods(root_module, root_module['ns3::Mac64AddressValue'])
    register_Ns3MinMaxAvgTotalCalculator__Unsigned_int_methods(root_module, root_module['ns3::MinMaxAvgTotalCalculator< unsigned int >'])
    register_Ns3NetDevice_methods(root_module, root_module['ns3::NetDevice'])
    register_Ns3NetDeviceQueue_methods(root_module, root_module['ns3::NetDeviceQueue'])
    register_Ns3NetDeviceQueueInterface_methods(root_module, root_module['ns3::NetDeviceQueueInterface'])
    register_Ns3NixVector_methods(root_module, root_module['ns3::NixVector'])
    register_Ns3Node_methods(root_module, root_module['ns3::Node'])
    register_Ns3NormalRandomVariable_methods(root_module, root_module['ns3::NormalRandomVariable'])
    register_Ns3ObjectFactoryChecker_methods(root_module, root_module['ns3::ObjectFactoryChecker'])
    register_Ns3ObjectFactoryValue_methods(root_module, root_module['ns3::ObjectFactoryValue'])
    register_Ns3OutputStreamWrapper_methods(root_module, root_module['ns3::OutputStreamWrapper'])
    register_Ns3Packet_methods(root_module, root_module['ns3::Packet'])
    register_Ns3PacketSizeMinMaxAvgTotalCalculator_methods(root_module, root_module['ns3::PacketSizeMinMaxAvgTotalCalculator'])
    register_Ns3PacketSocket_methods(root_module, root_module['ns3::PacketSocket'])
    register_Ns3PacketSocketClient_methods(root_module, root_module['ns3::PacketSocketClient'])
    register_Ns3PacketSocketFactory_methods(root_module, root_module['ns3::PacketSocketFactory'])
    register_Ns3PacketSocketServer_methods(root_module, root_module['ns3::PacketSocketServer'])
    register_Ns3ParetoRandomVariable_methods(root_module, root_module['ns3::ParetoRandomVariable'])
    register_Ns3PbbAddressBlock_methods(root_module, root_module['ns3::PbbAddressBlock'])
    register_Ns3PbbAddressBlockIpv4_methods(root_module, root_module['ns3::PbbAddressBlockIpv4'])
    register_Ns3PbbAddressBlockIpv6_methods(root_module, root_module['ns3::PbbAddressBlockIpv6'])
    register_Ns3PbbMessage_methods(root_module, root_module['ns3::PbbMessage'])
    register_Ns3PbbMessageIpv4_methods(root_module, root_module['ns3::PbbMessageIpv4'])
    register_Ns3PbbMessageIpv6_methods(root_module, root_module['ns3::PbbMessageIpv6'])
    register_Ns3PbbPacket_methods(root_module, root_module['ns3::PbbPacket'])
    register_Ns3PbbTlv_methods(root_module, root_module['ns3::PbbTlv'])
    register_Ns3Probe_methods(root_module, root_module['ns3::Probe'])
    register_Ns3Queue__Ns3Packet_methods(root_module, root_module['ns3::Queue< ns3::Packet >'])
    register_Ns3Queue__Ns3QueueDiscItem_methods(root_module, root_module['ns3::Queue< ns3::QueueDiscItem >'])
    register_Ns3QueueItem_methods(root_module, root_module['ns3::QueueItem'])
    register_Ns3QueueSizeChecker_methods(root_module, root_module['ns3::QueueSizeChecker'])
    register_Ns3QueueSizeValue_methods(root_module, root_module['ns3::QueueSizeValue'])
    register_Ns3RateErrorModel_methods(root_module, root_module['ns3::RateErrorModel'])
    register_Ns3ReceiveListErrorModel_methods(root_module, root_module['ns3::ReceiveListErrorModel'])
    register_Ns3SimpleChannel_methods(root_module, root_module['ns3::SimpleChannel'])
    register_Ns3SimpleNetDevice_methods(root_module, root_module['ns3::SimpleNetDevice'])
    register_Ns3TimeValue_methods(root_module, root_module['ns3::TimeValue'])
    register_Ns3TypeIdChecker_methods(root_module, root_module['ns3::TypeIdChecker'])
    register_Ns3TypeIdValue_methods(root_module, root_module['ns3::TypeIdValue'])
    register_Ns3UintegerValue_methods(root_module, root_module['ns3::UintegerValue'])
    register_Ns3AddressChecker_methods(root_module, root_module['ns3::AddressChecker'])
    register_Ns3AddressValue_methods(root_module, root_module['ns3::AddressValue'])
    register_Ns3BinaryErrorModel_methods(root_module, root_module['ns3::BinaryErrorModel'])
    register_Ns3BurstErrorModel_methods(root_module, root_module['ns3::BurstErrorModel'])
    register_Ns3CallbackImpl__Bool_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Ptr__lt__const_ns3Packet__gt___Unsigned_short_Const_ns3Address___amp___Const_ns3Address___amp___Ns3NetDevicePacketType_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Bool_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Ptr__lt__const_ns3Packet__gt___Unsigned_short_Const_ns3Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< bool, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Bool_Ns3Ptr__lt__ns3Socket__gt___Const_ns3Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< bool, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Ns3ObjectBase___star___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Const_ns3Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<const ns3::Packet>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<const ns3::Packet>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3QueueDiscItem__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<const ns3::QueueDiscItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Ptr__lt__const_ns3Packet__gt___Unsigned_short_Const_ns3Address___amp___Const_ns3Address___amp___Ns3NetDevicePacketType_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet>, unsigned short, const ns3::Address &, const ns3::Address &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3NetDevice__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<ns3::NetDevice>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3Socket__gt___Const_ns3Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3Socket__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3Socket__gt___Unsigned_int_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::Ptr<ns3::Socket>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Unsigned_int_Unsigned_int_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CounterCalculator__Unsigned_int_methods(root_module, root_module['ns3::CounterCalculator< unsigned int >'])
    register_Ns3DropTailQueue__Ns3Packet_methods(root_module, root_module['ns3::DropTailQueue< ns3::Packet >'])
    register_Ns3DropTailQueue__Ns3QueueDiscItem_methods(root_module, root_module['ns3::DropTailQueue< ns3::QueueDiscItem >'])
    register_Ns3ErrorChannel_methods(root_module, root_module['ns3::ErrorChannel'])
    register_Ns3PacketCounterCalculator_methods(root_module, root_module['ns3::PacketCounterCalculator'])
    register_Ns3PacketProbe_methods(root_module, root_module['ns3::PacketProbe'])
    register_Ns3PbbAddressTlv_methods(root_module, root_module['ns3::PbbAddressTlv'])
    register_Ns3QueueDiscItem_methods(root_module, root_module['ns3::QueueDiscItem'])
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

def register_Ns3ApplicationContainer_methods(root_module, cls):
    ## application-container.h (module 'network'): ns3::ApplicationContainer::ApplicationContainer(ns3::ApplicationContainer const & arg0) [constructor]
    cls.add_constructor([param('ns3::ApplicationContainer const &', 'arg0')])
    ## application-container.h (module 'network'): ns3::ApplicationContainer::ApplicationContainer() [constructor]
    cls.add_constructor([])
    ## application-container.h (module 'network'): ns3::ApplicationContainer::ApplicationContainer(ns3::Ptr<ns3::Application> application) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::Application >', 'application')])
    ## application-container.h (module 'network'): ns3::ApplicationContainer::ApplicationContainer(std::string name) [constructor]
    cls.add_constructor([param('std::string', 'name')])
    ## application-container.h (module 'network'): void ns3::ApplicationContainer::Add(ns3::ApplicationContainer other) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::ApplicationContainer', 'other')])
    ## application-container.h (module 'network'): void ns3::ApplicationContainer::Add(ns3::Ptr<ns3::Application> application) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::Ptr< ns3::Application >', 'application')])
    ## application-container.h (module 'network'): void ns3::ApplicationContainer::Add(std::string name) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('std::string', 'name')])
    ## application-container.h (module 'network'): ns3::ApplicationContainer::Iterator ns3::ApplicationContainer::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::ApplicationContainer::Iterator', 
                   [], 
                   is_const=True)
    ## application-container.h (module 'network'): ns3::ApplicationContainer::Iterator ns3::ApplicationContainer::End() const [member function]
    cls.add_method('End', 
                   'ns3::ApplicationContainer::Iterator', 
                   [], 
                   is_const=True)
    ## application-container.h (module 'network'): ns3::Ptr<ns3::Application> ns3::ApplicationContainer::Get(uint32_t i) const [member function]
    cls.add_method('Get', 
                   'ns3::Ptr< ns3::Application >', 
                   [param('uint32_t', 'i')], 
                   is_const=True)
    ## application-container.h (module 'network'): uint32_t ns3::ApplicationContainer::GetN() const [member function]
    cls.add_method('GetN', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## application-container.h (module 'network'): void ns3::ApplicationContainer::Start(ns3::Time start) [member function]
    cls.add_method('Start', 
                   'void', 
                   [param('ns3::Time', 'start')])
    ## application-container.h (module 'network'): void ns3::ApplicationContainer::StartWithJitter(ns3::Time start, ns3::Ptr<ns3::RandomVariableStream> rv) [member function]
    cls.add_method('StartWithJitter', 
                   'void', 
                   [param('ns3::Time', 'start'), param('ns3::Ptr< ns3::RandomVariableStream >', 'rv')])
    ## application-container.h (module 'network'): void ns3::ApplicationContainer::Stop(ns3::Time stop) [member function]
    cls.add_method('Stop', 
                   'void', 
                   [param('ns3::Time', 'stop')])
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

def register_Ns3BitDeserializer_methods(root_module, cls):
    ## bit-deserializer.h (module 'network'): ns3::BitDeserializer::BitDeserializer(ns3::BitDeserializer const & arg0) [constructor]
    cls.add_constructor([param('ns3::BitDeserializer const &', 'arg0')])
    ## bit-deserializer.h (module 'network'): ns3::BitDeserializer::BitDeserializer() [constructor]
    cls.add_constructor([])
    ## bit-deserializer.h (module 'network'): uint64_t ns3::BitDeserializer::GetBits(uint8_t size) [member function]
    cls.add_method('GetBits', 
                   'uint64_t', 
                   [param('uint8_t', 'size')])
    ## bit-deserializer.h (module 'network'): void ns3::BitDeserializer::PushByte(uint8_t byte) [member function]
    cls.add_method('PushByte', 
                   'void', 
                   [param('uint8_t', 'byte')])
    ## bit-deserializer.h (module 'network'): void ns3::BitDeserializer::PushBytes(std::vector<unsigned char, std::allocator<unsigned char>> bytes) [member function]
    cls.add_method('PushBytes', 
                   'void', 
                   [param('std::vector< unsigned char >', 'bytes')])
    ## bit-deserializer.h (module 'network'): void ns3::BitDeserializer::PushBytes(uint8_t * bytes, uint32_t size) [member function]
    cls.add_method('PushBytes', 
                   'void', 
                   [param('uint8_t *', 'bytes'), param('uint32_t', 'size')])
    return

def register_Ns3BitSerializer_methods(root_module, cls):
    ## bit-serializer.h (module 'network'): ns3::BitSerializer::BitSerializer(ns3::BitSerializer const & arg0) [constructor]
    cls.add_constructor([param('ns3::BitSerializer const &', 'arg0')])
    ## bit-serializer.h (module 'network'): ns3::BitSerializer::BitSerializer() [constructor]
    cls.add_constructor([])
    ## bit-serializer.h (module 'network'): std::vector<unsigned char, std::allocator<unsigned char>> ns3::BitSerializer::GetBytes() [member function]
    cls.add_method('GetBytes', 
                   'std::vector< unsigned char >', 
                   [])
    ## bit-serializer.h (module 'network'): uint8_t ns3::BitSerializer::GetBytes(uint8_t * buffer, uint32_t size) [member function]
    cls.add_method('GetBytes', 
                   'uint8_t', 
                   [param('uint8_t *', 'buffer'), param('uint32_t', 'size')])
    ## bit-serializer.h (module 'network'): void ns3::BitSerializer::InsertPaddingAtEnd(bool padAtEnd) [member function]
    cls.add_method('InsertPaddingAtEnd', 
                   'void', 
                   [param('bool', 'padAtEnd')])
    ## bit-serializer.h (module 'network'): void ns3::BitSerializer::PushBits(uint64_t value, uint8_t significantBits) [member function]
    cls.add_method('PushBits', 
                   'void', 
                   [param('uint64_t', 'value'), param('uint8_t', 'significantBits')])
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

def register_Ns3ChannelList_methods(root_module, cls):
    ## channel-list.h (module 'network'): ns3::ChannelList::ChannelList() [constructor]
    cls.add_constructor([])
    ## channel-list.h (module 'network'): ns3::ChannelList::ChannelList(ns3::ChannelList const & arg0) [constructor]
    cls.add_constructor([param('ns3::ChannelList const &', 'arg0')])
    ## channel-list.h (module 'network'): static uint32_t ns3::ChannelList::Add(ns3::Ptr<ns3::Channel> channel) [member function]
    cls.add_method('Add', 
                   'uint32_t', 
                   [param('ns3::Ptr< ns3::Channel >', 'channel')], 
                   is_static=True)
    ## channel-list.h (module 'network'): static ns3::ChannelList::Iterator ns3::ChannelList::Begin() [member function]
    cls.add_method('Begin', 
                   'ns3::ChannelList::Iterator', 
                   [], 
                   is_static=True)
    ## channel-list.h (module 'network'): static ns3::ChannelList::Iterator ns3::ChannelList::End() [member function]
    cls.add_method('End', 
                   'ns3::ChannelList::Iterator', 
                   [], 
                   is_static=True)
    ## channel-list.h (module 'network'): static ns3::Ptr<ns3::Channel> ns3::ChannelList::GetChannel(uint32_t n) [member function]
    cls.add_method('GetChannel', 
                   'ns3::Ptr< ns3::Channel >', 
                   [param('uint32_t', 'n')], 
                   is_static=True)
    ## channel-list.h (module 'network'): static uint32_t ns3::ChannelList::GetNChannels() [member function]
    cls.add_method('GetNChannels', 
                   'uint32_t', 
                   [], 
                   is_static=True)
    return

def register_Ns3DataOutputCallback_methods(root_module, cls):
    ## data-output-interface.h (module 'stats'): ns3::DataOutputCallback::DataOutputCallback() [constructor]
    cls.add_constructor([])
    ## data-output-interface.h (module 'stats'): ns3::DataOutputCallback::DataOutputCallback(ns3::DataOutputCallback const & arg0) [constructor]
    cls.add_constructor([param('ns3::DataOutputCallback const &', 'arg0')])
    ## data-output-interface.h (module 'stats'): void ns3::DataOutputCallback::OutputSingleton(std::string key, std::string variable, int val) [member function]
    cls.add_method('OutputSingleton', 
                   'void', 
                   [param('std::string', 'key'), param('std::string', 'variable'), param('int', 'val')], 
                   is_pure_virtual=True, is_virtual=True)
    ## data-output-interface.h (module 'stats'): void ns3::DataOutputCallback::OutputSingleton(std::string key, std::string variable, uint32_t val) [member function]
    cls.add_method('OutputSingleton', 
                   'void', 
                   [param('std::string', 'key'), param('std::string', 'variable'), param('uint32_t', 'val')], 
                   is_pure_virtual=True, is_virtual=True)
    ## data-output-interface.h (module 'stats'): void ns3::DataOutputCallback::OutputSingleton(std::string key, std::string variable, double val) [member function]
    cls.add_method('OutputSingleton', 
                   'void', 
                   [param('std::string', 'key'), param('std::string', 'variable'), param('double', 'val')], 
                   is_pure_virtual=True, is_virtual=True)
    ## data-output-interface.h (module 'stats'): void ns3::DataOutputCallback::OutputSingleton(std::string key, std::string variable, std::string val) [member function]
    cls.add_method('OutputSingleton', 
                   'void', 
                   [param('std::string', 'key'), param('std::string', 'variable'), param('std::string', 'val')], 
                   is_pure_virtual=True, is_virtual=True)
    ## data-output-interface.h (module 'stats'): void ns3::DataOutputCallback::OutputSingleton(std::string key, std::string variable, ns3::Time val) [member function]
    cls.add_method('OutputSingleton', 
                   'void', 
                   [param('std::string', 'key'), param('std::string', 'variable'), param('ns3::Time', 'val')], 
                   is_pure_virtual=True, is_virtual=True)
    ## data-output-interface.h (module 'stats'): void ns3::DataOutputCallback::OutputStatistic(std::string key, std::string variable, ns3::StatisticalSummary const * statSum) [member function]
    cls.add_method('OutputStatistic', 
                   'void', 
                   [param('std::string', 'key'), param('std::string', 'variable'), param('ns3::StatisticalSummary const *', 'statSum')], 
                   is_pure_virtual=True, is_virtual=True)
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

def register_Ns3DefaultDeleter__Ns3PbbAddressBlock_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::PbbAddressBlock>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::PbbAddressBlock>::DefaultDeleter(ns3::DefaultDeleter<ns3::PbbAddressBlock> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::PbbAddressBlock > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::PbbAddressBlock>::Delete(ns3::PbbAddressBlock * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::PbbAddressBlock *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3PbbMessage_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::PbbMessage>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::PbbMessage>::DefaultDeleter(ns3::DefaultDeleter<ns3::PbbMessage> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::PbbMessage > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::PbbMessage>::Delete(ns3::PbbMessage * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::PbbMessage *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3PbbTlv_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::PbbTlv>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::PbbTlv>::DefaultDeleter(ns3::DefaultDeleter<ns3::PbbTlv> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::PbbTlv > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::PbbTlv>::Delete(ns3::PbbTlv * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::PbbTlv *', 'object')], 
                   is_static=True)
    return

def register_Ns3DefaultDeleter__Ns3QueueItem_methods(root_module, cls):
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::QueueItem>::DefaultDeleter() [constructor]
    cls.add_constructor([])
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::QueueItem>::DefaultDeleter(ns3::DefaultDeleter<ns3::QueueItem> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DefaultDeleter< ns3::QueueItem > const &', 'arg0')])
    ## default-deleter.h (module 'core'): static void ns3::DefaultDeleter<ns3::QueueItem>::Delete(ns3::QueueItem * object) [member function]
    cls.add_method('Delete', 
                   'void', 
                   [param('ns3::QueueItem *', 'object')], 
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

def register_Ns3DelayJitterEstimation_methods(root_module, cls):
    ## delay-jitter-estimation.h (module 'network'): ns3::DelayJitterEstimation::DelayJitterEstimation(ns3::DelayJitterEstimation const & arg0) [constructor]
    cls.add_constructor([param('ns3::DelayJitterEstimation const &', 'arg0')])
    ## delay-jitter-estimation.h (module 'network'): ns3::DelayJitterEstimation::DelayJitterEstimation() [constructor]
    cls.add_constructor([])
    ## delay-jitter-estimation.h (module 'network'): ns3::Time ns3::DelayJitterEstimation::GetLastDelay() const [member function]
    cls.add_method('GetLastDelay', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## delay-jitter-estimation.h (module 'network'): uint64_t ns3::DelayJitterEstimation::GetLastJitter() const [member function]
    cls.add_method('GetLastJitter', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## delay-jitter-estimation.h (module 'network'): static void ns3::DelayJitterEstimation::PrepareTx(ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('PrepareTx', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet')], 
                   is_static=True)
    ## delay-jitter-estimation.h (module 'network'): void ns3::DelayJitterEstimation::RecordRx(ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('RecordRx', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet')])
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

def register_Ns3Inet6SocketAddress_methods(root_module, cls):
    ## inet6-socket-address.h (module 'network'): ns3::Inet6SocketAddress::Inet6SocketAddress(ns3::Inet6SocketAddress const & arg0) [constructor]
    cls.add_constructor([param('ns3::Inet6SocketAddress const &', 'arg0')])
    ## inet6-socket-address.h (module 'network'): ns3::Inet6SocketAddress::Inet6SocketAddress(ns3::Ipv6Address ipv6, uint16_t port) [constructor]
    cls.add_constructor([param('ns3::Ipv6Address', 'ipv6'), param('uint16_t', 'port')])
    ## inet6-socket-address.h (module 'network'): ns3::Inet6SocketAddress::Inet6SocketAddress(ns3::Ipv6Address ipv6) [constructor]
    cls.add_constructor([param('ns3::Ipv6Address', 'ipv6')])
    ## inet6-socket-address.h (module 'network'): ns3::Inet6SocketAddress::Inet6SocketAddress(uint16_t port) [constructor]
    cls.add_constructor([param('uint16_t', 'port')])
    ## inet6-socket-address.h (module 'network'): ns3::Inet6SocketAddress::Inet6SocketAddress(char const * ipv6, uint16_t port) [constructor]
    cls.add_constructor([param('char const *', 'ipv6'), param('uint16_t', 'port')])
    ## inet6-socket-address.h (module 'network'): ns3::Inet6SocketAddress::Inet6SocketAddress(char const * ipv6) [constructor]
    cls.add_constructor([param('char const *', 'ipv6')])
    ## inet6-socket-address.h (module 'network'): static ns3::Inet6SocketAddress ns3::Inet6SocketAddress::ConvertFrom(ns3::Address const & addr) [member function]
    cls.add_method('ConvertFrom', 
                   'ns3::Inet6SocketAddress', 
                   [param('ns3::Address const &', 'addr')], 
                   is_static=True)
    ## inet6-socket-address.h (module 'network'): ns3::Ipv6Address ns3::Inet6SocketAddress::GetIpv6() const [member function]
    cls.add_method('GetIpv6', 
                   'ns3::Ipv6Address', 
                   [], 
                   is_const=True)
    ## inet6-socket-address.h (module 'network'): uint16_t ns3::Inet6SocketAddress::GetPort() const [member function]
    cls.add_method('GetPort', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## inet6-socket-address.h (module 'network'): static bool ns3::Inet6SocketAddress::IsMatchingType(ns3::Address const & addr) [member function]
    cls.add_method('IsMatchingType', 
                   'bool', 
                   [param('ns3::Address const &', 'addr')], 
                   is_static=True)
    ## inet6-socket-address.h (module 'network'): void ns3::Inet6SocketAddress::SetIpv6(ns3::Ipv6Address ipv6) [member function]
    cls.add_method('SetIpv6', 
                   'void', 
                   [param('ns3::Ipv6Address', 'ipv6')])
    ## inet6-socket-address.h (module 'network'): void ns3::Inet6SocketAddress::SetPort(uint16_t port) [member function]
    cls.add_method('SetPort', 
                   'void', 
                   [param('uint16_t', 'port')])
    return

def register_Ns3InetSocketAddress_methods(root_module, cls):
    ## inet-socket-address.h (module 'network'): ns3::InetSocketAddress::InetSocketAddress(ns3::InetSocketAddress const & arg0) [constructor]
    cls.add_constructor([param('ns3::InetSocketAddress const &', 'arg0')])
    ## inet-socket-address.h (module 'network'): ns3::InetSocketAddress::InetSocketAddress(ns3::Ipv4Address ipv4, uint16_t port) [constructor]
    cls.add_constructor([param('ns3::Ipv4Address', 'ipv4'), param('uint16_t', 'port')])
    ## inet-socket-address.h (module 'network'): ns3::InetSocketAddress::InetSocketAddress(ns3::Ipv4Address ipv4) [constructor]
    cls.add_constructor([param('ns3::Ipv4Address', 'ipv4')])
    ## inet-socket-address.h (module 'network'): ns3::InetSocketAddress::InetSocketAddress(uint16_t port) [constructor]
    cls.add_constructor([param('uint16_t', 'port')])
    ## inet-socket-address.h (module 'network'): ns3::InetSocketAddress::InetSocketAddress(char const * ipv4, uint16_t port) [constructor]
    cls.add_constructor([param('char const *', 'ipv4'), param('uint16_t', 'port')])
    ## inet-socket-address.h (module 'network'): ns3::InetSocketAddress::InetSocketAddress(char const * ipv4) [constructor]
    cls.add_constructor([param('char const *', 'ipv4')])
    ## inet-socket-address.h (module 'network'): static ns3::InetSocketAddress ns3::InetSocketAddress::ConvertFrom(ns3::Address const & address) [member function]
    cls.add_method('ConvertFrom', 
                   'ns3::InetSocketAddress', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## inet-socket-address.h (module 'network'): ns3::Ipv4Address ns3::InetSocketAddress::GetIpv4() const [member function]
    cls.add_method('GetIpv4', 
                   'ns3::Ipv4Address', 
                   [], 
                   is_const=True)
    ## inet-socket-address.h (module 'network'): uint16_t ns3::InetSocketAddress::GetPort() const [member function]
    cls.add_method('GetPort', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## inet-socket-address.h (module 'network'): uint8_t ns3::InetSocketAddress::GetTos() const [member function]
    cls.add_method('GetTos', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## inet-socket-address.h (module 'network'): static bool ns3::InetSocketAddress::IsMatchingType(ns3::Address const & address) [member function]
    cls.add_method('IsMatchingType', 
                   'bool', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## inet-socket-address.h (module 'network'): void ns3::InetSocketAddress::SetIpv4(ns3::Ipv4Address address) [member function]
    cls.add_method('SetIpv4', 
                   'void', 
                   [param('ns3::Ipv4Address', 'address')])
    ## inet-socket-address.h (module 'network'): void ns3::InetSocketAddress::SetPort(uint16_t port) [member function]
    cls.add_method('SetPort', 
                   'void', 
                   [param('uint16_t', 'port')])
    ## inet-socket-address.h (module 'network'): void ns3::InetSocketAddress::SetTos(uint8_t tos) [member function]
    cls.add_method('SetTos', 
                   'void', 
                   [param('uint8_t', 'tos')])
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

def register_Ns3Mac16Address_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<')
    cls.add_output_stream_operator()
    ## mac16-address.h (module 'network'): ns3::Mac16Address::Mac16Address(ns3::Mac16Address const & arg0) [constructor]
    cls.add_constructor([param('ns3::Mac16Address const &', 'arg0')])
    ## mac16-address.h (module 'network'): ns3::Mac16Address::Mac16Address() [constructor]
    cls.add_constructor([])
    ## mac16-address.h (module 'network'): ns3::Mac16Address::Mac16Address(char const * str) [constructor]
    cls.add_constructor([param('char const *', 'str')])
    ## mac16-address.h (module 'network'): static ns3::Mac16Address ns3::Mac16Address::Allocate() [member function]
    cls.add_method('Allocate', 
                   'ns3::Mac16Address', 
                   [], 
                   is_static=True)
    ## mac16-address.h (module 'network'): static ns3::Mac16Address ns3::Mac16Address::ConvertFrom(ns3::Address const & address) [member function]
    cls.add_method('ConvertFrom', 
                   'ns3::Mac16Address', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## mac16-address.h (module 'network'): void ns3::Mac16Address::CopyFrom(uint8_t const * buffer) [member function]
    cls.add_method('CopyFrom', 
                   'void', 
                   [param('uint8_t const *', 'buffer')])
    ## mac16-address.h (module 'network'): void ns3::Mac16Address::CopyTo(uint8_t * buffer) const [member function]
    cls.add_method('CopyTo', 
                   'void', 
                   [param('uint8_t *', 'buffer')], 
                   is_const=True)
    ## mac16-address.h (module 'network'): static ns3::Mac16Address ns3::Mac16Address::GetBroadcast() [member function]
    cls.add_method('GetBroadcast', 
                   'ns3::Mac16Address', 
                   [], 
                   is_static=True)
    ## mac16-address.h (module 'network'): static ns3::Mac16Address ns3::Mac16Address::GetMulticast(ns3::Ipv6Address address) [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Mac16Address', 
                   [param('ns3::Ipv6Address', 'address')], 
                   is_static=True)
    ## mac16-address.h (module 'network'): bool ns3::Mac16Address::IsBroadcast() const [member function]
    cls.add_method('IsBroadcast', 
                   'bool', 
                   [], 
                   is_const=True)
    ## mac16-address.h (module 'network'): static bool ns3::Mac16Address::IsMatchingType(ns3::Address const & address) [member function]
    cls.add_method('IsMatchingType', 
                   'bool', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## mac16-address.h (module 'network'): bool ns3::Mac16Address::IsMulticast() const [member function]
    cls.add_method('IsMulticast', 
                   'bool', 
                   [], 
                   is_const=True)
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

def register_Ns3Mac64Address_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<')
    cls.add_output_stream_operator()
    ## mac64-address.h (module 'network'): ns3::Mac64Address::Mac64Address(ns3::Mac64Address const & arg0) [constructor]
    cls.add_constructor([param('ns3::Mac64Address const &', 'arg0')])
    ## mac64-address.h (module 'network'): ns3::Mac64Address::Mac64Address() [constructor]
    cls.add_constructor([])
    ## mac64-address.h (module 'network'): ns3::Mac64Address::Mac64Address(char const * str) [constructor]
    cls.add_constructor([param('char const *', 'str')])
    ## mac64-address.h (module 'network'): static ns3::Mac64Address ns3::Mac64Address::Allocate() [member function]
    cls.add_method('Allocate', 
                   'ns3::Mac64Address', 
                   [], 
                   is_static=True)
    ## mac64-address.h (module 'network'): static ns3::Mac64Address ns3::Mac64Address::ConvertFrom(ns3::Address const & address) [member function]
    cls.add_method('ConvertFrom', 
                   'ns3::Mac64Address', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## mac64-address.h (module 'network'): void ns3::Mac64Address::CopyFrom(uint8_t const * buffer) [member function]
    cls.add_method('CopyFrom', 
                   'void', 
                   [param('uint8_t const *', 'buffer')])
    ## mac64-address.h (module 'network'): void ns3::Mac64Address::CopyTo(uint8_t * buffer) const [member function]
    cls.add_method('CopyTo', 
                   'void', 
                   [param('uint8_t *', 'buffer')], 
                   is_const=True)
    ## mac64-address.h (module 'network'): static bool ns3::Mac64Address::IsMatchingType(ns3::Address const & address) [member function]
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

def register_Ns3NodeList_methods(root_module, cls):
    ## node-list.h (module 'network'): ns3::NodeList::NodeList() [constructor]
    cls.add_constructor([])
    ## node-list.h (module 'network'): ns3::NodeList::NodeList(ns3::NodeList const & arg0) [constructor]
    cls.add_constructor([param('ns3::NodeList const &', 'arg0')])
    ## node-list.h (module 'network'): static uint32_t ns3::NodeList::Add(ns3::Ptr<ns3::Node> node) [member function]
    cls.add_method('Add', 
                   'uint32_t', 
                   [param('ns3::Ptr< ns3::Node >', 'node')], 
                   is_static=True)
    ## node-list.h (module 'network'): static ns3::NodeList::Iterator ns3::NodeList::Begin() [member function]
    cls.add_method('Begin', 
                   'ns3::NodeList::Iterator', 
                   [], 
                   is_static=True)
    ## node-list.h (module 'network'): static ns3::NodeList::Iterator ns3::NodeList::End() [member function]
    cls.add_method('End', 
                   'ns3::NodeList::Iterator', 
                   [], 
                   is_static=True)
    ## node-list.h (module 'network'): static uint32_t ns3::NodeList::GetNNodes() [member function]
    cls.add_method('GetNNodes', 
                   'uint32_t', 
                   [], 
                   is_static=True)
    ## node-list.h (module 'network'): static ns3::Ptr<ns3::Node> ns3::NodeList::GetNode(uint32_t n) [member function]
    cls.add_method('GetNode', 
                   'ns3::Ptr< ns3::Node >', 
                   [param('uint32_t', 'n')], 
                   is_static=True)
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

def register_Ns3PacketSocketAddress_methods(root_module, cls):
    ## packet-socket-address.h (module 'network'): ns3::PacketSocketAddress::PacketSocketAddress(ns3::PacketSocketAddress const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketSocketAddress const &', 'arg0')])
    ## packet-socket-address.h (module 'network'): ns3::PacketSocketAddress::PacketSocketAddress() [constructor]
    cls.add_constructor([])
    ## packet-socket-address.h (module 'network'): static ns3::PacketSocketAddress ns3::PacketSocketAddress::ConvertFrom(ns3::Address const & address) [member function]
    cls.add_method('ConvertFrom', 
                   'ns3::PacketSocketAddress', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## packet-socket-address.h (module 'network'): ns3::Address ns3::PacketSocketAddress::GetPhysicalAddress() const [member function]
    cls.add_method('GetPhysicalAddress', 
                   'ns3::Address', 
                   [], 
                   is_const=True)
    ## packet-socket-address.h (module 'network'): uint16_t ns3::PacketSocketAddress::GetProtocol() const [member function]
    cls.add_method('GetProtocol', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## packet-socket-address.h (module 'network'): uint32_t ns3::PacketSocketAddress::GetSingleDevice() const [member function]
    cls.add_method('GetSingleDevice', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packet-socket-address.h (module 'network'): static bool ns3::PacketSocketAddress::IsMatchingType(ns3::Address const & address) [member function]
    cls.add_method('IsMatchingType', 
                   'bool', 
                   [param('ns3::Address const &', 'address')], 
                   is_static=True)
    ## packet-socket-address.h (module 'network'): bool ns3::PacketSocketAddress::IsSingleDevice() const [member function]
    cls.add_method('IsSingleDevice', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packet-socket-address.h (module 'network'): void ns3::PacketSocketAddress::SetAllDevices() [member function]
    cls.add_method('SetAllDevices', 
                   'void', 
                   [])
    ## packet-socket-address.h (module 'network'): void ns3::PacketSocketAddress::SetPhysicalAddress(ns3::Address const address) [member function]
    cls.add_method('SetPhysicalAddress', 
                   'void', 
                   [param('ns3::Address const', 'address')])
    ## packet-socket-address.h (module 'network'): void ns3::PacketSocketAddress::SetProtocol(uint16_t protocol) [member function]
    cls.add_method('SetProtocol', 
                   'void', 
                   [param('uint16_t', 'protocol')])
    ## packet-socket-address.h (module 'network'): void ns3::PacketSocketAddress::SetSingleDevice(uint32_t device) [member function]
    cls.add_method('SetSingleDevice', 
                   'void', 
                   [param('uint32_t', 'device')])
    return

def register_Ns3PacketSocketHelper_methods(root_module, cls):
    ## packet-socket-helper.h (module 'network'): ns3::PacketSocketHelper::PacketSocketHelper() [constructor]
    cls.add_constructor([])
    ## packet-socket-helper.h (module 'network'): ns3::PacketSocketHelper::PacketSocketHelper(ns3::PacketSocketHelper const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketSocketHelper const &', 'arg0')])
    ## packet-socket-helper.h (module 'network'): void ns3::PacketSocketHelper::Install(ns3::Ptr<ns3::Node> node) const [member function]
    cls.add_method('Install', 
                   'void', 
                   [param('ns3::Ptr< ns3::Node >', 'node')], 
                   is_const=True)
    ## packet-socket-helper.h (module 'network'): void ns3::PacketSocketHelper::Install(std::string nodeName) const [member function]
    cls.add_method('Install', 
                   'void', 
                   [param('std::string', 'nodeName')], 
                   is_const=True)
    ## packet-socket-helper.h (module 'network'): void ns3::PacketSocketHelper::Install(ns3::NodeContainer c) const [member function]
    cls.add_method('Install', 
                   'void', 
                   [param('ns3::NodeContainer', 'c')], 
                   is_const=True)
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

def register_Ns3PbbAddressTlvBlock_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    ## packetbb.h (module 'network'): ns3::PbbAddressTlvBlock::PbbAddressTlvBlock(ns3::PbbAddressTlvBlock const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbAddressTlvBlock const &', 'arg0')])
    ## packetbb.h (module 'network'): ns3::PbbAddressTlvBlock::PbbAddressTlvBlock() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressTlv> ns3::PbbAddressTlvBlock::Back() const [member function]
    cls.add_method('Back', 
                   'ns3::Ptr< ns3::PbbAddressTlv >', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressTlvBlock::Iterator ns3::PbbAddressTlvBlock::Begin() [member function]
    cls.add_method('Begin', 
                   'ns3::PbbAddressTlvBlock::Iterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbAddressTlvBlock::ConstIterator ns3::PbbAddressTlvBlock::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::PbbAddressTlvBlock::ConstIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlvBlock::Clear() [member function]
    cls.add_method('Clear', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlvBlock::Deserialize(ns3::Buffer::Iterator & start) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')])
    ## packetbb.h (module 'network'): bool ns3::PbbAddressTlvBlock::Empty() const [member function]
    cls.add_method('Empty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressTlvBlock::Iterator ns3::PbbAddressTlvBlock::End() [member function]
    cls.add_method('End', 
                   'ns3::PbbAddressTlvBlock::Iterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbAddressTlvBlock::ConstIterator ns3::PbbAddressTlvBlock::End() const [member function]
    cls.add_method('End', 
                   'ns3::PbbAddressTlvBlock::ConstIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressTlvBlock::Iterator ns3::PbbAddressTlvBlock::Erase(ns3::PbbAddressTlvBlock::Iterator position) [member function]
    cls.add_method('Erase', 
                   'ns3::PbbAddressTlvBlock::Iterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbAddressTlv > > iterator', 'position')])
    ## packetbb.h (module 'network'): ns3::PbbAddressTlvBlock::Iterator ns3::PbbAddressTlvBlock::Erase(ns3::PbbAddressTlvBlock::Iterator first, ns3::PbbAddressTlvBlock::Iterator last) [member function]
    cls.add_method('Erase', 
                   'ns3::PbbAddressTlvBlock::Iterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbAddressTlv > > iterator', 'first'), param('std::list< ns3::Ptr< ns3::PbbAddressTlv > > iterator', 'last')])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressTlv> ns3::PbbAddressTlvBlock::Front() const [member function]
    cls.add_method('Front', 
                   'ns3::Ptr< ns3::PbbAddressTlv >', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint32_t ns3::PbbAddressTlvBlock::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressTlvBlock::Iterator ns3::PbbAddressTlvBlock::Insert(ns3::PbbAddressTlvBlock::Iterator position, ns3::Ptr<ns3::PbbAddressTlv> const tlv) [member function]
    cls.add_method('Insert', 
                   'ns3::PbbAddressTlvBlock::Iterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbAddressTlv > > iterator', 'position'), param('ns3::Ptr< ns3::PbbAddressTlv > const', 'tlv')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlvBlock::PopBack() [member function]
    cls.add_method('PopBack', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlvBlock::PopFront() [member function]
    cls.add_method('PopFront', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlvBlock::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlvBlock::Print(std::ostream & os, int level) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os'), param('int', 'level')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlvBlock::PushBack(ns3::Ptr<ns3::PbbAddressTlv> tlv) [member function]
    cls.add_method('PushBack', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbAddressTlv >', 'tlv')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlvBlock::PushFront(ns3::Ptr<ns3::PbbAddressTlv> tlv) [member function]
    cls.add_method('PushFront', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbAddressTlv >', 'tlv')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlvBlock::Serialize(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True)
    ## packetbb.h (module 'network'): int ns3::PbbAddressTlvBlock::Size() const [member function]
    cls.add_method('Size', 
                   'int', 
                   [], 
                   is_const=True)
    return

def register_Ns3PbbTlvBlock_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    ## packetbb.h (module 'network'): ns3::PbbTlvBlock::PbbTlvBlock(ns3::PbbTlvBlock const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbTlvBlock const &', 'arg0')])
    ## packetbb.h (module 'network'): ns3::PbbTlvBlock::PbbTlvBlock() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbTlv> ns3::PbbTlvBlock::Back() const [member function]
    cls.add_method('Back', 
                   'ns3::Ptr< ns3::PbbTlv >', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbTlvBlock::Iterator ns3::PbbTlvBlock::Begin() [member function]
    cls.add_method('Begin', 
                   'ns3::PbbTlvBlock::Iterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbTlvBlock::ConstIterator ns3::PbbTlvBlock::Begin() const [member function]
    cls.add_method('Begin', 
                   'ns3::PbbTlvBlock::ConstIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbTlvBlock::Clear() [member function]
    cls.add_method('Clear', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbTlvBlock::Deserialize(ns3::Buffer::Iterator & start) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')])
    ## packetbb.h (module 'network'): bool ns3::PbbTlvBlock::Empty() const [member function]
    cls.add_method('Empty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbTlvBlock::Iterator ns3::PbbTlvBlock::End() [member function]
    cls.add_method('End', 
                   'ns3::PbbTlvBlock::Iterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbTlvBlock::ConstIterator ns3::PbbTlvBlock::End() const [member function]
    cls.add_method('End', 
                   'ns3::PbbTlvBlock::ConstIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbTlvBlock::Iterator ns3::PbbTlvBlock::Erase(ns3::PbbTlvBlock::Iterator position) [member function]
    cls.add_method('Erase', 
                   'ns3::PbbTlvBlock::Iterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'position')])
    ## packetbb.h (module 'network'): ns3::PbbTlvBlock::Iterator ns3::PbbTlvBlock::Erase(ns3::PbbTlvBlock::Iterator first, ns3::PbbTlvBlock::Iterator last) [member function]
    cls.add_method('Erase', 
                   'ns3::PbbTlvBlock::Iterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'first'), param('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'last')])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbTlv> ns3::PbbTlvBlock::Front() const [member function]
    cls.add_method('Front', 
                   'ns3::Ptr< ns3::PbbTlv >', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint32_t ns3::PbbTlvBlock::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbTlvBlock::Iterator ns3::PbbTlvBlock::Insert(ns3::PbbTlvBlock::Iterator position, ns3::Ptr<ns3::PbbTlv> const tlv) [member function]
    cls.add_method('Insert', 
                   'ns3::PbbTlvBlock::Iterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'position'), param('ns3::Ptr< ns3::PbbTlv > const', 'tlv')])
    ## packetbb.h (module 'network'): void ns3::PbbTlvBlock::PopBack() [member function]
    cls.add_method('PopBack', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbTlvBlock::PopFront() [member function]
    cls.add_method('PopFront', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbTlvBlock::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbTlvBlock::Print(std::ostream & os, int level) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os'), param('int', 'level')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbTlvBlock::PushBack(ns3::Ptr<ns3::PbbTlv> tlv) [member function]
    cls.add_method('PushBack', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbTlv >', 'tlv')])
    ## packetbb.h (module 'network'): void ns3::PbbTlvBlock::PushFront(ns3::Ptr<ns3::PbbTlv> tlv) [member function]
    cls.add_method('PushFront', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbTlv >', 'tlv')])
    ## packetbb.h (module 'network'): void ns3::PbbTlvBlock::Serialize(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True)
    ## packetbb.h (module 'network'): int ns3::PbbTlvBlock::Size() const [member function]
    cls.add_method('Size', 
                   'int', 
                   [], 
                   is_const=True)
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

def register_Ns3QueueSize_methods(root_module, cls):
    cls.add_output_stream_operator()
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_comparison_operator('<')
    cls.add_binary_comparison_operator('<=')
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('>')
    cls.add_binary_comparison_operator('>=')
    ## queue-size.h (module 'network'): ns3::QueueSize::QueueSize(ns3::QueueSize const & arg0) [constructor]
    cls.add_constructor([param('ns3::QueueSize const &', 'arg0')])
    ## queue-size.h (module 'network'): ns3::QueueSize::QueueSize() [constructor]
    cls.add_constructor([])
    ## queue-size.h (module 'network'): ns3::QueueSize::QueueSize(ns3::QueueSizeUnit unit, uint32_t value) [constructor]
    cls.add_constructor([param('ns3::QueueSizeUnit', 'unit'), param('uint32_t', 'value')])
    ## queue-size.h (module 'network'): ns3::QueueSize::QueueSize(std::string size) [constructor]
    cls.add_constructor([param('std::string', 'size')])
    ## queue-size.h (module 'network'): ns3::QueueSizeUnit ns3::QueueSize::GetUnit() const [member function]
    cls.add_method('GetUnit', 
                   'ns3::QueueSizeUnit', 
                   [], 
                   is_const=True)
    ## queue-size.h (module 'network'): uint32_t ns3::QueueSize::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    return

def register_Ns3SequenceNumber32_methods(root_module, cls):
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_numeric_operator('+', root_module['ns3::SequenceNumber32'], root_module['ns3::SequenceNumber32'], param('ns3::SequenceNumber< unsigned int, int > const &', 'right'))
    cls.add_binary_numeric_operator('+', root_module['ns3::SequenceNumber32'], root_module['ns3::SequenceNumber32'], param('int', 'right'))
    cls.add_inplace_numeric_operator('+=', param('int', 'right'))
    cls.add_binary_numeric_operator('-', root_module['ns3::SequenceNumber32'], root_module['ns3::SequenceNumber32'], param('int', 'right'))
    cls.add_inplace_numeric_operator('-=', param('int', 'right'))
    cls.add_binary_comparison_operator('<')
    cls.add_binary_comparison_operator('<=')
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('>')
    cls.add_binary_comparison_operator('>=')
    ## sequence-number.h (module 'network'): ns3::SequenceNumber<unsigned int, int>::SequenceNumber() [constructor]
    cls.add_constructor([])
    ## sequence-number.h (module 'network'): ns3::SequenceNumber<unsigned int, int>::SequenceNumber(unsigned int value) [constructor]
    cls.add_constructor([param('unsigned int', 'value')])
    ## sequence-number.h (module 'network'): ns3::SequenceNumber<unsigned int, int>::SequenceNumber(ns3::SequenceNumber<unsigned int, int> const & value) [constructor]
    cls.add_constructor([param('ns3::SequenceNumber< unsigned int, int > const &', 'value')])
    ## sequence-number.h (module 'network'): unsigned int ns3::SequenceNumber<unsigned int, int>::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'unsigned int', 
                   [], 
                   is_const=True)
    return

def register_Ns3SequenceNumber16_methods(root_module, cls):
    cls.add_binary_comparison_operator('!=')
    cls.add_binary_numeric_operator('+', root_module['ns3::SequenceNumber16'], root_module['ns3::SequenceNumber16'], param('ns3::SequenceNumber< unsigned short, short > const &', 'right'))
    cls.add_binary_numeric_operator('+', root_module['ns3::SequenceNumber16'], root_module['ns3::SequenceNumber16'], param('short int', 'right'))
    cls.add_inplace_numeric_operator('+=', param('short int', 'right'))
    cls.add_binary_numeric_operator('-', root_module['ns3::SequenceNumber16'], root_module['ns3::SequenceNumber16'], param('short int', 'right'))
    cls.add_inplace_numeric_operator('-=', param('short int', 'right'))
    cls.add_binary_comparison_operator('<')
    cls.add_binary_comparison_operator('<=')
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('>')
    cls.add_binary_comparison_operator('>=')
    ## sequence-number.h (module 'network'): ns3::SequenceNumber<unsigned short, short>::SequenceNumber() [constructor]
    cls.add_constructor([])
    ## sequence-number.h (module 'network'): ns3::SequenceNumber<unsigned short, short>::SequenceNumber(short unsigned int value) [constructor]
    cls.add_constructor([param('short unsigned int', 'value')])
    ## sequence-number.h (module 'network'): ns3::SequenceNumber<unsigned short, short>::SequenceNumber(ns3::SequenceNumber<unsigned short, short> const & value) [constructor]
    cls.add_constructor([param('ns3::SequenceNumber< unsigned short, short > const &', 'value')])
    ## sequence-number.h (module 'network'): short unsigned int ns3::SequenceNumber<unsigned short, short>::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'short unsigned int', 
                   [], 
                   is_const=True)
    return

def register_Ns3SimpleNetDeviceHelper_methods(root_module, cls):
    ## simple-net-device-helper.h (module 'network'): ns3::SimpleNetDeviceHelper::SimpleNetDeviceHelper(ns3::SimpleNetDeviceHelper const & arg0) [constructor]
    cls.add_constructor([param('ns3::SimpleNetDeviceHelper const &', 'arg0')])
    ## simple-net-device-helper.h (module 'network'): ns3::SimpleNetDeviceHelper::SimpleNetDeviceHelper() [constructor]
    cls.add_constructor([])
    ## simple-net-device-helper.h (module 'network'): void ns3::SimpleNetDeviceHelper::DisableFlowControl() [member function]
    cls.add_method('DisableFlowControl', 
                   'void', 
                   [])
    ## simple-net-device-helper.h (module 'network'): ns3::NetDeviceContainer ns3::SimpleNetDeviceHelper::Install(ns3::Ptr<ns3::Node> node) const [member function]
    cls.add_method('Install', 
                   'ns3::NetDeviceContainer', 
                   [param('ns3::Ptr< ns3::Node >', 'node')], 
                   is_const=True)
    ## simple-net-device-helper.h (module 'network'): ns3::NetDeviceContainer ns3::SimpleNetDeviceHelper::Install(ns3::Ptr<ns3::Node> node, ns3::Ptr<ns3::SimpleChannel> channel) const [member function]
    cls.add_method('Install', 
                   'ns3::NetDeviceContainer', 
                   [param('ns3::Ptr< ns3::Node >', 'node'), param('ns3::Ptr< ns3::SimpleChannel >', 'channel')], 
                   is_const=True)
    ## simple-net-device-helper.h (module 'network'): ns3::NetDeviceContainer ns3::SimpleNetDeviceHelper::Install(ns3::NodeContainer const & c) const [member function]
    cls.add_method('Install', 
                   'ns3::NetDeviceContainer', 
                   [param('ns3::NodeContainer const &', 'c')], 
                   is_const=True)
    ## simple-net-device-helper.h (module 'network'): ns3::NetDeviceContainer ns3::SimpleNetDeviceHelper::Install(ns3::NodeContainer const & c, ns3::Ptr<ns3::SimpleChannel> channel) const [member function]
    cls.add_method('Install', 
                   'ns3::NetDeviceContainer', 
                   [param('ns3::NodeContainer const &', 'c'), param('ns3::Ptr< ns3::SimpleChannel >', 'channel')], 
                   is_const=True)
    ## simple-net-device-helper.h (module 'network'): void ns3::SimpleNetDeviceHelper::SetChannel(std::string type, std::string n1="", ns3::AttributeValue const & v1=ns3::EmptyAttributeValue(), std::string n2="", ns3::AttributeValue const & v2=ns3::EmptyAttributeValue(), std::string n3="", ns3::AttributeValue const & v3=ns3::EmptyAttributeValue(), std::string n4="", ns3::AttributeValue const & v4=ns3::EmptyAttributeValue()) [member function]
    cls.add_method('SetChannel', 
                   'void', 
                   [param('std::string', 'type'), param('std::string', 'n1', default_value='""'), param('ns3::AttributeValue const &', 'v1', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n2', default_value='""'), param('ns3::AttributeValue const &', 'v2', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n3', default_value='""'), param('ns3::AttributeValue const &', 'v3', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n4', default_value='""'), param('ns3::AttributeValue const &', 'v4', default_value='ns3::EmptyAttributeValue()')])
    ## simple-net-device-helper.h (module 'network'): void ns3::SimpleNetDeviceHelper::SetChannelAttribute(std::string n1, ns3::AttributeValue const & v1) [member function]
    cls.add_method('SetChannelAttribute', 
                   'void', 
                   [param('std::string', 'n1'), param('ns3::AttributeValue const &', 'v1')])
    ## simple-net-device-helper.h (module 'network'): void ns3::SimpleNetDeviceHelper::SetDeviceAttribute(std::string n1, ns3::AttributeValue const & v1) [member function]
    cls.add_method('SetDeviceAttribute', 
                   'void', 
                   [param('std::string', 'n1'), param('ns3::AttributeValue const &', 'v1')])
    ## simple-net-device-helper.h (module 'network'): void ns3::SimpleNetDeviceHelper::SetNetDevicePointToPointMode(bool pointToPointMode) [member function]
    cls.add_method('SetNetDevicePointToPointMode', 
                   'void', 
                   [param('bool', 'pointToPointMode')])
    ## simple-net-device-helper.h (module 'network'): void ns3::SimpleNetDeviceHelper::SetQueue(std::string type, std::string n1="", ns3::AttributeValue const & v1=ns3::EmptyAttributeValue(), std::string n2="", ns3::AttributeValue const & v2=ns3::EmptyAttributeValue(), std::string n3="", ns3::AttributeValue const & v3=ns3::EmptyAttributeValue(), std::string n4="", ns3::AttributeValue const & v4=ns3::EmptyAttributeValue()) [member function]
    cls.add_method('SetQueue', 
                   'void', 
                   [param('std::string', 'type'), param('std::string', 'n1', default_value='""'), param('ns3::AttributeValue const &', 'v1', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n2', default_value='""'), param('ns3::AttributeValue const &', 'v2', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n3', default_value='""'), param('ns3::AttributeValue const &', 'v3', default_value='ns3::EmptyAttributeValue()'), param('std::string', 'n4', default_value='""'), param('ns3::AttributeValue const &', 'v4', default_value='ns3::EmptyAttributeValue()')])
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

def register_Ns3StatisticalSummary_methods(root_module, cls):
    ## data-calculator.h (module 'stats'): ns3::StatisticalSummary::StatisticalSummary() [constructor]
    cls.add_constructor([])
    ## data-calculator.h (module 'stats'): ns3::StatisticalSummary::StatisticalSummary(ns3::StatisticalSummary const & arg0) [constructor]
    cls.add_constructor([param('ns3::StatisticalSummary const &', 'arg0')])
    ## data-calculator.h (module 'stats'): long int ns3::StatisticalSummary::getCount() const [member function]
    cls.add_method('getCount', 
                   'long int', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## data-calculator.h (module 'stats'): double ns3::StatisticalSummary::getMax() const [member function]
    cls.add_method('getMax', 
                   'double', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## data-calculator.h (module 'stats'): double ns3::StatisticalSummary::getMean() const [member function]
    cls.add_method('getMean', 
                   'double', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## data-calculator.h (module 'stats'): double ns3::StatisticalSummary::getMin() const [member function]
    cls.add_method('getMin', 
                   'double', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## data-calculator.h (module 'stats'): double ns3::StatisticalSummary::getSqrSum() const [member function]
    cls.add_method('getSqrSum', 
                   'double', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## data-calculator.h (module 'stats'): double ns3::StatisticalSummary::getStddev() const [member function]
    cls.add_method('getStddev', 
                   'double', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## data-calculator.h (module 'stats'): double ns3::StatisticalSummary::getSum() const [member function]
    cls.add_method('getSum', 
                   'double', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## data-calculator.h (module 'stats'): double ns3::StatisticalSummary::getVariance() const [member function]
    cls.add_method('getVariance', 
                   'double', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3SystemWallClockMs_methods(root_module, cls):
    ## system-wall-clock-ms.h (module 'core'): ns3::SystemWallClockMs::SystemWallClockMs(ns3::SystemWallClockMs const & arg0) [constructor]
    cls.add_constructor([param('ns3::SystemWallClockMs const &', 'arg0')])
    ## system-wall-clock-ms.h (module 'core'): ns3::SystemWallClockMs::SystemWallClockMs() [constructor]
    cls.add_constructor([])
    ## system-wall-clock-ms.h (module 'core'): int64_t ns3::SystemWallClockMs::End() [member function]
    cls.add_method('End', 
                   'int64_t', 
                   [])
    ## system-wall-clock-ms.h (module 'core'): int64_t ns3::SystemWallClockMs::GetElapsedReal() const [member function]
    cls.add_method('GetElapsedReal', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## system-wall-clock-ms.h (module 'core'): int64_t ns3::SystemWallClockMs::GetElapsedSystem() const [member function]
    cls.add_method('GetElapsedSystem', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## system-wall-clock-ms.h (module 'core'): int64_t ns3::SystemWallClockMs::GetElapsedUser() const [member function]
    cls.add_method('GetElapsedUser', 
                   'int64_t', 
                   [], 
                   is_const=True)
    ## system-wall-clock-ms.h (module 'core'): void ns3::SystemWallClockMs::Start() [member function]
    cls.add_method('Start', 
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

def register_Ns3TracedValue__Unsigned_int_methods(root_module, cls):
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned int>::TracedValue() [constructor]
    cls.add_constructor([])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned int>::TracedValue(ns3::TracedValue<unsigned int> const & o) [constructor]
    cls.add_constructor([param('ns3::TracedValue< unsigned int > const &', 'o')])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned int>::TracedValue(unsigned int const & v) [constructor]
    cls.add_constructor([param('unsigned int const &', 'v')])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned int>::TracedValue(ns3::TracedValue<unsigned int> const & other) [constructor]
    cls.add_constructor([param('ns3::TracedValue< unsigned int > const &', 'other')])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned int>::TracedValue(ns3::TracedValue<unsigned int> const & other) [constructor]
    cls.add_constructor([param('ns3::TracedValue< unsigned int > const &', 'other')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned int>::Connect(ns3::CallbackBase const & cb, std::string path) [member function]
    cls.add_method('Connect', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb'), param('std::string', 'path')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned int>::ConnectWithoutContext(ns3::CallbackBase const & cb) [member function]
    cls.add_method('ConnectWithoutContext', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned int>::Disconnect(ns3::CallbackBase const & cb, std::string path) [member function]
    cls.add_method('Disconnect', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb'), param('std::string', 'path')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned int>::DisconnectWithoutContext(ns3::CallbackBase const & cb) [member function]
    cls.add_method('DisconnectWithoutContext', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb')])
    ## traced-value.h (module 'core'): unsigned int ns3::TracedValue<unsigned int>::Get() const [member function]
    cls.add_method('Get', 
                   'unsigned int', 
                   [], 
                   is_const=True)
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned int>::Set(unsigned int const & v) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('unsigned int const &', 'v')])
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
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeId::SetParent() [member function]
    cls.add_method('SetParent', 
                   'ns3::TypeId', 
                   [], 
                   template_parameters=['ns3::QueueBase'])
    ## type-id.h (module 'core'): ns3::TypeId ns3::TypeId::SetParent() [member function]
    cls.add_method('SetParent', 
                   'ns3::TypeId', 
                   [], 
                   template_parameters=['ns3::Object'])
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

def register_Ns3DeviceNameTag_methods(root_module, cls):
    ## packet-socket.h (module 'network'): ns3::DeviceNameTag::DeviceNameTag(ns3::DeviceNameTag const & arg0) [constructor]
    cls.add_constructor([param('ns3::DeviceNameTag const &', 'arg0')])
    ## packet-socket.h (module 'network'): ns3::DeviceNameTag::DeviceNameTag() [constructor]
    cls.add_constructor([])
    ## packet-socket.h (module 'network'): void ns3::DeviceNameTag::Deserialize(ns3::TagBuffer i) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): std::string ns3::DeviceNameTag::GetDeviceName() const [member function]
    cls.add_method('GetDeviceName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## packet-socket.h (module 'network'): ns3::TypeId ns3::DeviceNameTag::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): uint32_t ns3::DeviceNameTag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): static ns3::TypeId ns3::DeviceNameTag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## packet-socket.h (module 'network'): void ns3::DeviceNameTag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): void ns3::DeviceNameTag::Serialize(ns3::TagBuffer i) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): void ns3::DeviceNameTag::SetDeviceName(std::string n) [member function]
    cls.add_method('SetDeviceName', 
                   'void', 
                   [param('std::string', 'n')])
    return

def register_Ns3FlowIdTag_methods(root_module, cls):
    ## flow-id-tag.h (module 'network'): ns3::FlowIdTag::FlowIdTag(ns3::FlowIdTag const & arg0) [constructor]
    cls.add_constructor([param('ns3::FlowIdTag const &', 'arg0')])
    ## flow-id-tag.h (module 'network'): ns3::FlowIdTag::FlowIdTag() [constructor]
    cls.add_constructor([])
    ## flow-id-tag.h (module 'network'): ns3::FlowIdTag::FlowIdTag(uint32_t flowId) [constructor]
    cls.add_constructor([param('uint32_t', 'flowId')])
    ## flow-id-tag.h (module 'network'): static uint32_t ns3::FlowIdTag::AllocateFlowId() [member function]
    cls.add_method('AllocateFlowId', 
                   'uint32_t', 
                   [], 
                   is_static=True)
    ## flow-id-tag.h (module 'network'): void ns3::FlowIdTag::Deserialize(ns3::TagBuffer buf) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'buf')], 
                   is_virtual=True)
    ## flow-id-tag.h (module 'network'): uint32_t ns3::FlowIdTag::GetFlowId() const [member function]
    cls.add_method('GetFlowId', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## flow-id-tag.h (module 'network'): ns3::TypeId ns3::FlowIdTag::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## flow-id-tag.h (module 'network'): uint32_t ns3::FlowIdTag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## flow-id-tag.h (module 'network'): static ns3::TypeId ns3::FlowIdTag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## flow-id-tag.h (module 'network'): void ns3::FlowIdTag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## flow-id-tag.h (module 'network'): void ns3::FlowIdTag::Serialize(ns3::TagBuffer buf) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'buf')], 
                   is_const=True, is_virtual=True)
    ## flow-id-tag.h (module 'network'): void ns3::FlowIdTag::SetFlowId(uint32_t flowId) [member function]
    cls.add_method('SetFlowId', 
                   'void', 
                   [param('uint32_t', 'flowId')])
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

def register_Ns3LlcSnapHeader_methods(root_module, cls):
    ## llc-snap-header.h (module 'network'): ns3::LlcSnapHeader::LlcSnapHeader(ns3::LlcSnapHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::LlcSnapHeader const &', 'arg0')])
    ## llc-snap-header.h (module 'network'): ns3::LlcSnapHeader::LlcSnapHeader() [constructor]
    cls.add_constructor([])
    ## llc-snap-header.h (module 'network'): uint32_t ns3::LlcSnapHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## llc-snap-header.h (module 'network'): ns3::TypeId ns3::LlcSnapHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## llc-snap-header.h (module 'network'): uint32_t ns3::LlcSnapHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## llc-snap-header.h (module 'network'): uint16_t ns3::LlcSnapHeader::GetType() [member function]
    cls.add_method('GetType', 
                   'uint16_t', 
                   [])
    ## llc-snap-header.h (module 'network'): static ns3::TypeId ns3::LlcSnapHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## llc-snap-header.h (module 'network'): void ns3::LlcSnapHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## llc-snap-header.h (module 'network'): void ns3::LlcSnapHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## llc-snap-header.h (module 'network'): void ns3::LlcSnapHeader::SetType(uint16_t type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('uint16_t', 'type')])
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

def register_Ns3PacketSocketTag_methods(root_module, cls):
    ## packet-socket.h (module 'network'): ns3::PacketSocketTag::PacketSocketTag(ns3::PacketSocketTag const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketSocketTag const &', 'arg0')])
    ## packet-socket.h (module 'network'): ns3::PacketSocketTag::PacketSocketTag() [constructor]
    cls.add_constructor([])
    ## packet-socket.h (module 'network'): void ns3::PacketSocketTag::Deserialize(ns3::TagBuffer i) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): ns3::Address ns3::PacketSocketTag::GetDestAddress() const [member function]
    cls.add_method('GetDestAddress', 
                   'ns3::Address', 
                   [], 
                   is_const=True)
    ## packet-socket.h (module 'network'): ns3::TypeId ns3::PacketSocketTag::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): ns3::NetDevice::PacketType ns3::PacketSocketTag::GetPacketType() const [member function]
    cls.add_method('GetPacketType', 
                   'ns3::NetDevice::PacketType', 
                   [], 
                   is_const=True)
    ## packet-socket.h (module 'network'): uint32_t ns3::PacketSocketTag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): static ns3::TypeId ns3::PacketSocketTag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## packet-socket.h (module 'network'): void ns3::PacketSocketTag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): void ns3::PacketSocketTag::Serialize(ns3::TagBuffer i) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): void ns3::PacketSocketTag::SetDestAddress(ns3::Address a) [member function]
    cls.add_method('SetDestAddress', 
                   'void', 
                   [param('ns3::Address', 'a')])
    ## packet-socket.h (module 'network'): void ns3::PacketSocketTag::SetPacketType(ns3::NetDevice::PacketType t) [member function]
    cls.add_method('SetPacketType', 
                   'void', 
                   [param('ns3::NetDevice::PacketType', 't')])
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

def register_Ns3QueueBase_methods(root_module, cls):
    ## queue.h (module 'network'): ns3::QueueBase::QueueBase(ns3::QueueBase const & arg0) [constructor]
    cls.add_constructor([param('ns3::QueueBase const &', 'arg0')])
    ## queue.h (module 'network'): ns3::QueueBase::QueueBase() [constructor]
    cls.add_constructor([])
    ## queue.h (module 'network'): static void ns3::QueueBase::AppendItemTypeIfNotPresent(std::string & typeId, std::string const & itemType) [member function]
    cls.add_method('AppendItemTypeIfNotPresent', 
                   'void', 
                   [param('std::string &', 'typeId'), param('std::string const &', 'itemType')], 
                   is_static=True)
    ## queue.h (module 'network'): ns3::QueueSize ns3::QueueBase::GetCurrentSize() const [member function]
    cls.add_method('GetCurrentSize', 
                   'ns3::QueueSize', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): ns3::QueueSize ns3::QueueBase::GetMaxSize() const [member function]
    cls.add_method('GetMaxSize', 
                   'ns3::QueueSize', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): uint32_t ns3::QueueBase::GetNBytes() const [member function]
    cls.add_method('GetNBytes', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): uint32_t ns3::QueueBase::GetNPackets() const [member function]
    cls.add_method('GetNPackets', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): uint32_t ns3::QueueBase::GetTotalDroppedBytes() const [member function]
    cls.add_method('GetTotalDroppedBytes', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): uint32_t ns3::QueueBase::GetTotalDroppedBytesAfterDequeue() const [member function]
    cls.add_method('GetTotalDroppedBytesAfterDequeue', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): uint32_t ns3::QueueBase::GetTotalDroppedBytesBeforeEnqueue() const [member function]
    cls.add_method('GetTotalDroppedBytesBeforeEnqueue', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): uint32_t ns3::QueueBase::GetTotalDroppedPackets() const [member function]
    cls.add_method('GetTotalDroppedPackets', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): uint32_t ns3::QueueBase::GetTotalDroppedPacketsAfterDequeue() const [member function]
    cls.add_method('GetTotalDroppedPacketsAfterDequeue', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): uint32_t ns3::QueueBase::GetTotalDroppedPacketsBeforeEnqueue() const [member function]
    cls.add_method('GetTotalDroppedPacketsBeforeEnqueue', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): uint32_t ns3::QueueBase::GetTotalReceivedBytes() const [member function]
    cls.add_method('GetTotalReceivedBytes', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): uint32_t ns3::QueueBase::GetTotalReceivedPackets() const [member function]
    cls.add_method('GetTotalReceivedPackets', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): static ns3::TypeId ns3::QueueBase::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## queue.h (module 'network'): bool ns3::QueueBase::IsEmpty() const [member function]
    cls.add_method('IsEmpty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## queue.h (module 'network'): void ns3::QueueBase::ResetStatistics() [member function]
    cls.add_method('ResetStatistics', 
                   'void', 
                   [])
    ## queue.h (module 'network'): void ns3::QueueBase::SetMaxSize(ns3::QueueSize size) [member function]
    cls.add_method('SetMaxSize', 
                   'void', 
                   [param('ns3::QueueSize', 'size')])
    ## queue.h (module 'network'): bool ns3::QueueBase::WouldOverflow(uint32_t nPackets, uint32_t nBytes) const [member function]
    cls.add_method('WouldOverflow', 
                   'bool', 
                   [param('uint32_t', 'nPackets'), param('uint32_t', 'nBytes')], 
                   is_const=True)
    return

def register_Ns3QueueLimits_methods(root_module, cls):
    ## queue-limits.h (module 'network'): ns3::QueueLimits::QueueLimits() [constructor]
    cls.add_constructor([])
    ## queue-limits.h (module 'network'): ns3::QueueLimits::QueueLimits(ns3::QueueLimits const & arg0) [constructor]
    cls.add_constructor([param('ns3::QueueLimits const &', 'arg0')])
    ## queue-limits.h (module 'network'): int32_t ns3::QueueLimits::Available() const [member function]
    cls.add_method('Available', 
                   'int32_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## queue-limits.h (module 'network'): void ns3::QueueLimits::Completed(uint32_t count) [member function]
    cls.add_method('Completed', 
                   'void', 
                   [param('uint32_t', 'count')], 
                   is_pure_virtual=True, is_virtual=True)
    ## queue-limits.h (module 'network'): static ns3::TypeId ns3::QueueLimits::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## queue-limits.h (module 'network'): void ns3::QueueLimits::Queued(uint32_t count) [member function]
    cls.add_method('Queued', 
                   'void', 
                   [param('uint32_t', 'count')], 
                   is_pure_virtual=True, is_virtual=True)
    ## queue-limits.h (module 'network'): void ns3::QueueLimits::Reset() [member function]
    cls.add_method('Reset', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3RadiotapHeader_methods(root_module, cls):
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::RadiotapHeader(ns3::RadiotapHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::RadiotapHeader const &', 'arg0')])
    ## radiotap-header.h (module 'network'): ns3::RadiotapHeader::RadiotapHeader() [constructor]
    cls.add_constructor([])
    ## radiotap-header.h (module 'network'): uint32_t ns3::RadiotapHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## radiotap-header.h (module 'network'): ns3::TypeId ns3::RadiotapHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## radiotap-header.h (module 'network'): uint32_t ns3::RadiotapHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## radiotap-header.h (module 'network'): static ns3::TypeId ns3::RadiotapHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetAmpduStatus(uint32_t referenceNumber, uint16_t flags, uint8_t crc) [member function]
    cls.add_method('SetAmpduStatus', 
                   'void', 
                   [param('uint32_t', 'referenceNumber'), param('uint16_t', 'flags'), param('uint8_t', 'crc')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetAntennaNoisePower(double noise) [member function]
    cls.add_method('SetAntennaNoisePower', 
                   'void', 
                   [param('double', 'noise')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetAntennaSignalPower(double signal) [member function]
    cls.add_method('SetAntennaSignalPower', 
                   'void', 
                   [param('double', 'signal')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetChannelFrequencyAndFlags(uint16_t frequency, uint16_t flags) [member function]
    cls.add_method('SetChannelFrequencyAndFlags', 
                   'void', 
                   [param('uint16_t', 'frequency'), param('uint16_t', 'flags')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetFrameFlags(uint8_t flags) [member function]
    cls.add_method('SetFrameFlags', 
                   'void', 
                   [param('uint8_t', 'flags')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetHeFields(uint16_t data1, uint16_t data2, uint16_t data3, uint16_t data4, uint16_t data5, uint16_t data6) [member function]
    cls.add_method('SetHeFields', 
                   'void', 
                   [param('uint16_t', 'data1'), param('uint16_t', 'data2'), param('uint16_t', 'data3'), param('uint16_t', 'data4'), param('uint16_t', 'data5'), param('uint16_t', 'data6')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetHeMuFields(uint16_t flags1, uint16_t flags2, std::array<unsigned char, 4> const & ruChannel1, std::array<unsigned char, 4> const & ruChannel2) [member function]
    cls.add_method('SetHeMuFields', 
                   'void', 
                   [param('uint16_t', 'flags1'), param('uint16_t', 'flags2'), param('std::array< unsigned char, 4 > const &', 'ruChannel1'), param('std::array< unsigned char, 4 > const &', 'ruChannel2')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetHeMuPerUserFields(uint16_t perUser1, uint16_t perUser2, uint8_t perUserPosition, uint8_t perUserKnown) [member function]
    cls.add_method('SetHeMuPerUserFields', 
                   'void', 
                   [param('uint16_t', 'perUser1'), param('uint16_t', 'perUser2'), param('uint8_t', 'perUserPosition'), param('uint8_t', 'perUserKnown')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetMcsFields(uint8_t known, uint8_t flags, uint8_t mcs) [member function]
    cls.add_method('SetMcsFields', 
                   'void', 
                   [param('uint8_t', 'known'), param('uint8_t', 'flags'), param('uint8_t', 'mcs')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetRate(uint8_t rate) [member function]
    cls.add_method('SetRate', 
                   'void', 
                   [param('uint8_t', 'rate')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetTsft(uint64_t tsft) [member function]
    cls.add_method('SetTsft', 
                   'void', 
                   [param('uint64_t', 'tsft')])
    ## radiotap-header.h (module 'network'): void ns3::RadiotapHeader::SetVhtFields(uint16_t known, uint8_t flags, uint8_t bandwidth, uint8_t * mcs_nss, uint8_t coding, uint8_t group_id, uint16_t partial_aid) [member function]
    cls.add_method('SetVhtFields', 
                   'void', 
                   [param('uint16_t', 'known'), param('uint8_t', 'flags'), param('uint8_t', 'bandwidth'), param('uint8_t *', 'mcs_nss'), param('uint8_t', 'coding'), param('uint8_t', 'group_id'), param('uint16_t', 'partial_aid')])
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

def register_Ns3SimpleRefCount__Ns3PbbAddressBlock_Ns3Empty_Ns3DefaultDeleter__lt__ns3PbbAddressBlock__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbAddressBlock, ns3::empty, ns3::DefaultDeleter<ns3::PbbAddressBlock>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbAddressBlock, ns3::empty, ns3::DefaultDeleter<ns3::PbbAddressBlock>>::SimpleRefCount(ns3::SimpleRefCount<ns3::PbbAddressBlock, ns3::empty, ns3::DefaultDeleter<ns3::PbbAddressBlock>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::PbbAddressBlock, ns3::empty, ns3::DefaultDeleter< ns3::PbbAddressBlock > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3PbbMessage_Ns3Empty_Ns3DefaultDeleter__lt__ns3PbbMessage__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbMessage, ns3::empty, ns3::DefaultDeleter<ns3::PbbMessage>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbMessage, ns3::empty, ns3::DefaultDeleter<ns3::PbbMessage>>::SimpleRefCount(ns3::SimpleRefCount<ns3::PbbMessage, ns3::empty, ns3::DefaultDeleter<ns3::PbbMessage>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::PbbMessage, ns3::empty, ns3::DefaultDeleter< ns3::PbbMessage > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3PbbPacket_Ns3Header_Ns3DefaultDeleter__lt__ns3PbbPacket__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbPacket, ns3::Header, ns3::DefaultDeleter<ns3::PbbPacket>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbPacket, ns3::Header, ns3::DefaultDeleter<ns3::PbbPacket>>::SimpleRefCount(ns3::SimpleRefCount<ns3::PbbPacket, ns3::Header, ns3::DefaultDeleter<ns3::PbbPacket>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::PbbPacket, ns3::Header, ns3::DefaultDeleter< ns3::PbbPacket > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3PbbTlv_Ns3Empty_Ns3DefaultDeleter__lt__ns3PbbTlv__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbTlv, ns3::empty, ns3::DefaultDeleter<ns3::PbbTlv>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::PbbTlv, ns3::empty, ns3::DefaultDeleter<ns3::PbbTlv>>::SimpleRefCount(ns3::SimpleRefCount<ns3::PbbTlv, ns3::empty, ns3::DefaultDeleter<ns3::PbbTlv>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::PbbTlv, ns3::empty, ns3::DefaultDeleter< ns3::PbbTlv > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3QueueItem_Ns3Empty_Ns3DefaultDeleter__lt__ns3QueueItem__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::QueueItem, ns3::empty, ns3::DefaultDeleter<ns3::QueueItem>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::QueueItem, ns3::empty, ns3::DefaultDeleter<ns3::QueueItem>>::SimpleRefCount(ns3::SimpleRefCount<ns3::QueueItem, ns3::empty, ns3::DefaultDeleter<ns3::QueueItem>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::QueueItem, ns3::empty, ns3::DefaultDeleter< ns3::QueueItem > > const &', 'o')])
    return

def register_Ns3SimpleRefCount__Ns3TraceSourceAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3TraceSourceAccessor__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>>::SimpleRefCount(ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter< ns3::TraceSourceAccessor > > const &', 'o')])
    return

def register_Ns3SllHeader_methods(root_module, cls):
    ## sll-header.h (module 'network'): ns3::SllHeader::SllHeader(ns3::SllHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::SllHeader const &', 'arg0')])
    ## sll-header.h (module 'network'): ns3::SllHeader::SllHeader() [constructor]
    cls.add_constructor([])
    ## sll-header.h (module 'network'): uint32_t ns3::SllHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## sll-header.h (module 'network'): uint16_t ns3::SllHeader::GetArpType() const [member function]
    cls.add_method('GetArpType', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## sll-header.h (module 'network'): ns3::TypeId ns3::SllHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## sll-header.h (module 'network'): ns3::SllHeader::PacketType ns3::SllHeader::GetPacketType() const [member function]
    cls.add_method('GetPacketType', 
                   'ns3::SllHeader::PacketType', 
                   [], 
                   is_const=True)
    ## sll-header.h (module 'network'): uint32_t ns3::SllHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## sll-header.h (module 'network'): static ns3::TypeId ns3::SllHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## sll-header.h (module 'network'): void ns3::SllHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## sll-header.h (module 'network'): void ns3::SllHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## sll-header.h (module 'network'): void ns3::SllHeader::SetArpType(uint16_t arphdType) [member function]
    cls.add_method('SetArpType', 
                   'void', 
                   [param('uint16_t', 'arphdType')])
    ## sll-header.h (module 'network'): void ns3::SllHeader::SetPacketType(ns3::SllHeader::PacketType type) [member function]
    cls.add_method('SetPacketType', 
                   'void', 
                   [param('ns3::SllHeader::PacketType', 'type')])
    return

def register_Ns3Socket_methods(root_module, cls):
    ## socket.h (module 'network'): ns3::Socket::Socket(ns3::Socket const & arg0) [constructor]
    cls.add_constructor([param('ns3::Socket const &', 'arg0')])
    ## socket.h (module 'network'): ns3::Socket::Socket() [constructor]
    cls.add_constructor([])
    ## socket.h (module 'network'): int ns3::Socket::Bind(ns3::Address const & address) [member function]
    cls.add_method('Bind', 
                   'int', 
                   [param('ns3::Address const &', 'address')], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): int ns3::Socket::Bind() [member function]
    cls.add_method('Bind', 
                   'int', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): int ns3::Socket::Bind6() [member function]
    cls.add_method('Bind6', 
                   'int', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::Socket::BindToNetDevice(ns3::Ptr<ns3::NetDevice> netdevice) [member function]
    cls.add_method('BindToNetDevice', 
                   'void', 
                   [param('ns3::Ptr< ns3::NetDevice >', 'netdevice')], 
                   is_virtual=True)
    ## socket.h (module 'network'): int ns3::Socket::Close() [member function]
    cls.add_method('Close', 
                   'int', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): int ns3::Socket::Connect(ns3::Address const & address) [member function]
    cls.add_method('Connect', 
                   'int', 
                   [param('ns3::Address const &', 'address')], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): static ns3::Ptr<ns3::Socket> ns3::Socket::CreateSocket(ns3::Ptr<ns3::Node> node, ns3::TypeId tid) [member function]
    cls.add_method('CreateSocket', 
                   'ns3::Ptr< ns3::Socket >', 
                   [param('ns3::Ptr< ns3::Node >', 'node'), param('ns3::TypeId', 'tid')], 
                   is_static=True)
    ## socket.h (module 'network'): bool ns3::Socket::GetAllowBroadcast() const [member function]
    cls.add_method('GetAllowBroadcast', 
                   'bool', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): ns3::Ptr<ns3::NetDevice> ns3::Socket::GetBoundNetDevice() [member function]
    cls.add_method('GetBoundNetDevice', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [])
    ## socket.h (module 'network'): ns3::Socket::SocketErrno ns3::Socket::GetErrno() const [member function]
    cls.add_method('GetErrno', 
                   'ns3::Socket::SocketErrno', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): uint8_t ns3::Socket::GetIpTos() const [member function]
    cls.add_method('GetIpTos', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): uint8_t ns3::Socket::GetIpTtl() const [member function]
    cls.add_method('GetIpTtl', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint8_t ns3::Socket::GetIpv6HopLimit() const [member function]
    cls.add_method('GetIpv6HopLimit', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint8_t ns3::Socket::GetIpv6Tclass() const [member function]
    cls.add_method('GetIpv6Tclass', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): ns3::Ptr<ns3::Node> ns3::Socket::GetNode() const [member function]
    cls.add_method('GetNode', 
                   'ns3::Ptr< ns3::Node >', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): int ns3::Socket::GetPeerName(ns3::Address & address) const [member function]
    cls.add_method('GetPeerName', 
                   'int', 
                   [param('ns3::Address &', 'address')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): uint8_t ns3::Socket::GetPriority() const [member function]
    cls.add_method('GetPriority', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): uint32_t ns3::Socket::GetRxAvailable() const [member function]
    cls.add_method('GetRxAvailable', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): int ns3::Socket::GetSockName(ns3::Address & address) const [member function]
    cls.add_method('GetSockName', 
                   'int', 
                   [param('ns3::Address &', 'address')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): ns3::Socket::SocketType ns3::Socket::GetSocketType() const [member function]
    cls.add_method('GetSocketType', 
                   'ns3::Socket::SocketType', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): uint32_t ns3::Socket::GetTxAvailable() const [member function]
    cls.add_method('GetTxAvailable', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): static ns3::TypeId ns3::Socket::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## socket.h (module 'network'): static uint8_t ns3::Socket::IpTos2Priority(uint8_t ipTos) [member function]
    cls.add_method('IpTos2Priority', 
                   'uint8_t', 
                   [param('uint8_t', 'ipTos')], 
                   is_static=True)
    ## socket.h (module 'network'): void ns3::Socket::Ipv6JoinGroup(ns3::Ipv6Address address, ns3::Socket::Ipv6MulticastFilterMode filterMode, std::vector<ns3::Ipv6Address, std::allocator<ns3::Ipv6Address>> sourceAddresses) [member function]
    cls.add_method('Ipv6JoinGroup', 
                   'void', 
                   [param('ns3::Ipv6Address', 'address'), param('ns3::Socket::Ipv6MulticastFilterMode', 'filterMode'), param('std::vector< ns3::Ipv6Address >', 'sourceAddresses')], 
                   is_virtual=True)
    ## socket.h (module 'network'): void ns3::Socket::Ipv6JoinGroup(ns3::Ipv6Address address) [member function]
    cls.add_method('Ipv6JoinGroup', 
                   'void', 
                   [param('ns3::Ipv6Address', 'address')], 
                   is_virtual=True)
    ## socket.h (module 'network'): void ns3::Socket::Ipv6LeaveGroup() [member function]
    cls.add_method('Ipv6LeaveGroup', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## socket.h (module 'network'): bool ns3::Socket::IsIpRecvTos() const [member function]
    cls.add_method('IsIpRecvTos', 
                   'bool', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): bool ns3::Socket::IsIpRecvTtl() const [member function]
    cls.add_method('IsIpRecvTtl', 
                   'bool', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): bool ns3::Socket::IsIpv6RecvHopLimit() const [member function]
    cls.add_method('IsIpv6RecvHopLimit', 
                   'bool', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): bool ns3::Socket::IsIpv6RecvTclass() const [member function]
    cls.add_method('IsIpv6RecvTclass', 
                   'bool', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): bool ns3::Socket::IsRecvPktInfo() const [member function]
    cls.add_method('IsRecvPktInfo', 
                   'bool', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): int ns3::Socket::Listen() [member function]
    cls.add_method('Listen', 
                   'int', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::Socket::Recv(uint32_t maxSize, uint32_t flags) [member function]
    cls.add_method('Recv', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('uint32_t', 'maxSize'), param('uint32_t', 'flags')], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::Socket::Recv() [member function]
    cls.add_method('Recv', 
                   'ns3::Ptr< ns3::Packet >', 
                   [])
    ## socket.h (module 'network'): int ns3::Socket::Recv(uint8_t * buf, uint32_t size, uint32_t flags) [member function]
    cls.add_method('Recv', 
                   'int', 
                   [param('uint8_t *', 'buf'), param('uint32_t', 'size'), param('uint32_t', 'flags')])
    ## socket.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::Socket::RecvFrom(uint32_t maxSize, uint32_t flags, ns3::Address & fromAddress) [member function]
    cls.add_method('RecvFrom', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('uint32_t', 'maxSize'), param('uint32_t', 'flags'), param('ns3::Address &', 'fromAddress')], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::Socket::RecvFrom(ns3::Address & fromAddress) [member function]
    cls.add_method('RecvFrom', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('ns3::Address &', 'fromAddress')])
    ## socket.h (module 'network'): int ns3::Socket::RecvFrom(uint8_t * buf, uint32_t size, uint32_t flags, ns3::Address & fromAddress) [member function]
    cls.add_method('RecvFrom', 
                   'int', 
                   [param('uint8_t *', 'buf'), param('uint32_t', 'size'), param('uint32_t', 'flags'), param('ns3::Address &', 'fromAddress')])
    ## socket.h (module 'network'): int ns3::Socket::Send(ns3::Ptr<ns3::Packet> p, uint32_t flags) [member function]
    cls.add_method('Send', 
                   'int', 
                   [param('ns3::Ptr< ns3::Packet >', 'p'), param('uint32_t', 'flags')], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): int ns3::Socket::Send(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('Send', 
                   'int', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')])
    ## socket.h (module 'network'): int ns3::Socket::Send(uint8_t const * buf, uint32_t size, uint32_t flags) [member function]
    cls.add_method('Send', 
                   'int', 
                   [param('uint8_t const *', 'buf'), param('uint32_t', 'size'), param('uint32_t', 'flags')])
    ## socket.h (module 'network'): int ns3::Socket::SendTo(ns3::Ptr<ns3::Packet> p, uint32_t flags, ns3::Address const & toAddress) [member function]
    cls.add_method('SendTo', 
                   'int', 
                   [param('ns3::Ptr< ns3::Packet >', 'p'), param('uint32_t', 'flags'), param('ns3::Address const &', 'toAddress')], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): int ns3::Socket::SendTo(uint8_t const * buf, uint32_t size, uint32_t flags, ns3::Address const & address) [member function]
    cls.add_method('SendTo', 
                   'int', 
                   [param('uint8_t const *', 'buf'), param('uint32_t', 'size'), param('uint32_t', 'flags'), param('ns3::Address const &', 'address')])
    ## socket.h (module 'network'): void ns3::Socket::SetAcceptCallback(ns3::Callback<bool, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> connectionRequest, ns3::Callback<void, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> newConnectionCreated) [member function]
    cls.add_method('SetAcceptCallback', 
                   'void', 
                   [param('ns3::Callback< bool, ns3::Ptr< ns3::Socket >, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'connectionRequest'), param('ns3::Callback< void, ns3::Ptr< ns3::Socket >, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'newConnectionCreated')])
    ## socket.h (module 'network'): bool ns3::Socket::SetAllowBroadcast(bool allowBroadcast) [member function]
    cls.add_method('SetAllowBroadcast', 
                   'bool', 
                   [param('bool', 'allowBroadcast')], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::Socket::SetCloseCallbacks(ns3::Callback<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> normalClose, ns3::Callback<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> errorClose) [member function]
    cls.add_method('SetCloseCallbacks', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::Socket >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'normalClose'), param('ns3::Callback< void, ns3::Ptr< ns3::Socket >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'errorClose')])
    ## socket.h (module 'network'): void ns3::Socket::SetConnectCallback(ns3::Callback<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> connectionSucceeded, ns3::Callback<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> connectionFailed) [member function]
    cls.add_method('SetConnectCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::Socket >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'connectionSucceeded'), param('ns3::Callback< void, ns3::Ptr< ns3::Socket >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'connectionFailed')])
    ## socket.h (module 'network'): void ns3::Socket::SetDataSentCallback(ns3::Callback<void, ns3::Ptr<ns3::Socket>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> dataSent) [member function]
    cls.add_method('SetDataSentCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::Socket >, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'dataSent')])
    ## socket.h (module 'network'): void ns3::Socket::SetIpRecvTos(bool ipv4RecvTos) [member function]
    cls.add_method('SetIpRecvTos', 
                   'void', 
                   [param('bool', 'ipv4RecvTos')])
    ## socket.h (module 'network'): void ns3::Socket::SetIpRecvTtl(bool ipv4RecvTtl) [member function]
    cls.add_method('SetIpRecvTtl', 
                   'void', 
                   [param('bool', 'ipv4RecvTtl')])
    ## socket.h (module 'network'): void ns3::Socket::SetIpTos(uint8_t ipTos) [member function]
    cls.add_method('SetIpTos', 
                   'void', 
                   [param('uint8_t', 'ipTos')])
    ## socket.h (module 'network'): void ns3::Socket::SetIpTtl(uint8_t ipTtl) [member function]
    cls.add_method('SetIpTtl', 
                   'void', 
                   [param('uint8_t', 'ipTtl')], 
                   is_virtual=True)
    ## socket.h (module 'network'): void ns3::Socket::SetIpv6HopLimit(uint8_t ipHopLimit) [member function]
    cls.add_method('SetIpv6HopLimit', 
                   'void', 
                   [param('uint8_t', 'ipHopLimit')], 
                   is_virtual=True)
    ## socket.h (module 'network'): void ns3::Socket::SetIpv6RecvHopLimit(bool ipv6RecvHopLimit) [member function]
    cls.add_method('SetIpv6RecvHopLimit', 
                   'void', 
                   [param('bool', 'ipv6RecvHopLimit')])
    ## socket.h (module 'network'): void ns3::Socket::SetIpv6RecvTclass(bool ipv6RecvTclass) [member function]
    cls.add_method('SetIpv6RecvTclass', 
                   'void', 
                   [param('bool', 'ipv6RecvTclass')])
    ## socket.h (module 'network'): void ns3::Socket::SetIpv6Tclass(int ipTclass) [member function]
    cls.add_method('SetIpv6Tclass', 
                   'void', 
                   [param('int', 'ipTclass')])
    ## socket.h (module 'network'): void ns3::Socket::SetPriority(uint8_t priority) [member function]
    cls.add_method('SetPriority', 
                   'void', 
                   [param('uint8_t', 'priority')])
    ## socket.h (module 'network'): void ns3::Socket::SetRecvCallback(ns3::Callback<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> receivedData) [member function]
    cls.add_method('SetRecvCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::Socket >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'receivedData')])
    ## socket.h (module 'network'): void ns3::Socket::SetRecvPktInfo(bool flag) [member function]
    cls.add_method('SetRecvPktInfo', 
                   'void', 
                   [param('bool', 'flag')])
    ## socket.h (module 'network'): void ns3::Socket::SetSendCallback(ns3::Callback<void, ns3::Ptr<ns3::Socket>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> sendCb) [member function]
    cls.add_method('SetSendCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::Ptr< ns3::Socket >, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'sendCb')])
    ## socket.h (module 'network'): int ns3::Socket::ShutdownRecv() [member function]
    cls.add_method('ShutdownRecv', 
                   'int', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): int ns3::Socket::ShutdownSend() [member function]
    cls.add_method('ShutdownSend', 
                   'int', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::Socket::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## socket.h (module 'network'): bool ns3::Socket::IsManualIpTtl() const [member function]
    cls.add_method('IsManualIpTtl', 
                   'bool', 
                   [], 
                   is_const=True, visibility='protected')
    ## socket.h (module 'network'): bool ns3::Socket::IsManualIpv6HopLimit() const [member function]
    cls.add_method('IsManualIpv6HopLimit', 
                   'bool', 
                   [], 
                   is_const=True, visibility='protected')
    ## socket.h (module 'network'): bool ns3::Socket::IsManualIpv6Tclass() const [member function]
    cls.add_method('IsManualIpv6Tclass', 
                   'bool', 
                   [], 
                   is_const=True, visibility='protected')
    ## socket.h (module 'network'): void ns3::Socket::NotifyConnectionFailed() [member function]
    cls.add_method('NotifyConnectionFailed', 
                   'void', 
                   [], 
                   visibility='protected')
    ## socket.h (module 'network'): bool ns3::Socket::NotifyConnectionRequest(ns3::Address const & from) [member function]
    cls.add_method('NotifyConnectionRequest', 
                   'bool', 
                   [param('ns3::Address const &', 'from')], 
                   visibility='protected')
    ## socket.h (module 'network'): void ns3::Socket::NotifyConnectionSucceeded() [member function]
    cls.add_method('NotifyConnectionSucceeded', 
                   'void', 
                   [], 
                   visibility='protected')
    ## socket.h (module 'network'): void ns3::Socket::NotifyDataRecv() [member function]
    cls.add_method('NotifyDataRecv', 
                   'void', 
                   [], 
                   visibility='protected')
    ## socket.h (module 'network'): void ns3::Socket::NotifyDataSent(uint32_t size) [member function]
    cls.add_method('NotifyDataSent', 
                   'void', 
                   [param('uint32_t', 'size')], 
                   visibility='protected')
    ## socket.h (module 'network'): void ns3::Socket::NotifyErrorClose() [member function]
    cls.add_method('NotifyErrorClose', 
                   'void', 
                   [], 
                   visibility='protected')
    ## socket.h (module 'network'): void ns3::Socket::NotifyNewConnectionCreated(ns3::Ptr<ns3::Socket> socket, ns3::Address const & from) [member function]
    cls.add_method('NotifyNewConnectionCreated', 
                   'void', 
                   [param('ns3::Ptr< ns3::Socket >', 'socket'), param('ns3::Address const &', 'from')], 
                   visibility='protected')
    ## socket.h (module 'network'): void ns3::Socket::NotifyNormalClose() [member function]
    cls.add_method('NotifyNormalClose', 
                   'void', 
                   [], 
                   visibility='protected')
    ## socket.h (module 'network'): void ns3::Socket::NotifySend(uint32_t spaceAvailable) [member function]
    cls.add_method('NotifySend', 
                   'void', 
                   [param('uint32_t', 'spaceAvailable')], 
                   visibility='protected')
    return

def register_Ns3SocketFactory_methods(root_module, cls):
    ## socket-factory.h (module 'network'): ns3::SocketFactory::SocketFactory(ns3::SocketFactory const & arg0) [constructor]
    cls.add_constructor([param('ns3::SocketFactory const &', 'arg0')])
    ## socket-factory.h (module 'network'): ns3::SocketFactory::SocketFactory() [constructor]
    cls.add_constructor([])
    ## socket-factory.h (module 'network'): ns3::Ptr<ns3::Socket> ns3::SocketFactory::CreateSocket() [member function]
    cls.add_method('CreateSocket', 
                   'ns3::Ptr< ns3::Socket >', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## socket-factory.h (module 'network'): static ns3::TypeId ns3::SocketFactory::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    return

def register_Ns3SocketIpTosTag_methods(root_module, cls):
    ## socket.h (module 'network'): ns3::SocketIpTosTag::SocketIpTosTag(ns3::SocketIpTosTag const & arg0) [constructor]
    cls.add_constructor([param('ns3::SocketIpTosTag const &', 'arg0')])
    ## socket.h (module 'network'): ns3::SocketIpTosTag::SocketIpTosTag() [constructor]
    cls.add_constructor([])
    ## socket.h (module 'network'): void ns3::SocketIpTosTag::Deserialize(ns3::TagBuffer i) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_virtual=True)
    ## socket.h (module 'network'): ns3::TypeId ns3::SocketIpTosTag::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint32_t ns3::SocketIpTosTag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint8_t ns3::SocketIpTosTag::GetTos() const [member function]
    cls.add_method('GetTos', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): static ns3::TypeId ns3::SocketIpTosTag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## socket.h (module 'network'): void ns3::SocketIpTosTag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketIpTosTag::Serialize(ns3::TagBuffer i) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketIpTosTag::SetTos(uint8_t tos) [member function]
    cls.add_method('SetTos', 
                   'void', 
                   [param('uint8_t', 'tos')])
    return

def register_Ns3SocketIpTtlTag_methods(root_module, cls):
    ## socket.h (module 'network'): ns3::SocketIpTtlTag::SocketIpTtlTag(ns3::SocketIpTtlTag const & arg0) [constructor]
    cls.add_constructor([param('ns3::SocketIpTtlTag const &', 'arg0')])
    ## socket.h (module 'network'): ns3::SocketIpTtlTag::SocketIpTtlTag() [constructor]
    cls.add_constructor([])
    ## socket.h (module 'network'): void ns3::SocketIpTtlTag::Deserialize(ns3::TagBuffer i) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_virtual=True)
    ## socket.h (module 'network'): ns3::TypeId ns3::SocketIpTtlTag::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint32_t ns3::SocketIpTtlTag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint8_t ns3::SocketIpTtlTag::GetTtl() const [member function]
    cls.add_method('GetTtl', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): static ns3::TypeId ns3::SocketIpTtlTag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## socket.h (module 'network'): void ns3::SocketIpTtlTag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketIpTtlTag::Serialize(ns3::TagBuffer i) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketIpTtlTag::SetTtl(uint8_t ttl) [member function]
    cls.add_method('SetTtl', 
                   'void', 
                   [param('uint8_t', 'ttl')])
    return

def register_Ns3SocketIpv6HopLimitTag_methods(root_module, cls):
    ## socket.h (module 'network'): ns3::SocketIpv6HopLimitTag::SocketIpv6HopLimitTag(ns3::SocketIpv6HopLimitTag const & arg0) [constructor]
    cls.add_constructor([param('ns3::SocketIpv6HopLimitTag const &', 'arg0')])
    ## socket.h (module 'network'): ns3::SocketIpv6HopLimitTag::SocketIpv6HopLimitTag() [constructor]
    cls.add_constructor([])
    ## socket.h (module 'network'): void ns3::SocketIpv6HopLimitTag::Deserialize(ns3::TagBuffer i) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_virtual=True)
    ## socket.h (module 'network'): uint8_t ns3::SocketIpv6HopLimitTag::GetHopLimit() const [member function]
    cls.add_method('GetHopLimit', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): ns3::TypeId ns3::SocketIpv6HopLimitTag::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint32_t ns3::SocketIpv6HopLimitTag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): static ns3::TypeId ns3::SocketIpv6HopLimitTag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## socket.h (module 'network'): void ns3::SocketIpv6HopLimitTag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketIpv6HopLimitTag::Serialize(ns3::TagBuffer i) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketIpv6HopLimitTag::SetHopLimit(uint8_t hopLimit) [member function]
    cls.add_method('SetHopLimit', 
                   'void', 
                   [param('uint8_t', 'hopLimit')])
    return

def register_Ns3SocketIpv6TclassTag_methods(root_module, cls):
    ## socket.h (module 'network'): ns3::SocketIpv6TclassTag::SocketIpv6TclassTag(ns3::SocketIpv6TclassTag const & arg0) [constructor]
    cls.add_constructor([param('ns3::SocketIpv6TclassTag const &', 'arg0')])
    ## socket.h (module 'network'): ns3::SocketIpv6TclassTag::SocketIpv6TclassTag() [constructor]
    cls.add_constructor([])
    ## socket.h (module 'network'): void ns3::SocketIpv6TclassTag::Deserialize(ns3::TagBuffer i) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_virtual=True)
    ## socket.h (module 'network'): ns3::TypeId ns3::SocketIpv6TclassTag::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint32_t ns3::SocketIpv6TclassTag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint8_t ns3::SocketIpv6TclassTag::GetTclass() const [member function]
    cls.add_method('GetTclass', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): static ns3::TypeId ns3::SocketIpv6TclassTag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## socket.h (module 'network'): void ns3::SocketIpv6TclassTag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketIpv6TclassTag::Serialize(ns3::TagBuffer i) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketIpv6TclassTag::SetTclass(uint8_t tclass) [member function]
    cls.add_method('SetTclass', 
                   'void', 
                   [param('uint8_t', 'tclass')])
    return

def register_Ns3SocketPriorityTag_methods(root_module, cls):
    ## socket.h (module 'network'): ns3::SocketPriorityTag::SocketPriorityTag(ns3::SocketPriorityTag const & arg0) [constructor]
    cls.add_constructor([param('ns3::SocketPriorityTag const &', 'arg0')])
    ## socket.h (module 'network'): ns3::SocketPriorityTag::SocketPriorityTag() [constructor]
    cls.add_constructor([])
    ## socket.h (module 'network'): void ns3::SocketPriorityTag::Deserialize(ns3::TagBuffer i) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_virtual=True)
    ## socket.h (module 'network'): ns3::TypeId ns3::SocketPriorityTag::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint8_t ns3::SocketPriorityTag::GetPriority() const [member function]
    cls.add_method('GetPriority', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): uint32_t ns3::SocketPriorityTag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): static ns3::TypeId ns3::SocketPriorityTag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## socket.h (module 'network'): void ns3::SocketPriorityTag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketPriorityTag::Serialize(ns3::TagBuffer i) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketPriorityTag::SetPriority(uint8_t priority) [member function]
    cls.add_method('SetPriority', 
                   'void', 
                   [param('uint8_t', 'priority')])
    return

def register_Ns3SocketSetDontFragmentTag_methods(root_module, cls):
    ## socket.h (module 'network'): ns3::SocketSetDontFragmentTag::SocketSetDontFragmentTag(ns3::SocketSetDontFragmentTag const & arg0) [constructor]
    cls.add_constructor([param('ns3::SocketSetDontFragmentTag const &', 'arg0')])
    ## socket.h (module 'network'): ns3::SocketSetDontFragmentTag::SocketSetDontFragmentTag() [constructor]
    cls.add_constructor([])
    ## socket.h (module 'network'): void ns3::SocketSetDontFragmentTag::Deserialize(ns3::TagBuffer i) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
                   is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketSetDontFragmentTag::Disable() [member function]
    cls.add_method('Disable', 
                   'void', 
                   [])
    ## socket.h (module 'network'): void ns3::SocketSetDontFragmentTag::Enable() [member function]
    cls.add_method('Enable', 
                   'void', 
                   [])
    ## socket.h (module 'network'): ns3::TypeId ns3::SocketSetDontFragmentTag::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): uint32_t ns3::SocketSetDontFragmentTag::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): static ns3::TypeId ns3::SocketSetDontFragmentTag::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## socket.h (module 'network'): bool ns3::SocketSetDontFragmentTag::IsEnabled() const [member function]
    cls.add_method('IsEnabled', 
                   'bool', 
                   [], 
                   is_const=True)
    ## socket.h (module 'network'): void ns3::SocketSetDontFragmentTag::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## socket.h (module 'network'): void ns3::SocketSetDontFragmentTag::Serialize(ns3::TagBuffer i) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::TagBuffer', 'i')], 
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

def register_Ns3Application_methods(root_module, cls):
    ## application.h (module 'network'): ns3::Application::Application(ns3::Application const & arg0) [constructor]
    cls.add_constructor([param('ns3::Application const &', 'arg0')])
    ## application.h (module 'network'): ns3::Application::Application() [constructor]
    cls.add_constructor([])
    ## application.h (module 'network'): ns3::Ptr<ns3::Node> ns3::Application::GetNode() const [member function]
    cls.add_method('GetNode', 
                   'ns3::Ptr< ns3::Node >', 
                   [], 
                   is_const=True)
    ## application.h (module 'network'): static ns3::TypeId ns3::Application::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## application.h (module 'network'): void ns3::Application::SetNode(ns3::Ptr<ns3::Node> node) [member function]
    cls.add_method('SetNode', 
                   'void', 
                   [param('ns3::Ptr< ns3::Node >', 'node')])
    ## application.h (module 'network'): void ns3::Application::SetStartTime(ns3::Time start) [member function]
    cls.add_method('SetStartTime', 
                   'void', 
                   [param('ns3::Time', 'start')])
    ## application.h (module 'network'): void ns3::Application::SetStopTime(ns3::Time stop) [member function]
    cls.add_method('SetStopTime', 
                   'void', 
                   [param('ns3::Time', 'stop')])
    ## application.h (module 'network'): void ns3::Application::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## application.h (module 'network'): void ns3::Application::DoInitialize() [member function]
    cls.add_method('DoInitialize', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## application.h (module 'network'): void ns3::Application::StartApplication() [member function]
    cls.add_method('StartApplication', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    ## application.h (module 'network'): void ns3::Application::StopApplication() [member function]
    cls.add_method('StopApplication', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
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
                   is_static=True, template_parameters=['unsigned int'], visibility='protected')
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
                   is_static=True, template_parameters=['ns3::Ptr<ns3::Socket> '], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['bool'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['ns3::Ptr<ns3::QueueDiscItem const> '], visibility='protected')
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

def register_Ns3DataCalculator_methods(root_module, cls):
    ## data-calculator.h (module 'stats'): ns3::DataCalculator::DataCalculator(ns3::DataCalculator const & arg0) [constructor]
    cls.add_constructor([param('ns3::DataCalculator const &', 'arg0')])
    ## data-calculator.h (module 'stats'): ns3::DataCalculator::DataCalculator() [constructor]
    cls.add_constructor([])
    ## data-calculator.h (module 'stats'): void ns3::DataCalculator::Disable() [member function]
    cls.add_method('Disable', 
                   'void', 
                   [])
    ## data-calculator.h (module 'stats'): void ns3::DataCalculator::Enable() [member function]
    cls.add_method('Enable', 
                   'void', 
                   [])
    ## data-calculator.h (module 'stats'): std::string ns3::DataCalculator::GetContext() const [member function]
    cls.add_method('GetContext', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## data-calculator.h (module 'stats'): bool ns3::DataCalculator::GetEnabled() const [member function]
    cls.add_method('GetEnabled', 
                   'bool', 
                   [], 
                   is_const=True)
    ## data-calculator.h (module 'stats'): std::string ns3::DataCalculator::GetKey() const [member function]
    cls.add_method('GetKey', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## data-calculator.h (module 'stats'): static ns3::TypeId ns3::DataCalculator::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## data-calculator.h (module 'stats'): void ns3::DataCalculator::Output(ns3::DataOutputCallback & callback) const [member function]
    cls.add_method('Output', 
                   'void', 
                   [param('ns3::DataOutputCallback &', 'callback')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## data-calculator.h (module 'stats'): void ns3::DataCalculator::SetContext(std::string const context) [member function]
    cls.add_method('SetContext', 
                   'void', 
                   [param('std::string const', 'context')])
    ## data-calculator.h (module 'stats'): void ns3::DataCalculator::SetKey(std::string const key) [member function]
    cls.add_method('SetKey', 
                   'void', 
                   [param('std::string const', 'key')])
    ## data-calculator.h (module 'stats'): void ns3::DataCalculator::Start(ns3::Time const & startTime) [member function]
    cls.add_method('Start', 
                   'void', 
                   [param('ns3::Time const &', 'startTime')], 
                   is_virtual=True)
    ## data-calculator.h (module 'stats'): void ns3::DataCalculator::Stop(ns3::Time const & stopTime) [member function]
    cls.add_method('Stop', 
                   'void', 
                   [param('ns3::Time const &', 'stopTime')], 
                   is_virtual=True)
    ## data-calculator.h (module 'stats'): void ns3::DataCalculator::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3DataCollectionObject_methods(root_module, cls):
    ## data-collection-object.h (module 'stats'): ns3::DataCollectionObject::DataCollectionObject(ns3::DataCollectionObject const & arg0) [constructor]
    cls.add_constructor([param('ns3::DataCollectionObject const &', 'arg0')])
    ## data-collection-object.h (module 'stats'): ns3::DataCollectionObject::DataCollectionObject() [constructor]
    cls.add_constructor([])
    ## data-collection-object.h (module 'stats'): void ns3::DataCollectionObject::Disable() [member function]
    cls.add_method('Disable', 
                   'void', 
                   [])
    ## data-collection-object.h (module 'stats'): void ns3::DataCollectionObject::Enable() [member function]
    cls.add_method('Enable', 
                   'void', 
                   [])
    ## data-collection-object.h (module 'stats'): std::string ns3::DataCollectionObject::GetName() const [member function]
    cls.add_method('GetName', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## data-collection-object.h (module 'stats'): static ns3::TypeId ns3::DataCollectionObject::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## data-collection-object.h (module 'stats'): bool ns3::DataCollectionObject::IsEnabled() const [member function]
    cls.add_method('IsEnabled', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## data-collection-object.h (module 'stats'): void ns3::DataCollectionObject::SetName(std::string name) [member function]
    cls.add_method('SetName', 
                   'void', 
                   [param('std::string', 'name')])
    return

def register_Ns3DataOutputInterface_methods(root_module, cls):
    ## data-output-interface.h (module 'stats'): ns3::DataOutputInterface::DataOutputInterface(ns3::DataOutputInterface const & arg0) [constructor]
    cls.add_constructor([param('ns3::DataOutputInterface const &', 'arg0')])
    ## data-output-interface.h (module 'stats'): ns3::DataOutputInterface::DataOutputInterface() [constructor]
    cls.add_constructor([])
    ## data-output-interface.h (module 'stats'): std::string ns3::DataOutputInterface::GetFilePrefix() const [member function]
    cls.add_method('GetFilePrefix', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## data-output-interface.h (module 'stats'): static ns3::TypeId ns3::DataOutputInterface::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## data-output-interface.h (module 'stats'): void ns3::DataOutputInterface::Output(ns3::DataCollector & dc) [member function]
    cls.add_method('Output', 
                   'void', 
                   [param('ns3::DataCollector &', 'dc')], 
                   is_pure_virtual=True, is_virtual=True)
    ## data-output-interface.h (module 'stats'): void ns3::DataOutputInterface::SetFilePrefix(std::string const prefix) [member function]
    cls.add_method('SetFilePrefix', 
                   'void', 
                   [param('std::string const', 'prefix')])
    ## data-output-interface.h (module 'stats'): void ns3::DataOutputInterface::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
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

def register_Ns3DynamicQueueLimits_methods(root_module, cls):
    ## dynamic-queue-limits.h (module 'network'): ns3::DynamicQueueLimits::DynamicQueueLimits(ns3::DynamicQueueLimits const & arg0) [constructor]
    cls.add_constructor([param('ns3::DynamicQueueLimits const &', 'arg0')])
    ## dynamic-queue-limits.h (module 'network'): ns3::DynamicQueueLimits::DynamicQueueLimits() [constructor]
    cls.add_constructor([])
    ## dynamic-queue-limits.h (module 'network'): int32_t ns3::DynamicQueueLimits::Available() const [member function]
    cls.add_method('Available', 
                   'int32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## dynamic-queue-limits.h (module 'network'): void ns3::DynamicQueueLimits::Completed(uint32_t count) [member function]
    cls.add_method('Completed', 
                   'void', 
                   [param('uint32_t', 'count')], 
                   is_virtual=True)
    ## dynamic-queue-limits.h (module 'network'): static ns3::TypeId ns3::DynamicQueueLimits::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## dynamic-queue-limits.h (module 'network'): void ns3::DynamicQueueLimits::Queued(uint32_t count) [member function]
    cls.add_method('Queued', 
                   'void', 
                   [param('uint32_t', 'count')], 
                   is_virtual=True)
    ## dynamic-queue-limits.h (module 'network'): void ns3::DynamicQueueLimits::Reset() [member function]
    cls.add_method('Reset', 
                   'void', 
                   [], 
                   is_virtual=True)
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

def register_Ns3ErrorModel_methods(root_module, cls):
    ## error-model.h (module 'network'): ns3::ErrorModel::ErrorModel(ns3::ErrorModel const & arg0) [constructor]
    cls.add_constructor([param('ns3::ErrorModel const &', 'arg0')])
    ## error-model.h (module 'network'): ns3::ErrorModel::ErrorModel() [constructor]
    cls.add_constructor([])
    ## error-model.h (module 'network'): void ns3::ErrorModel::Disable() [member function]
    cls.add_method('Disable', 
                   'void', 
                   [])
    ## error-model.h (module 'network'): void ns3::ErrorModel::Enable() [member function]
    cls.add_method('Enable', 
                   'void', 
                   [])
    ## error-model.h (module 'network'): static ns3::TypeId ns3::ErrorModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## error-model.h (module 'network'): bool ns3::ErrorModel::IsCorrupt(ns3::Ptr<ns3::Packet> pkt) [member function]
    cls.add_method('IsCorrupt', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'pkt')])
    ## error-model.h (module 'network'): bool ns3::ErrorModel::IsEnabled() const [member function]
    cls.add_method('IsEnabled', 
                   'bool', 
                   [], 
                   is_const=True)
    ## error-model.h (module 'network'): void ns3::ErrorModel::Reset() [member function]
    cls.add_method('Reset', 
                   'void', 
                   [])
    ## error-model.h (module 'network'): bool ns3::ErrorModel::DoCorrupt(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('DoCorrupt', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    ## error-model.h (module 'network'): void ns3::ErrorModel::DoReset() [member function]
    cls.add_method('DoReset', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True, visibility='private')
    return

def register_Ns3EthernetHeader_methods(root_module, cls):
    ## ethernet-header.h (module 'network'): ns3::EthernetHeader::EthernetHeader(ns3::EthernetHeader const & arg0) [constructor]
    cls.add_constructor([param('ns3::EthernetHeader const &', 'arg0')])
    ## ethernet-header.h (module 'network'): ns3::EthernetHeader::EthernetHeader(bool hasPreamble) [constructor]
    cls.add_constructor([param('bool', 'hasPreamble')])
    ## ethernet-header.h (module 'network'): ns3::EthernetHeader::EthernetHeader() [constructor]
    cls.add_constructor([])
    ## ethernet-header.h (module 'network'): uint32_t ns3::EthernetHeader::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## ethernet-header.h (module 'network'): ns3::Mac48Address ns3::EthernetHeader::GetDestination() const [member function]
    cls.add_method('GetDestination', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## ethernet-header.h (module 'network'): uint32_t ns3::EthernetHeader::GetHeaderSize() const [member function]
    cls.add_method('GetHeaderSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ethernet-header.h (module 'network'): ns3::TypeId ns3::EthernetHeader::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ethernet-header.h (module 'network'): uint16_t ns3::EthernetHeader::GetLengthType() const [member function]
    cls.add_method('GetLengthType', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## ethernet-header.h (module 'network'): ns3::ethernet_header_t ns3::EthernetHeader::GetPacketType() const [member function]
    cls.add_method('GetPacketType', 
                   'ns3::ethernet_header_t', 
                   [], 
                   is_const=True)
    ## ethernet-header.h (module 'network'): uint64_t ns3::EthernetHeader::GetPreambleSfd() const [member function]
    cls.add_method('GetPreambleSfd', 
                   'uint64_t', 
                   [], 
                   is_const=True)
    ## ethernet-header.h (module 'network'): uint32_t ns3::EthernetHeader::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ethernet-header.h (module 'network'): ns3::Mac48Address ns3::EthernetHeader::GetSource() const [member function]
    cls.add_method('GetSource', 
                   'ns3::Mac48Address', 
                   [], 
                   is_const=True)
    ## ethernet-header.h (module 'network'): static ns3::TypeId ns3::EthernetHeader::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## ethernet-header.h (module 'network'): void ns3::EthernetHeader::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ethernet-header.h (module 'network'): void ns3::EthernetHeader::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## ethernet-header.h (module 'network'): void ns3::EthernetHeader::SetDestination(ns3::Mac48Address destination) [member function]
    cls.add_method('SetDestination', 
                   'void', 
                   [param('ns3::Mac48Address', 'destination')])
    ## ethernet-header.h (module 'network'): void ns3::EthernetHeader::SetLengthType(uint16_t size) [member function]
    cls.add_method('SetLengthType', 
                   'void', 
                   [param('uint16_t', 'size')])
    ## ethernet-header.h (module 'network'): void ns3::EthernetHeader::SetPreambleSfd(uint64_t preambleSfd) [member function]
    cls.add_method('SetPreambleSfd', 
                   'void', 
                   [param('uint64_t', 'preambleSfd')])
    ## ethernet-header.h (module 'network'): void ns3::EthernetHeader::SetSource(ns3::Mac48Address source) [member function]
    cls.add_method('SetSource', 
                   'void', 
                   [param('ns3::Mac48Address', 'source')])
    return

def register_Ns3EthernetTrailer_methods(root_module, cls):
    ## ethernet-trailer.h (module 'network'): ns3::EthernetTrailer::EthernetTrailer(ns3::EthernetTrailer const & arg0) [constructor]
    cls.add_constructor([param('ns3::EthernetTrailer const &', 'arg0')])
    ## ethernet-trailer.h (module 'network'): ns3::EthernetTrailer::EthernetTrailer() [constructor]
    cls.add_constructor([])
    ## ethernet-trailer.h (module 'network'): void ns3::EthernetTrailer::CalcFcs(ns3::Ptr<const ns3::Packet> p) [member function]
    cls.add_method('CalcFcs', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'p')])
    ## ethernet-trailer.h (module 'network'): bool ns3::EthernetTrailer::CheckFcs(ns3::Ptr<const ns3::Packet> p) const [member function]
    cls.add_method('CheckFcs', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet const >', 'p')], 
                   is_const=True)
    ## ethernet-trailer.h (module 'network'): uint32_t ns3::EthernetTrailer::Deserialize(ns3::Buffer::Iterator end) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'end')], 
                   is_virtual=True)
    ## ethernet-trailer.h (module 'network'): void ns3::EthernetTrailer::EnableFcs(bool enable) [member function]
    cls.add_method('EnableFcs', 
                   'void', 
                   [param('bool', 'enable')])
    ## ethernet-trailer.h (module 'network'): uint32_t ns3::EthernetTrailer::GetFcs() const [member function]
    cls.add_method('GetFcs', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ethernet-trailer.h (module 'network'): ns3::TypeId ns3::EthernetTrailer::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ethernet-trailer.h (module 'network'): uint32_t ns3::EthernetTrailer::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## ethernet-trailer.h (module 'network'): uint32_t ns3::EthernetTrailer::GetTrailerSize() const [member function]
    cls.add_method('GetTrailerSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## ethernet-trailer.h (module 'network'): static ns3::TypeId ns3::EthernetTrailer::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## ethernet-trailer.h (module 'network'): void ns3::EthernetTrailer::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## ethernet-trailer.h (module 'network'): void ns3::EthernetTrailer::Serialize(ns3::Buffer::Iterator end) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'end')], 
                   is_const=True, is_virtual=True)
    ## ethernet-trailer.h (module 'network'): void ns3::EthernetTrailer::SetFcs(uint32_t fcs) [member function]
    cls.add_method('SetFcs', 
                   'void', 
                   [param('uint32_t', 'fcs')])
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

def register_Ns3ListErrorModel_methods(root_module, cls):
    ## error-model.h (module 'network'): ns3::ListErrorModel::ListErrorModel(ns3::ListErrorModel const & arg0) [constructor]
    cls.add_constructor([param('ns3::ListErrorModel const &', 'arg0')])
    ## error-model.h (module 'network'): ns3::ListErrorModel::ListErrorModel() [constructor]
    cls.add_constructor([])
    ## error-model.h (module 'network'): std::list<unsigned int, std::allocator<unsigned int>> ns3::ListErrorModel::GetList() const [member function]
    cls.add_method('GetList', 
                   'std::list< unsigned int >', 
                   [], 
                   is_const=True)
    ## error-model.h (module 'network'): static ns3::TypeId ns3::ListErrorModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## error-model.h (module 'network'): void ns3::ListErrorModel::SetList(std::list<unsigned int, std::allocator<unsigned int>> const & packetlist) [member function]
    cls.add_method('SetList', 
                   'void', 
                   [param('std::list< unsigned int > const &', 'packetlist')])
    ## error-model.h (module 'network'): bool ns3::ListErrorModel::DoCorrupt(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('DoCorrupt', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')], 
                   is_virtual=True, visibility='private')
    ## error-model.h (module 'network'): void ns3::ListErrorModel::DoReset() [member function]
    cls.add_method('DoReset', 
                   'void', 
                   [], 
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

def register_Ns3Mac16AddressChecker_methods(root_module, cls):
    ## mac16-address.h (module 'network'): ns3::Mac16AddressChecker::Mac16AddressChecker() [constructor]
    cls.add_constructor([])
    ## mac16-address.h (module 'network'): ns3::Mac16AddressChecker::Mac16AddressChecker(ns3::Mac16AddressChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::Mac16AddressChecker const &', 'arg0')])
    return

def register_Ns3Mac16AddressValue_methods(root_module, cls):
    ## mac16-address.h (module 'network'): ns3::Mac16AddressValue::Mac16AddressValue() [constructor]
    cls.add_constructor([])
    ## mac16-address.h (module 'network'): ns3::Mac16AddressValue::Mac16AddressValue(ns3::Mac16Address const & value) [constructor]
    cls.add_constructor([param('ns3::Mac16Address const &', 'value')])
    ## mac16-address.h (module 'network'): ns3::Mac16AddressValue::Mac16AddressValue(ns3::Mac16AddressValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::Mac16AddressValue const &', 'arg0')])
    ## mac16-address.h (module 'network'): ns3::Ptr<ns3::AttributeValue> ns3::Mac16AddressValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac16-address.h (module 'network'): bool ns3::Mac16AddressValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## mac16-address.h (module 'network'): ns3::Mac16Address ns3::Mac16AddressValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::Mac16Address', 
                   [], 
                   is_const=True)
    ## mac16-address.h (module 'network'): std::string ns3::Mac16AddressValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## mac16-address.h (module 'network'): void ns3::Mac16AddressValue::Set(ns3::Mac16Address const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::Mac16Address const &', 'value')])
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

def register_Ns3Mac64AddressChecker_methods(root_module, cls):
    ## mac64-address.h (module 'network'): ns3::Mac64AddressChecker::Mac64AddressChecker() [constructor]
    cls.add_constructor([])
    ## mac64-address.h (module 'network'): ns3::Mac64AddressChecker::Mac64AddressChecker(ns3::Mac64AddressChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::Mac64AddressChecker const &', 'arg0')])
    return

def register_Ns3Mac64AddressValue_methods(root_module, cls):
    ## mac64-address.h (module 'network'): ns3::Mac64AddressValue::Mac64AddressValue() [constructor]
    cls.add_constructor([])
    ## mac64-address.h (module 'network'): ns3::Mac64AddressValue::Mac64AddressValue(ns3::Mac64Address const & value) [constructor]
    cls.add_constructor([param('ns3::Mac64Address const &', 'value')])
    ## mac64-address.h (module 'network'): ns3::Mac64AddressValue::Mac64AddressValue(ns3::Mac64AddressValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::Mac64AddressValue const &', 'arg0')])
    ## mac64-address.h (module 'network'): ns3::Ptr<ns3::AttributeValue> ns3::Mac64AddressValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## mac64-address.h (module 'network'): bool ns3::Mac64AddressValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## mac64-address.h (module 'network'): ns3::Mac64Address ns3::Mac64AddressValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::Mac64Address', 
                   [], 
                   is_const=True)
    ## mac64-address.h (module 'network'): std::string ns3::Mac64AddressValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## mac64-address.h (module 'network'): void ns3::Mac64AddressValue::Set(ns3::Mac64Address const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::Mac64Address const &', 'value')])
    return

def register_Ns3MinMaxAvgTotalCalculator__Unsigned_int_methods(root_module, cls):
    ## basic-data-calculators.h (module 'stats'): ns3::MinMaxAvgTotalCalculator<unsigned int>::MinMaxAvgTotalCalculator(ns3::MinMaxAvgTotalCalculator<unsigned int> const & arg0) [constructor]
    cls.add_constructor([param('ns3::MinMaxAvgTotalCalculator< unsigned int > const &', 'arg0')])
    ## basic-data-calculators.h (module 'stats'): ns3::MinMaxAvgTotalCalculator<unsigned int>::MinMaxAvgTotalCalculator() [constructor]
    cls.add_constructor([])
    ## basic-data-calculators.h (module 'stats'): static ns3::TypeId ns3::MinMaxAvgTotalCalculator<unsigned int>::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## basic-data-calculators.h (module 'stats'): void ns3::MinMaxAvgTotalCalculator<unsigned int>::Output(ns3::DataOutputCallback & callback) const [member function]
    cls.add_method('Output', 
                   'void', 
                   [param('ns3::DataOutputCallback &', 'callback')], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): void ns3::MinMaxAvgTotalCalculator<unsigned int>::Reset() [member function]
    cls.add_method('Reset', 
                   'void', 
                   [])
    ## basic-data-calculators.h (module 'stats'): void ns3::MinMaxAvgTotalCalculator<unsigned int>::Update(unsigned int const i) [member function]
    cls.add_method('Update', 
                   'void', 
                   [param('unsigned int const', 'i')])
    ## basic-data-calculators.h (module 'stats'): long int ns3::MinMaxAvgTotalCalculator<unsigned int>::getCount() const [member function]
    cls.add_method('getCount', 
                   'long int', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<unsigned int>::getMax() const [member function]
    cls.add_method('getMax', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<unsigned int>::getMean() const [member function]
    cls.add_method('getMean', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<unsigned int>::getMin() const [member function]
    cls.add_method('getMin', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<unsigned int>::getSqrSum() const [member function]
    cls.add_method('getSqrSum', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<unsigned int>::getStddev() const [member function]
    cls.add_method('getStddev', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<unsigned int>::getSum() const [member function]
    cls.add_method('getSum', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<unsigned int>::getVariance() const [member function]
    cls.add_method('getVariance', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): void ns3::MinMaxAvgTotalCalculator<unsigned int>::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
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

def register_Ns3NetDeviceQueue_methods(root_module, cls):
    ## net-device-queue-interface.h (module 'network'): static ns3::TypeId ns3::NetDeviceQueue::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## net-device-queue-interface.h (module 'network'): ns3::NetDeviceQueue::NetDeviceQueue() [constructor]
    cls.add_constructor([])
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueue::Start() [member function]
    cls.add_method('Start', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueue::Stop() [member function]
    cls.add_method('Stop', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueue::Wake() [member function]
    cls.add_method('Wake', 
                   'void', 
                   [], 
                   is_virtual=True)
    ## net-device-queue-interface.h (module 'network'): bool ns3::NetDeviceQueue::IsStopped() const [member function]
    cls.add_method('IsStopped', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueue::NotifyAggregatedObject(ns3::Ptr<ns3::NetDeviceQueueInterface> ndqi) [member function]
    cls.add_method('NotifyAggregatedObject', 
                   'void', 
                   [param('ns3::Ptr< ns3::NetDeviceQueueInterface >', 'ndqi')])
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueue::SetWakeCallback(ns3::NetDeviceQueue::WakeCallback cb) [member function]
    cls.add_method('SetWakeCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')], 
                   is_virtual=True)
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueue::NotifyQueuedBytes(uint32_t bytes) [member function]
    cls.add_method('NotifyQueuedBytes', 
                   'void', 
                   [param('uint32_t', 'bytes')], 
                   is_virtual=True)
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueue::NotifyTransmittedBytes(uint32_t bytes) [member function]
    cls.add_method('NotifyTransmittedBytes', 
                   'void', 
                   [param('uint32_t', 'bytes')], 
                   is_virtual=True)
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueue::ResetQueueLimits() [member function]
    cls.add_method('ResetQueueLimits', 
                   'void', 
                   [])
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueue::SetQueueLimits(ns3::Ptr<ns3::QueueLimits> ql) [member function]
    cls.add_method('SetQueueLimits', 
                   'void', 
                   [param('ns3::Ptr< ns3::QueueLimits >', 'ql')])
    ## net-device-queue-interface.h (module 'network'): ns3::Ptr<ns3::QueueLimits> ns3::NetDeviceQueue::GetQueueLimits() [member function]
    cls.add_method('GetQueueLimits', 
                   'ns3::Ptr< ns3::QueueLimits >', 
                   [])
    ## net-device-queue-interface.h (module 'network'): ns3::NetDeviceQueue::NetDeviceQueue(ns3::NetDeviceQueue const & arg0) [constructor]
    cls.add_constructor([param('ns3::NetDeviceQueue const &', 'arg0')])
    return

def register_Ns3NetDeviceQueueInterface_methods(root_module, cls):
    ## net-device-queue-interface.h (module 'network'): ns3::NetDeviceQueueInterface::NetDeviceQueueInterface(ns3::NetDeviceQueueInterface const & arg0) [constructor]
    cls.add_constructor([param('ns3::NetDeviceQueueInterface const &', 'arg0')])
    ## net-device-queue-interface.h (module 'network'): ns3::NetDeviceQueueInterface::NetDeviceQueueInterface() [constructor]
    cls.add_constructor([])
    ## net-device-queue-interface.h (module 'network'): std::size_t ns3::NetDeviceQueueInterface::GetNTxQueues() const [member function]
    cls.add_method('GetNTxQueues', 
                   'std::size_t', 
                   [], 
                   is_const=True)
    ## net-device-queue-interface.h (module 'network'): ns3::NetDeviceQueueInterface::SelectQueueCallback ns3::NetDeviceQueueInterface::GetSelectQueueCallback() const [member function]
    cls.add_method('GetSelectQueueCallback', 
                   'ns3::NetDeviceQueueInterface::SelectQueueCallback', 
                   [], 
                   is_const=True)
    ## net-device-queue-interface.h (module 'network'): ns3::Ptr<ns3::NetDeviceQueue> ns3::NetDeviceQueueInterface::GetTxQueue(std::size_t i) const [member function]
    cls.add_method('GetTxQueue', 
                   'ns3::Ptr< ns3::NetDeviceQueue >', 
                   [param('std::size_t', 'i')], 
                   is_const=True)
    ## net-device-queue-interface.h (module 'network'): static ns3::TypeId ns3::NetDeviceQueueInterface::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueueInterface::SetNTxQueues(std::size_t numTxQueues) [member function]
    cls.add_method('SetNTxQueues', 
                   'void', 
                   [param('std::size_t', 'numTxQueues')])
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueueInterface::SetSelectQueueCallback(ns3::NetDeviceQueueInterface::SelectQueueCallback cb) [member function]
    cls.add_method('SetSelectQueueCallback', 
                   'void', 
                   [param('std::function< unsigned long long ( ns3::Ptr< ns3::QueueItem > ) >', 'cb')])
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueueInterface::SetTxQueuesType(ns3::TypeId type) [member function]
    cls.add_method('SetTxQueuesType', 
                   'void', 
                   [param('ns3::TypeId', 'type')])
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueueInterface::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## net-device-queue-interface.h (module 'network'): void ns3::NetDeviceQueueInterface::NotifyNewAggregate() [member function]
    cls.add_method('NotifyNewAggregate', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
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

def register_Ns3PacketSizeMinMaxAvgTotalCalculator_methods(root_module, cls):
    ## packet-data-calculators.h (module 'network'): ns3::PacketSizeMinMaxAvgTotalCalculator::PacketSizeMinMaxAvgTotalCalculator(ns3::PacketSizeMinMaxAvgTotalCalculator const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketSizeMinMaxAvgTotalCalculator const &', 'arg0')])
    ## packet-data-calculators.h (module 'network'): ns3::PacketSizeMinMaxAvgTotalCalculator::PacketSizeMinMaxAvgTotalCalculator() [constructor]
    cls.add_constructor([])
    ## packet-data-calculators.h (module 'network'): void ns3::PacketSizeMinMaxAvgTotalCalculator::FrameUpdate(std::string path, ns3::Ptr<const ns3::Packet> packet, ns3::Mac48Address realto) [member function]
    cls.add_method('FrameUpdate', 
                   'void', 
                   [param('std::string', 'path'), param('ns3::Ptr< ns3::Packet const >', 'packet'), param('ns3::Mac48Address', 'realto')])
    ## packet-data-calculators.h (module 'network'): static ns3::TypeId ns3::PacketSizeMinMaxAvgTotalCalculator::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## packet-data-calculators.h (module 'network'): void ns3::PacketSizeMinMaxAvgTotalCalculator::PacketUpdate(std::string path, ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('PacketUpdate', 
                   'void', 
                   [param('std::string', 'path'), param('ns3::Ptr< ns3::Packet const >', 'packet')])
    ## packet-data-calculators.h (module 'network'): void ns3::PacketSizeMinMaxAvgTotalCalculator::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3PacketSocket_methods(root_module, cls):
    ## packet-socket.h (module 'network'): ns3::PacketSocket::PacketSocket(ns3::PacketSocket const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketSocket const &', 'arg0')])
    ## packet-socket.h (module 'network'): ns3::PacketSocket::PacketSocket() [constructor]
    cls.add_constructor([])
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::Bind() [member function]
    cls.add_method('Bind', 
                   'int', 
                   [], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::Bind(ns3::Address const & address) [member function]
    cls.add_method('Bind', 
                   'int', 
                   [param('ns3::Address const &', 'address')], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::Bind6() [member function]
    cls.add_method('Bind6', 
                   'int', 
                   [], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::Close() [member function]
    cls.add_method('Close', 
                   'int', 
                   [], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::Connect(ns3::Address const & address) [member function]
    cls.add_method('Connect', 
                   'int', 
                   [param('ns3::Address const &', 'address')], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): bool ns3::PacketSocket::GetAllowBroadcast() const [member function]
    cls.add_method('GetAllowBroadcast', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): ns3::Socket::SocketErrno ns3::PacketSocket::GetErrno() const [member function]
    cls.add_method('GetErrno', 
                   'ns3::Socket::SocketErrno', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): ns3::Ptr<ns3::Node> ns3::PacketSocket::GetNode() const [member function]
    cls.add_method('GetNode', 
                   'ns3::Ptr< ns3::Node >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::GetPeerName(ns3::Address & address) const [member function]
    cls.add_method('GetPeerName', 
                   'int', 
                   [param('ns3::Address &', 'address')], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): uint32_t ns3::PacketSocket::GetRxAvailable() const [member function]
    cls.add_method('GetRxAvailable', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::GetSockName(ns3::Address & address) const [member function]
    cls.add_method('GetSockName', 
                   'int', 
                   [param('ns3::Address &', 'address')], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): ns3::Socket::SocketType ns3::PacketSocket::GetSocketType() const [member function]
    cls.add_method('GetSocketType', 
                   'ns3::Socket::SocketType', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): uint32_t ns3::PacketSocket::GetTxAvailable() const [member function]
    cls.add_method('GetTxAvailable', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packet-socket.h (module 'network'): static ns3::TypeId ns3::PacketSocket::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::Listen() [member function]
    cls.add_method('Listen', 
                   'int', 
                   [], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::PacketSocket::Recv(uint32_t maxSize, uint32_t flags) [member function]
    cls.add_method('Recv', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('uint32_t', 'maxSize'), param('uint32_t', 'flags')], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::PacketSocket::RecvFrom(uint32_t maxSize, uint32_t flags, ns3::Address & fromAddress) [member function]
    cls.add_method('RecvFrom', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('uint32_t', 'maxSize'), param('uint32_t', 'flags'), param('ns3::Address &', 'fromAddress')], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::Send(ns3::Ptr<ns3::Packet> p, uint32_t flags) [member function]
    cls.add_method('Send', 
                   'int', 
                   [param('ns3::Ptr< ns3::Packet >', 'p'), param('uint32_t', 'flags')], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::SendTo(ns3::Ptr<ns3::Packet> p, uint32_t flags, ns3::Address const & toAddress) [member function]
    cls.add_method('SendTo', 
                   'int', 
                   [param('ns3::Ptr< ns3::Packet >', 'p'), param('uint32_t', 'flags'), param('ns3::Address const &', 'toAddress')], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): bool ns3::PacketSocket::SetAllowBroadcast(bool allowBroadcast) [member function]
    cls.add_method('SetAllowBroadcast', 
                   'bool', 
                   [param('bool', 'allowBroadcast')], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): void ns3::PacketSocket::SetNode(ns3::Ptr<ns3::Node> node) [member function]
    cls.add_method('SetNode', 
                   'void', 
                   [param('ns3::Ptr< ns3::Node >', 'node')])
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::ShutdownRecv() [member function]
    cls.add_method('ShutdownRecv', 
                   'int', 
                   [], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): int ns3::PacketSocket::ShutdownSend() [member function]
    cls.add_method('ShutdownSend', 
                   'int', 
                   [], 
                   is_virtual=True)
    ## packet-socket.h (module 'network'): void ns3::PacketSocket::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3PacketSocketClient_methods(root_module, cls):
    ## packet-socket-client.h (module 'network'): ns3::PacketSocketClient::PacketSocketClient(ns3::PacketSocketClient const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketSocketClient const &', 'arg0')])
    ## packet-socket-client.h (module 'network'): ns3::PacketSocketClient::PacketSocketClient() [constructor]
    cls.add_constructor([])
    ## packet-socket-client.h (module 'network'): uint8_t ns3::PacketSocketClient::GetPriority() const [member function]
    cls.add_method('GetPriority', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packet-socket-client.h (module 'network'): static ns3::TypeId ns3::PacketSocketClient::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## packet-socket-client.h (module 'network'): void ns3::PacketSocketClient::SetRemote(ns3::PacketSocketAddress addr) [member function]
    cls.add_method('SetRemote', 
                   'void', 
                   [param('ns3::PacketSocketAddress', 'addr')])
    ## packet-socket-client.h (module 'network'): void ns3::PacketSocketClient::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## packet-socket-client.h (module 'network'): void ns3::PacketSocketClient::StartApplication() [member function]
    cls.add_method('StartApplication', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    ## packet-socket-client.h (module 'network'): void ns3::PacketSocketClient::StopApplication() [member function]
    cls.add_method('StopApplication', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3PacketSocketFactory_methods(root_module, cls):
    ## packet-socket-factory.h (module 'network'): ns3::PacketSocketFactory::PacketSocketFactory(ns3::PacketSocketFactory const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketSocketFactory const &', 'arg0')])
    ## packet-socket-factory.h (module 'network'): ns3::PacketSocketFactory::PacketSocketFactory() [constructor]
    cls.add_constructor([])
    ## packet-socket-factory.h (module 'network'): ns3::Ptr<ns3::Socket> ns3::PacketSocketFactory::CreateSocket() [member function]
    cls.add_method('CreateSocket', 
                   'ns3::Ptr< ns3::Socket >', 
                   [], 
                   is_virtual=True)
    ## packet-socket-factory.h (module 'network'): static ns3::TypeId ns3::PacketSocketFactory::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    return

def register_Ns3PacketSocketServer_methods(root_module, cls):
    ## packet-socket-server.h (module 'network'): ns3::PacketSocketServer::PacketSocketServer(ns3::PacketSocketServer const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketSocketServer const &', 'arg0')])
    ## packet-socket-server.h (module 'network'): ns3::PacketSocketServer::PacketSocketServer() [constructor]
    cls.add_constructor([])
    ## packet-socket-server.h (module 'network'): static ns3::TypeId ns3::PacketSocketServer::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## packet-socket-server.h (module 'network'): void ns3::PacketSocketServer::SetLocal(ns3::PacketSocketAddress addr) [member function]
    cls.add_method('SetLocal', 
                   'void', 
                   [param('ns3::PacketSocketAddress', 'addr')])
    ## packet-socket-server.h (module 'network'): void ns3::PacketSocketServer::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    ## packet-socket-server.h (module 'network'): void ns3::PacketSocketServer::StartApplication() [member function]
    cls.add_method('StartApplication', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    ## packet-socket-server.h (module 'network'): void ns3::PacketSocketServer::StopApplication() [member function]
    cls.add_method('StopApplication', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
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

def register_Ns3PbbAddressBlock_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::PbbAddressBlock(ns3::PbbAddressBlock const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbAddressBlock const &', 'arg0')])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::PbbAddressBlock() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): ns3::Address ns3::PbbAddressBlock::AddressBack() const [member function]
    cls.add_method('AddressBack', 
                   'ns3::Address', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::AddressIterator ns3::PbbAddressBlock::AddressBegin() [member function]
    cls.add_method('AddressBegin', 
                   'ns3::PbbAddressBlock::AddressIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::ConstAddressIterator ns3::PbbAddressBlock::AddressBegin() const [member function]
    cls.add_method('AddressBegin', 
                   'ns3::PbbAddressBlock::ConstAddressIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::AddressClear() [member function]
    cls.add_method('AddressClear', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): bool ns3::PbbAddressBlock::AddressEmpty() const [member function]
    cls.add_method('AddressEmpty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::AddressIterator ns3::PbbAddressBlock::AddressEnd() [member function]
    cls.add_method('AddressEnd', 
                   'ns3::PbbAddressBlock::AddressIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::ConstAddressIterator ns3::PbbAddressBlock::AddressEnd() const [member function]
    cls.add_method('AddressEnd', 
                   'ns3::PbbAddressBlock::ConstAddressIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::AddressIterator ns3::PbbAddressBlock::AddressErase(ns3::PbbAddressBlock::AddressIterator position) [member function]
    cls.add_method('AddressErase', 
                   'ns3::PbbAddressBlock::AddressIterator', 
                   [param('std::list< ns3::Address > iterator', 'position')])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::AddressIterator ns3::PbbAddressBlock::AddressErase(ns3::PbbAddressBlock::AddressIterator first, ns3::PbbAddressBlock::AddressIterator last) [member function]
    cls.add_method('AddressErase', 
                   'ns3::PbbAddressBlock::AddressIterator', 
                   [param('std::list< ns3::Address > iterator', 'first'), param('std::list< ns3::Address > iterator', 'last')])
    ## packetbb.h (module 'network'): ns3::Address ns3::PbbAddressBlock::AddressFront() const [member function]
    cls.add_method('AddressFront', 
                   'ns3::Address', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::AddressIterator ns3::PbbAddressBlock::AddressInsert(ns3::PbbAddressBlock::AddressIterator position, ns3::Address const value) [member function]
    cls.add_method('AddressInsert', 
                   'ns3::PbbAddressBlock::AddressIterator', 
                   [param('std::list< ns3::Address > iterator', 'position'), param('ns3::Address const', 'value')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::AddressPopBack() [member function]
    cls.add_method('AddressPopBack', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::AddressPopFront() [member function]
    cls.add_method('AddressPopFront', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::AddressPushBack(ns3::Address address) [member function]
    cls.add_method('AddressPushBack', 
                   'void', 
                   [param('ns3::Address', 'address')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::AddressPushFront(ns3::Address address) [member function]
    cls.add_method('AddressPushFront', 
                   'void', 
                   [param('ns3::Address', 'address')])
    ## packetbb.h (module 'network'): int ns3::PbbAddressBlock::AddressSize() const [member function]
    cls.add_method('AddressSize', 
                   'int', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::Deserialize(ns3::Buffer::Iterator & start) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')])
    ## packetbb.h (module 'network'): uint32_t ns3::PbbAddressBlock::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint8_t ns3::PbbAddressBlock::PrefixBack() const [member function]
    cls.add_method('PrefixBack', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::PrefixIterator ns3::PbbAddressBlock::PrefixBegin() [member function]
    cls.add_method('PrefixBegin', 
                   'ns3::PbbAddressBlock::PrefixIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::ConstPrefixIterator ns3::PbbAddressBlock::PrefixBegin() const [member function]
    cls.add_method('PrefixBegin', 
                   'ns3::PbbAddressBlock::ConstPrefixIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::PrefixClear() [member function]
    cls.add_method('PrefixClear', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): bool ns3::PbbAddressBlock::PrefixEmpty() const [member function]
    cls.add_method('PrefixEmpty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::PrefixIterator ns3::PbbAddressBlock::PrefixEnd() [member function]
    cls.add_method('PrefixEnd', 
                   'ns3::PbbAddressBlock::PrefixIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::ConstPrefixIterator ns3::PbbAddressBlock::PrefixEnd() const [member function]
    cls.add_method('PrefixEnd', 
                   'ns3::PbbAddressBlock::ConstPrefixIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::PrefixIterator ns3::PbbAddressBlock::PrefixErase(ns3::PbbAddressBlock::PrefixIterator position) [member function]
    cls.add_method('PrefixErase', 
                   'ns3::PbbAddressBlock::PrefixIterator', 
                   [param('std::list< unsigned char > iterator', 'position')])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::PrefixIterator ns3::PbbAddressBlock::PrefixErase(ns3::PbbAddressBlock::PrefixIterator first, ns3::PbbAddressBlock::PrefixIterator last) [member function]
    cls.add_method('PrefixErase', 
                   'ns3::PbbAddressBlock::PrefixIterator', 
                   [param('std::list< unsigned char > iterator', 'first'), param('std::list< unsigned char > iterator', 'last')])
    ## packetbb.h (module 'network'): uint8_t ns3::PbbAddressBlock::PrefixFront() const [member function]
    cls.add_method('PrefixFront', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::PrefixIterator ns3::PbbAddressBlock::PrefixInsert(ns3::PbbAddressBlock::PrefixIterator position, uint8_t const value) [member function]
    cls.add_method('PrefixInsert', 
                   'ns3::PbbAddressBlock::PrefixIterator', 
                   [param('std::list< unsigned char > iterator', 'position'), param('uint8_t const', 'value')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::PrefixPopBack() [member function]
    cls.add_method('PrefixPopBack', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::PrefixPopFront() [member function]
    cls.add_method('PrefixPopFront', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::PrefixPushBack(uint8_t prefix) [member function]
    cls.add_method('PrefixPushBack', 
                   'void', 
                   [param('uint8_t', 'prefix')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::PrefixPushFront(uint8_t prefix) [member function]
    cls.add_method('PrefixPushFront', 
                   'void', 
                   [param('uint8_t', 'prefix')])
    ## packetbb.h (module 'network'): int ns3::PbbAddressBlock::PrefixSize() const [member function]
    cls.add_method('PrefixSize', 
                   'int', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::Print(std::ostream & os, int level) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os'), param('int', 'level')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::Serialize(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressTlv> ns3::PbbAddressBlock::TlvBack() [member function]
    cls.add_method('TlvBack', 
                   'ns3::Ptr< ns3::PbbAddressTlv >', 
                   [])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressTlv> const ns3::PbbAddressBlock::TlvBack() const [member function]
    cls.add_method('TlvBack', 
                   'ns3::Ptr< ns3::PbbAddressTlv > const', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::TlvIterator ns3::PbbAddressBlock::TlvBegin() [member function]
    cls.add_method('TlvBegin', 
                   'ns3::PbbAddressBlock::TlvIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::ConstTlvIterator ns3::PbbAddressBlock::TlvBegin() const [member function]
    cls.add_method('TlvBegin', 
                   'ns3::PbbAddressBlock::ConstTlvIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::TlvClear() [member function]
    cls.add_method('TlvClear', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): bool ns3::PbbAddressBlock::TlvEmpty() const [member function]
    cls.add_method('TlvEmpty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::TlvIterator ns3::PbbAddressBlock::TlvEnd() [member function]
    cls.add_method('TlvEnd', 
                   'ns3::PbbAddressBlock::TlvIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::ConstTlvIterator ns3::PbbAddressBlock::TlvEnd() const [member function]
    cls.add_method('TlvEnd', 
                   'ns3::PbbAddressBlock::ConstTlvIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::TlvIterator ns3::PbbAddressBlock::TlvErase(ns3::PbbAddressBlock::TlvIterator position) [member function]
    cls.add_method('TlvErase', 
                   'ns3::PbbAddressBlock::TlvIterator', 
                   [param('ns3::PbbAddressTlvBlock::Iterator', 'position')])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::TlvIterator ns3::PbbAddressBlock::TlvErase(ns3::PbbAddressBlock::TlvIterator first, ns3::PbbAddressBlock::TlvIterator last) [member function]
    cls.add_method('TlvErase', 
                   'ns3::PbbAddressBlock::TlvIterator', 
                   [param('ns3::PbbAddressTlvBlock::Iterator', 'first'), param('ns3::PbbAddressTlvBlock::Iterator', 'last')])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressTlv> ns3::PbbAddressBlock::TlvFront() [member function]
    cls.add_method('TlvFront', 
                   'ns3::Ptr< ns3::PbbAddressTlv >', 
                   [])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressTlv> const ns3::PbbAddressBlock::TlvFront() const [member function]
    cls.add_method('TlvFront', 
                   'ns3::Ptr< ns3::PbbAddressTlv > const', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbAddressBlock::TlvIterator ns3::PbbAddressBlock::TlvInsert(ns3::PbbAddressBlock::TlvIterator position, ns3::Ptr<ns3::PbbTlv> const value) [member function]
    cls.add_method('TlvInsert', 
                   'ns3::PbbAddressBlock::TlvIterator', 
                   [param('ns3::PbbAddressTlvBlock::Iterator', 'position'), param('ns3::Ptr< ns3::PbbTlv > const', 'value')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::TlvPopBack() [member function]
    cls.add_method('TlvPopBack', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::TlvPopFront() [member function]
    cls.add_method('TlvPopFront', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::TlvPushBack(ns3::Ptr<ns3::PbbAddressTlv> address) [member function]
    cls.add_method('TlvPushBack', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbAddressTlv >', 'address')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::TlvPushFront(ns3::Ptr<ns3::PbbAddressTlv> address) [member function]
    cls.add_method('TlvPushFront', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbAddressTlv >', 'address')])
    ## packetbb.h (module 'network'): int ns3::PbbAddressBlock::TlvSize() const [member function]
    cls.add_method('TlvSize', 
                   'int', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::Address ns3::PbbAddressBlock::DeserializeAddress(uint8_t * buffer) const [member function]
    cls.add_method('DeserializeAddress', 
                   'ns3::Address', 
                   [param('uint8_t *', 'buffer')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): uint8_t ns3::PbbAddressBlock::GetAddressLength() const [member function]
    cls.add_method('GetAddressLength', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::PrintAddress(std::ostream & os, ns3::PbbAddressBlock::ConstAddressIterator iter) const [member function]
    cls.add_method('PrintAddress', 
                   'void', 
                   [param('std::ostream &', 'os'), param('std::list< ns3::Address > const_iterator', 'iter')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlock::SerializeAddress(uint8_t * buffer, ns3::PbbAddressBlock::ConstAddressIterator iter) const [member function]
    cls.add_method('SerializeAddress', 
                   'void', 
                   [param('uint8_t *', 'buffer'), param('std::list< ns3::Address > const_iterator', 'iter')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='protected')
    return

def register_Ns3PbbAddressBlockIpv4_methods(root_module, cls):
    ## packetbb.h (module 'network'): ns3::PbbAddressBlockIpv4::PbbAddressBlockIpv4(ns3::PbbAddressBlockIpv4 const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbAddressBlockIpv4 const &', 'arg0')])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlockIpv4::PbbAddressBlockIpv4() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): ns3::Address ns3::PbbAddressBlockIpv4::DeserializeAddress(uint8_t * buffer) const [member function]
    cls.add_method('DeserializeAddress', 
                   'ns3::Address', 
                   [param('uint8_t *', 'buffer')], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): uint8_t ns3::PbbAddressBlockIpv4::GetAddressLength() const [member function]
    cls.add_method('GetAddressLength', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlockIpv4::PrintAddress(std::ostream & os, ns3::PbbAddressBlock::ConstAddressIterator iter) const [member function]
    cls.add_method('PrintAddress', 
                   'void', 
                   [param('std::ostream &', 'os'), param('std::list< ns3::Address > const_iterator', 'iter')], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlockIpv4::SerializeAddress(uint8_t * buffer, ns3::PbbAddressBlock::ConstAddressIterator iter) const [member function]
    cls.add_method('SerializeAddress', 
                   'void', 
                   [param('uint8_t *', 'buffer'), param('std::list< ns3::Address > const_iterator', 'iter')], 
                   is_const=True, is_virtual=True, visibility='protected')
    return

def register_Ns3PbbAddressBlockIpv6_methods(root_module, cls):
    ## packetbb.h (module 'network'): ns3::PbbAddressBlockIpv6::PbbAddressBlockIpv6(ns3::PbbAddressBlockIpv6 const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbAddressBlockIpv6 const &', 'arg0')])
    ## packetbb.h (module 'network'): ns3::PbbAddressBlockIpv6::PbbAddressBlockIpv6() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): ns3::Address ns3::PbbAddressBlockIpv6::DeserializeAddress(uint8_t * buffer) const [member function]
    cls.add_method('DeserializeAddress', 
                   'ns3::Address', 
                   [param('uint8_t *', 'buffer')], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): uint8_t ns3::PbbAddressBlockIpv6::GetAddressLength() const [member function]
    cls.add_method('GetAddressLength', 
                   'uint8_t', 
                   [], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlockIpv6::PrintAddress(std::ostream & os, ns3::PbbAddressBlock::ConstAddressIterator iter) const [member function]
    cls.add_method('PrintAddress', 
                   'void', 
                   [param('std::ostream &', 'os'), param('std::list< ns3::Address > const_iterator', 'iter')], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbAddressBlockIpv6::SerializeAddress(uint8_t * buffer, ns3::PbbAddressBlock::ConstAddressIterator iter) const [member function]
    cls.add_method('SerializeAddress', 
                   'void', 
                   [param('uint8_t *', 'buffer'), param('std::list< ns3::Address > const_iterator', 'iter')], 
                   is_const=True, is_virtual=True, visibility='protected')
    return

def register_Ns3PbbMessage_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    ## packetbb.h (module 'network'): ns3::PbbMessage::PbbMessage(ns3::PbbMessage const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbMessage const &', 'arg0')])
    ## packetbb.h (module 'network'): ns3::PbbMessage::PbbMessage() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressBlock> ns3::PbbMessage::AddressBlockBack() [member function]
    cls.add_method('AddressBlockBack', 
                   'ns3::Ptr< ns3::PbbAddressBlock >', 
                   [])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressBlock> const ns3::PbbMessage::AddressBlockBack() const [member function]
    cls.add_method('AddressBlockBack', 
                   'ns3::Ptr< ns3::PbbAddressBlock > const', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbMessage::AddressBlockIterator ns3::PbbMessage::AddressBlockBegin() [member function]
    cls.add_method('AddressBlockBegin', 
                   'ns3::PbbMessage::AddressBlockIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbMessage::ConstAddressBlockIterator ns3::PbbMessage::AddressBlockBegin() const [member function]
    cls.add_method('AddressBlockBegin', 
                   'ns3::PbbMessage::ConstAddressBlockIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbMessage::AddressBlockClear() [member function]
    cls.add_method('AddressBlockClear', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): bool ns3::PbbMessage::AddressBlockEmpty() const [member function]
    cls.add_method('AddressBlockEmpty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbMessage::AddressBlockIterator ns3::PbbMessage::AddressBlockEnd() [member function]
    cls.add_method('AddressBlockEnd', 
                   'ns3::PbbMessage::AddressBlockIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbMessage::ConstAddressBlockIterator ns3::PbbMessage::AddressBlockEnd() const [member function]
    cls.add_method('AddressBlockEnd', 
                   'ns3::PbbMessage::ConstAddressBlockIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbMessage::AddressBlockIterator ns3::PbbMessage::AddressBlockErase(ns3::PbbMessage::AddressBlockIterator position) [member function]
    cls.add_method('AddressBlockErase', 
                   'ns3::PbbMessage::AddressBlockIterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbAddressBlock > > iterator', 'position')])
    ## packetbb.h (module 'network'): ns3::PbbMessage::AddressBlockIterator ns3::PbbMessage::AddressBlockErase(ns3::PbbMessage::AddressBlockIterator first, ns3::PbbMessage::AddressBlockIterator last) [member function]
    cls.add_method('AddressBlockErase', 
                   'ns3::PbbMessage::AddressBlockIterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbAddressBlock > > iterator', 'first'), param('std::list< ns3::Ptr< ns3::PbbAddressBlock > > iterator', 'last')])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressBlock> ns3::PbbMessage::AddressBlockFront() [member function]
    cls.add_method('AddressBlockFront', 
                   'ns3::Ptr< ns3::PbbAddressBlock >', 
                   [])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressBlock> const ns3::PbbMessage::AddressBlockFront() const [member function]
    cls.add_method('AddressBlockFront', 
                   'ns3::Ptr< ns3::PbbAddressBlock > const', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbMessage::AddressBlockPopBack() [member function]
    cls.add_method('AddressBlockPopBack', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbMessage::AddressBlockPopFront() [member function]
    cls.add_method('AddressBlockPopFront', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbMessage::AddressBlockPushBack(ns3::Ptr<ns3::PbbAddressBlock> block) [member function]
    cls.add_method('AddressBlockPushBack', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbAddressBlock >', 'block')])
    ## packetbb.h (module 'network'): void ns3::PbbMessage::AddressBlockPushFront(ns3::Ptr<ns3::PbbAddressBlock> block) [member function]
    cls.add_method('AddressBlockPushFront', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbAddressBlock >', 'block')])
    ## packetbb.h (module 'network'): int ns3::PbbMessage::AddressBlockSize() const [member function]
    cls.add_method('AddressBlockSize', 
                   'int', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbMessage::Deserialize(ns3::Buffer::Iterator & start) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')])
    ## packetbb.h (module 'network'): static ns3::Ptr<ns3::PbbMessage> ns3::PbbMessage::DeserializeMessage(ns3::Buffer::Iterator & start) [member function]
    cls.add_method('DeserializeMessage', 
                   'ns3::Ptr< ns3::PbbMessage >', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_static=True)
    ## packetbb.h (module 'network'): uint8_t ns3::PbbMessage::GetHopCount() const [member function]
    cls.add_method('GetHopCount', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint8_t ns3::PbbMessage::GetHopLimit() const [member function]
    cls.add_method('GetHopLimit', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::Address ns3::PbbMessage::GetOriginatorAddress() const [member function]
    cls.add_method('GetOriginatorAddress', 
                   'ns3::Address', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint16_t ns3::PbbMessage::GetSequenceNumber() const [member function]
    cls.add_method('GetSequenceNumber', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint32_t ns3::PbbMessage::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint8_t ns3::PbbMessage::GetType() const [member function]
    cls.add_method('GetType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): bool ns3::PbbMessage::HasHopCount() const [member function]
    cls.add_method('HasHopCount', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): bool ns3::PbbMessage::HasHopLimit() const [member function]
    cls.add_method('HasHopLimit', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): bool ns3::PbbMessage::HasOriginatorAddress() const [member function]
    cls.add_method('HasOriginatorAddress', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): bool ns3::PbbMessage::HasSequenceNumber() const [member function]
    cls.add_method('HasSequenceNumber', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbMessage::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbMessage::Print(std::ostream & os, int level) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os'), param('int', 'level')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbMessage::Serialize(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbMessage::SetHopCount(uint8_t hopcount) [member function]
    cls.add_method('SetHopCount', 
                   'void', 
                   [param('uint8_t', 'hopcount')])
    ## packetbb.h (module 'network'): void ns3::PbbMessage::SetHopLimit(uint8_t hoplimit) [member function]
    cls.add_method('SetHopLimit', 
                   'void', 
                   [param('uint8_t', 'hoplimit')])
    ## packetbb.h (module 'network'): void ns3::PbbMessage::SetOriginatorAddress(ns3::Address address) [member function]
    cls.add_method('SetOriginatorAddress', 
                   'void', 
                   [param('ns3::Address', 'address')])
    ## packetbb.h (module 'network'): void ns3::PbbMessage::SetSequenceNumber(uint16_t seqnum) [member function]
    cls.add_method('SetSequenceNumber', 
                   'void', 
                   [param('uint16_t', 'seqnum')])
    ## packetbb.h (module 'network'): void ns3::PbbMessage::SetType(uint8_t type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('uint8_t', 'type')])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbTlv> ns3::PbbMessage::TlvBack() [member function]
    cls.add_method('TlvBack', 
                   'ns3::Ptr< ns3::PbbTlv >', 
                   [])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbTlv> const ns3::PbbMessage::TlvBack() const [member function]
    cls.add_method('TlvBack', 
                   'ns3::Ptr< ns3::PbbTlv > const', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbMessage::TlvIterator ns3::PbbMessage::TlvBegin() [member function]
    cls.add_method('TlvBegin', 
                   'ns3::PbbMessage::TlvIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbMessage::ConstTlvIterator ns3::PbbMessage::TlvBegin() const [member function]
    cls.add_method('TlvBegin', 
                   'ns3::PbbMessage::ConstTlvIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbMessage::TlvClear() [member function]
    cls.add_method('TlvClear', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): bool ns3::PbbMessage::TlvEmpty() const [member function]
    cls.add_method('TlvEmpty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbMessage::TlvIterator ns3::PbbMessage::TlvEnd() [member function]
    cls.add_method('TlvEnd', 
                   'ns3::PbbMessage::TlvIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbMessage::ConstTlvIterator ns3::PbbMessage::TlvEnd() const [member function]
    cls.add_method('TlvEnd', 
                   'ns3::PbbMessage::ConstTlvIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbMessage::TlvIterator ns3::PbbMessage::TlvErase(ns3::PbbMessage::TlvIterator position) [member function]
    cls.add_method('TlvErase', 
                   'ns3::PbbMessage::TlvIterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'position')])
    ## packetbb.h (module 'network'): ns3::PbbMessage::TlvIterator ns3::PbbMessage::TlvErase(ns3::PbbMessage::TlvIterator first, ns3::PbbMessage::TlvIterator last) [member function]
    cls.add_method('TlvErase', 
                   'ns3::PbbMessage::TlvIterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'first'), param('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'last')])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbTlv> ns3::PbbMessage::TlvFront() [member function]
    cls.add_method('TlvFront', 
                   'ns3::Ptr< ns3::PbbTlv >', 
                   [])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbTlv> const ns3::PbbMessage::TlvFront() const [member function]
    cls.add_method('TlvFront', 
                   'ns3::Ptr< ns3::PbbTlv > const', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbMessage::TlvPopBack() [member function]
    cls.add_method('TlvPopBack', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbMessage::TlvPopFront() [member function]
    cls.add_method('TlvPopFront', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbMessage::TlvPushBack(ns3::Ptr<ns3::PbbTlv> tlv) [member function]
    cls.add_method('TlvPushBack', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbTlv >', 'tlv')])
    ## packetbb.h (module 'network'): void ns3::PbbMessage::TlvPushFront(ns3::Ptr<ns3::PbbTlv> tlv) [member function]
    cls.add_method('TlvPushFront', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbTlv >', 'tlv')])
    ## packetbb.h (module 'network'): int ns3::PbbMessage::TlvSize() const [member function]
    cls.add_method('TlvSize', 
                   'int', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressBlock> ns3::PbbMessage::AddressBlockDeserialize(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('AddressBlockDeserialize', 
                   'ns3::Ptr< ns3::PbbAddressBlock >', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): ns3::Address ns3::PbbMessage::DeserializeOriginatorAddress(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('DeserializeOriginatorAddress', 
                   'ns3::Address', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): ns3::PbbAddressLength ns3::PbbMessage::GetAddressLength() const [member function]
    cls.add_method('GetAddressLength', 
                   'ns3::PbbAddressLength', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbMessage::PrintOriginatorAddress(std::ostream & os) const [member function]
    cls.add_method('PrintOriginatorAddress', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbMessage::SerializeOriginatorAddress(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('SerializeOriginatorAddress', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True, is_pure_virtual=True, is_virtual=True, visibility='protected')
    return

def register_Ns3PbbMessageIpv4_methods(root_module, cls):
    ## packetbb.h (module 'network'): ns3::PbbMessageIpv4::PbbMessageIpv4(ns3::PbbMessageIpv4 const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbMessageIpv4 const &', 'arg0')])
    ## packetbb.h (module 'network'): ns3::PbbMessageIpv4::PbbMessageIpv4() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressBlock> ns3::PbbMessageIpv4::AddressBlockDeserialize(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('AddressBlockDeserialize', 
                   'ns3::Ptr< ns3::PbbAddressBlock >', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): ns3::Address ns3::PbbMessageIpv4::DeserializeOriginatorAddress(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('DeserializeOriginatorAddress', 
                   'ns3::Address', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): ns3::PbbAddressLength ns3::PbbMessageIpv4::GetAddressLength() const [member function]
    cls.add_method('GetAddressLength', 
                   'ns3::PbbAddressLength', 
                   [], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbMessageIpv4::PrintOriginatorAddress(std::ostream & os) const [member function]
    cls.add_method('PrintOriginatorAddress', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbMessageIpv4::SerializeOriginatorAddress(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('SerializeOriginatorAddress', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True, is_virtual=True, visibility='protected')
    return

def register_Ns3PbbMessageIpv6_methods(root_module, cls):
    ## packetbb.h (module 'network'): ns3::PbbMessageIpv6::PbbMessageIpv6(ns3::PbbMessageIpv6 const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbMessageIpv6 const &', 'arg0')])
    ## packetbb.h (module 'network'): ns3::PbbMessageIpv6::PbbMessageIpv6() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbAddressBlock> ns3::PbbMessageIpv6::AddressBlockDeserialize(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('AddressBlockDeserialize', 
                   'ns3::Ptr< ns3::PbbAddressBlock >', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): ns3::Address ns3::PbbMessageIpv6::DeserializeOriginatorAddress(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('DeserializeOriginatorAddress', 
                   'ns3::Address', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): ns3::PbbAddressLength ns3::PbbMessageIpv6::GetAddressLength() const [member function]
    cls.add_method('GetAddressLength', 
                   'ns3::PbbAddressLength', 
                   [], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbMessageIpv6::PrintOriginatorAddress(std::ostream & os) const [member function]
    cls.add_method('PrintOriginatorAddress', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbMessageIpv6::SerializeOriginatorAddress(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('SerializeOriginatorAddress', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True, is_virtual=True, visibility='protected')
    return

def register_Ns3PbbPacket_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    ## packetbb.h (module 'network'): ns3::PbbPacket::PbbPacket(ns3::PbbPacket const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbPacket const &', 'arg0')])
    ## packetbb.h (module 'network'): ns3::PbbPacket::PbbPacket() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): uint32_t ns3::PbbPacket::Deserialize(ns3::Buffer::Iterator start) [member function]
    cls.add_method('Deserialize', 
                   'uint32_t', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_virtual=True)
    ## packetbb.h (module 'network'): ns3::PbbPacket::TlvIterator ns3::PbbPacket::Erase(ns3::PbbPacket::TlvIterator position) [member function]
    cls.add_method('Erase', 
                   'ns3::PbbPacket::TlvIterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'position')])
    ## packetbb.h (module 'network'): ns3::PbbPacket::TlvIterator ns3::PbbPacket::Erase(ns3::PbbPacket::TlvIterator first, ns3::PbbPacket::TlvIterator last) [member function]
    cls.add_method('Erase', 
                   'ns3::PbbPacket::TlvIterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'first'), param('std::list< ns3::Ptr< ns3::PbbTlv > > iterator', 'last')])
    ## packetbb.h (module 'network'): ns3::PbbPacket::MessageIterator ns3::PbbPacket::Erase(ns3::PbbPacket::MessageIterator position) [member function]
    cls.add_method('Erase', 
                   'ns3::PbbPacket::MessageIterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbMessage > > iterator', 'position')])
    ## packetbb.h (module 'network'): ns3::PbbPacket::MessageIterator ns3::PbbPacket::Erase(ns3::PbbPacket::MessageIterator first, ns3::PbbPacket::MessageIterator last) [member function]
    cls.add_method('Erase', 
                   'ns3::PbbPacket::MessageIterator', 
                   [param('std::list< ns3::Ptr< ns3::PbbMessage > > iterator', 'first'), param('std::list< ns3::Ptr< ns3::PbbMessage > > iterator', 'last')])
    ## packetbb.h (module 'network'): ns3::TypeId ns3::PbbPacket::GetInstanceTypeId() const [member function]
    cls.add_method('GetInstanceTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packetbb.h (module 'network'): uint16_t ns3::PbbPacket::GetSequenceNumber() const [member function]
    cls.add_method('GetSequenceNumber', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint32_t ns3::PbbPacket::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## packetbb.h (module 'network'): static ns3::TypeId ns3::PbbPacket::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## packetbb.h (module 'network'): uint8_t ns3::PbbPacket::GetVersion() const [member function]
    cls.add_method('GetVersion', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): bool ns3::PbbPacket::HasSequenceNumber() const [member function]
    cls.add_method('HasSequenceNumber', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbMessage> ns3::PbbPacket::MessageBack() [member function]
    cls.add_method('MessageBack', 
                   'ns3::Ptr< ns3::PbbMessage >', 
                   [])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbMessage> const ns3::PbbPacket::MessageBack() const [member function]
    cls.add_method('MessageBack', 
                   'ns3::Ptr< ns3::PbbMessage > const', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbPacket::MessageIterator ns3::PbbPacket::MessageBegin() [member function]
    cls.add_method('MessageBegin', 
                   'ns3::PbbPacket::MessageIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbPacket::ConstMessageIterator ns3::PbbPacket::MessageBegin() const [member function]
    cls.add_method('MessageBegin', 
                   'ns3::PbbPacket::ConstMessageIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbPacket::MessageClear() [member function]
    cls.add_method('MessageClear', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): bool ns3::PbbPacket::MessageEmpty() const [member function]
    cls.add_method('MessageEmpty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbPacket::MessageIterator ns3::PbbPacket::MessageEnd() [member function]
    cls.add_method('MessageEnd', 
                   'ns3::PbbPacket::MessageIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbPacket::ConstMessageIterator ns3::PbbPacket::MessageEnd() const [member function]
    cls.add_method('MessageEnd', 
                   'ns3::PbbPacket::ConstMessageIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbMessage> ns3::PbbPacket::MessageFront() [member function]
    cls.add_method('MessageFront', 
                   'ns3::Ptr< ns3::PbbMessage >', 
                   [])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbMessage> const ns3::PbbPacket::MessageFront() const [member function]
    cls.add_method('MessageFront', 
                   'ns3::Ptr< ns3::PbbMessage > const', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbPacket::MessagePopBack() [member function]
    cls.add_method('MessagePopBack', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbPacket::MessagePopFront() [member function]
    cls.add_method('MessagePopFront', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbPacket::MessagePushBack(ns3::Ptr<ns3::PbbMessage> message) [member function]
    cls.add_method('MessagePushBack', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbMessage >', 'message')])
    ## packetbb.h (module 'network'): void ns3::PbbPacket::MessagePushFront(ns3::Ptr<ns3::PbbMessage> message) [member function]
    cls.add_method('MessagePushFront', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbMessage >', 'message')])
    ## packetbb.h (module 'network'): int ns3::PbbPacket::MessageSize() const [member function]
    cls.add_method('MessageSize', 
                   'int', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbPacket::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## packetbb.h (module 'network'): void ns3::PbbPacket::Serialize(ns3::Buffer::Iterator start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator', 'start')], 
                   is_const=True, is_virtual=True)
    ## packetbb.h (module 'network'): void ns3::PbbPacket::SetSequenceNumber(uint16_t number) [member function]
    cls.add_method('SetSequenceNumber', 
                   'void', 
                   [param('uint16_t', 'number')])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbTlv> ns3::PbbPacket::TlvBack() [member function]
    cls.add_method('TlvBack', 
                   'ns3::Ptr< ns3::PbbTlv >', 
                   [])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbTlv> const ns3::PbbPacket::TlvBack() const [member function]
    cls.add_method('TlvBack', 
                   'ns3::Ptr< ns3::PbbTlv > const', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbPacket::TlvIterator ns3::PbbPacket::TlvBegin() [member function]
    cls.add_method('TlvBegin', 
                   'ns3::PbbPacket::TlvIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbPacket::ConstTlvIterator ns3::PbbPacket::TlvBegin() const [member function]
    cls.add_method('TlvBegin', 
                   'ns3::PbbPacket::ConstTlvIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbPacket::TlvClear() [member function]
    cls.add_method('TlvClear', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): bool ns3::PbbPacket::TlvEmpty() const [member function]
    cls.add_method('TlvEmpty', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::PbbPacket::TlvIterator ns3::PbbPacket::TlvEnd() [member function]
    cls.add_method('TlvEnd', 
                   'ns3::PbbPacket::TlvIterator', 
                   [])
    ## packetbb.h (module 'network'): ns3::PbbPacket::ConstTlvIterator ns3::PbbPacket::TlvEnd() const [member function]
    cls.add_method('TlvEnd', 
                   'ns3::PbbPacket::ConstTlvIterator', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbTlv> ns3::PbbPacket::TlvFront() [member function]
    cls.add_method('TlvFront', 
                   'ns3::Ptr< ns3::PbbTlv >', 
                   [])
    ## packetbb.h (module 'network'): ns3::Ptr<ns3::PbbTlv> const ns3::PbbPacket::TlvFront() const [member function]
    cls.add_method('TlvFront', 
                   'ns3::Ptr< ns3::PbbTlv > const', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbPacket::TlvPopBack() [member function]
    cls.add_method('TlvPopBack', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbPacket::TlvPopFront() [member function]
    cls.add_method('TlvPopFront', 
                   'void', 
                   [])
    ## packetbb.h (module 'network'): void ns3::PbbPacket::TlvPushBack(ns3::Ptr<ns3::PbbTlv> tlv) [member function]
    cls.add_method('TlvPushBack', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbTlv >', 'tlv')])
    ## packetbb.h (module 'network'): void ns3::PbbPacket::TlvPushFront(ns3::Ptr<ns3::PbbTlv> tlv) [member function]
    cls.add_method('TlvPushFront', 
                   'void', 
                   [param('ns3::Ptr< ns3::PbbTlv >', 'tlv')])
    ## packetbb.h (module 'network'): int ns3::PbbPacket::TlvSize() const [member function]
    cls.add_method('TlvSize', 
                   'int', 
                   [], 
                   is_const=True)
    return

def register_Ns3PbbTlv_methods(root_module, cls):
    cls.add_binary_comparison_operator('==')
    cls.add_binary_comparison_operator('!=')
    ## packetbb.h (module 'network'): ns3::PbbTlv::PbbTlv(ns3::PbbTlv const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbTlv const &', 'arg0')])
    ## packetbb.h (module 'network'): ns3::PbbTlv::PbbTlv() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): void ns3::PbbTlv::Deserialize(ns3::Buffer::Iterator & start) [member function]
    cls.add_method('Deserialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')])
    ## packetbb.h (module 'network'): uint32_t ns3::PbbTlv::GetSerializedSize() const [member function]
    cls.add_method('GetSerializedSize', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint8_t ns3::PbbTlv::GetType() const [member function]
    cls.add_method('GetType', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint8_t ns3::PbbTlv::GetTypeExt() const [member function]
    cls.add_method('GetTypeExt', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): ns3::Buffer ns3::PbbTlv::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'ns3::Buffer', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): bool ns3::PbbTlv::HasTypeExt() const [member function]
    cls.add_method('HasTypeExt', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): bool ns3::PbbTlv::HasValue() const [member function]
    cls.add_method('HasValue', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbTlv::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbTlv::Print(std::ostream & os, int level) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os'), param('int', 'level')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbTlv::Serialize(ns3::Buffer::Iterator & start) const [member function]
    cls.add_method('Serialize', 
                   'void', 
                   [param('ns3::Buffer::Iterator &', 'start')], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbTlv::SetType(uint8_t type) [member function]
    cls.add_method('SetType', 
                   'void', 
                   [param('uint8_t', 'type')])
    ## packetbb.h (module 'network'): void ns3::PbbTlv::SetTypeExt(uint8_t type) [member function]
    cls.add_method('SetTypeExt', 
                   'void', 
                   [param('uint8_t', 'type')])
    ## packetbb.h (module 'network'): void ns3::PbbTlv::SetValue(ns3::Buffer start) [member function]
    cls.add_method('SetValue', 
                   'void', 
                   [param('ns3::Buffer', 'start')])
    ## packetbb.h (module 'network'): void ns3::PbbTlv::SetValue(uint8_t const * buffer, uint32_t size) [member function]
    cls.add_method('SetValue', 
                   'void', 
                   [param('uint8_t const *', 'buffer'), param('uint32_t', 'size')])
    ## packetbb.h (module 'network'): uint8_t ns3::PbbTlv::GetIndexStart() const [member function]
    cls.add_method('GetIndexStart', 
                   'uint8_t', 
                   [], 
                   is_const=True, visibility='protected')
    ## packetbb.h (module 'network'): uint8_t ns3::PbbTlv::GetIndexStop() const [member function]
    cls.add_method('GetIndexStop', 
                   'uint8_t', 
                   [], 
                   is_const=True, visibility='protected')
    ## packetbb.h (module 'network'): bool ns3::PbbTlv::HasIndexStart() const [member function]
    cls.add_method('HasIndexStart', 
                   'bool', 
                   [], 
                   is_const=True, visibility='protected')
    ## packetbb.h (module 'network'): bool ns3::PbbTlv::HasIndexStop() const [member function]
    cls.add_method('HasIndexStop', 
                   'bool', 
                   [], 
                   is_const=True, visibility='protected')
    ## packetbb.h (module 'network'): bool ns3::PbbTlv::IsMultivalue() const [member function]
    cls.add_method('IsMultivalue', 
                   'bool', 
                   [], 
                   is_const=True, visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbTlv::SetIndexStart(uint8_t index) [member function]
    cls.add_method('SetIndexStart', 
                   'void', 
                   [param('uint8_t', 'index')], 
                   visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbTlv::SetIndexStop(uint8_t index) [member function]
    cls.add_method('SetIndexStop', 
                   'void', 
                   [param('uint8_t', 'index')], 
                   visibility='protected')
    ## packetbb.h (module 'network'): void ns3::PbbTlv::SetMultivalue(bool isMultivalue) [member function]
    cls.add_method('SetMultivalue', 
                   'void', 
                   [param('bool', 'isMultivalue')], 
                   visibility='protected')
    return

def register_Ns3Probe_methods(root_module, cls):
    ## probe.h (module 'stats'): ns3::Probe::Probe(ns3::Probe const & arg0) [constructor]
    cls.add_constructor([param('ns3::Probe const &', 'arg0')])
    ## probe.h (module 'stats'): ns3::Probe::Probe() [constructor]
    cls.add_constructor([])
    ## probe.h (module 'stats'): bool ns3::Probe::ConnectByObject(std::string traceSource, ns3::Ptr<ns3::Object> obj) [member function]
    cls.add_method('ConnectByObject', 
                   'bool', 
                   [param('std::string', 'traceSource'), param('ns3::Ptr< ns3::Object >', 'obj')], 
                   is_pure_virtual=True, is_virtual=True)
    ## probe.h (module 'stats'): void ns3::Probe::ConnectByPath(std::string path) [member function]
    cls.add_method('ConnectByPath', 
                   'void', 
                   [param('std::string', 'path')], 
                   is_pure_virtual=True, is_virtual=True)
    ## probe.h (module 'stats'): static ns3::TypeId ns3::Probe::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## probe.h (module 'stats'): bool ns3::Probe::IsEnabled() const [member function]
    cls.add_method('IsEnabled', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3Queue__Ns3Packet_methods(root_module, cls):
    ## queue.h (module 'network'): static ns3::TypeId ns3::Queue<ns3::Packet>::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## queue.h (module 'network'): ns3::Queue<ns3::Packet>::Queue() [constructor]
    cls.add_constructor([])
    ## queue.h (module 'network'): bool ns3::Queue<ns3::Packet>::Enqueue(ns3::Ptr<ns3::Packet> item) [member function]
    cls.add_method('Enqueue', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'item')], 
                   is_pure_virtual=True, is_virtual=True)
    ## queue.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::Queue<ns3::Packet>::Dequeue() [member function]
    cls.add_method('Dequeue', 
                   'ns3::Ptr< ns3::Packet >', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## queue.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::Queue<ns3::Packet>::Remove() [member function]
    cls.add_method('Remove', 
                   'ns3::Ptr< ns3::Packet >', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## queue.h (module 'network'): ns3::Ptr<const ns3::Packet> ns3::Queue<ns3::Packet>::Peek() const [member function]
    cls.add_method('Peek', 
                   'ns3::Ptr< ns3::Packet const >', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## queue.h (module 'network'): void ns3::Queue<ns3::Packet>::Flush() [member function]
    cls.add_method('Flush', 
                   'void', 
                   [])
    ## queue.h (module 'network'): ns3::Queue<ns3::Packet>::Queue(ns3::Queue<ns3::Packet> const & arg0) [constructor]
    cls.add_constructor([param('ns3::Queue< ns3::Packet > const &', 'arg0')])
    ## queue.h (module 'network'): ns3::Queue<ns3::Packet>::ConstIterator ns3::Queue<ns3::Packet>::begin() const [member function]
    cls.add_method('begin', 
                   'ns3::Queue< ns3::Packet > ConstIterator', 
                   [], 
                   is_const=True, visibility='protected')
    ## queue.h (module 'network'): ns3::Queue<ns3::Packet>::Iterator ns3::Queue<ns3::Packet>::begin() [member function]
    cls.add_method('begin', 
                   'ns3::Queue< ns3::Packet > Iterator', 
                   [], 
                   visibility='protected')
    ## queue.h (module 'network'): ns3::Queue<ns3::Packet>::ConstIterator ns3::Queue<ns3::Packet>::end() const [member function]
    cls.add_method('end', 
                   'ns3::Queue< ns3::Packet > ConstIterator', 
                   [], 
                   is_const=True, visibility='protected')
    ## queue.h (module 'network'): ns3::Queue<ns3::Packet>::Iterator ns3::Queue<ns3::Packet>::end() [member function]
    cls.add_method('end', 
                   'ns3::Queue< ns3::Packet > Iterator', 
                   [], 
                   visibility='protected')
    ## queue.h (module 'network'): bool ns3::Queue<ns3::Packet>::DoEnqueue(ns3::Queue<ns3::Packet>::ConstIterator pos, ns3::Ptr<ns3::Packet> item) [member function]
    cls.add_method('DoEnqueue', 
                   'bool', 
                   [param('std::list< ns3::Ptr< ns3::Packet > > const_iterator', 'pos'), param('ns3::Ptr< ns3::Packet >', 'item')], 
                   visibility='protected')
    ## queue.h (module 'network'): bool ns3::Queue<ns3::Packet>::DoEnqueue(ns3::Queue<ns3::Packet>::ConstIterator pos, ns3::Ptr<ns3::Packet> item, ns3::Queue<ns3::Packet>::Iterator & ret) [member function]
    cls.add_method('DoEnqueue', 
                   'bool', 
                   [param('std::list< ns3::Ptr< ns3::Packet > > const_iterator', 'pos'), param('ns3::Ptr< ns3::Packet >', 'item'), param('ns3::Queue< ns3::Packet > Iterator &', 'ret')], 
                   visibility='protected')
    ## queue.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::Queue<ns3::Packet>::DoDequeue(ns3::Queue<ns3::Packet>::ConstIterator pos) [member function]
    cls.add_method('DoDequeue', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('std::list< ns3::Ptr< ns3::Packet > > const_iterator', 'pos')], 
                   visibility='protected')
    ## queue.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::Queue<ns3::Packet>::DoRemove(ns3::Queue<ns3::Packet>::ConstIterator pos) [member function]
    cls.add_method('DoRemove', 
                   'ns3::Ptr< ns3::Packet >', 
                   [param('std::list< ns3::Ptr< ns3::Packet > > const_iterator', 'pos')], 
                   visibility='protected')
    ## queue.h (module 'network'): ns3::Ptr<const ns3::Packet> ns3::Queue<ns3::Packet>::DoPeek(ns3::Queue<ns3::Packet>::ConstIterator pos) const [member function]
    cls.add_method('DoPeek', 
                   'ns3::Ptr< ns3::Packet const >', 
                   [param('std::list< ns3::Ptr< ns3::Packet > > const_iterator', 'pos')], 
                   is_const=True, visibility='protected')
    ## queue.h (module 'network'): void ns3::Queue<ns3::Packet>::DropBeforeEnqueue(ns3::Ptr<ns3::Packet> item) [member function]
    cls.add_method('DropBeforeEnqueue', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'item')], 
                   visibility='protected')
    ## queue.h (module 'network'): void ns3::Queue<ns3::Packet>::DropAfterDequeue(ns3::Ptr<ns3::Packet> item) [member function]
    cls.add_method('DropAfterDequeue', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'item')], 
                   visibility='protected')
    ## queue.h (module 'network'): void ns3::Queue<ns3::Packet>::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3Queue__Ns3QueueDiscItem_methods(root_module, cls):
    ## queue.h (module 'network'): static ns3::TypeId ns3::Queue<ns3::QueueDiscItem>::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## queue.h (module 'network'): ns3::Queue<ns3::QueueDiscItem>::Queue() [constructor]
    cls.add_constructor([])
    ## queue.h (module 'network'): bool ns3::Queue<ns3::QueueDiscItem>::Enqueue(ns3::Ptr<ns3::QueueDiscItem> item) [member function]
    cls.add_method('Enqueue', 
                   'bool', 
                   [param('ns3::Ptr< ns3::QueueDiscItem >', 'item')], 
                   is_pure_virtual=True, is_virtual=True)
    ## queue.h (module 'network'): ns3::Ptr<ns3::QueueDiscItem> ns3::Queue<ns3::QueueDiscItem>::Dequeue() [member function]
    cls.add_method('Dequeue', 
                   'ns3::Ptr< ns3::QueueDiscItem >', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## queue.h (module 'network'): ns3::Ptr<ns3::QueueDiscItem> ns3::Queue<ns3::QueueDiscItem>::Remove() [member function]
    cls.add_method('Remove', 
                   'ns3::Ptr< ns3::QueueDiscItem >', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## queue.h (module 'network'): ns3::Ptr<const ns3::QueueDiscItem> ns3::Queue<ns3::QueueDiscItem>::Peek() const [member function]
    cls.add_method('Peek', 
                   'ns3::Ptr< ns3::QueueDiscItem const >', 
                   [], 
                   is_const=True, is_pure_virtual=True, is_virtual=True)
    ## queue.h (module 'network'): void ns3::Queue<ns3::QueueDiscItem>::Flush() [member function]
    cls.add_method('Flush', 
                   'void', 
                   [])
    ## queue.h (module 'network'): ns3::Queue<ns3::QueueDiscItem>::Queue(ns3::Queue<ns3::QueueDiscItem> const & arg0) [constructor]
    cls.add_constructor([param('ns3::Queue< ns3::QueueDiscItem > const &', 'arg0')])
    ## queue.h (module 'network'): ns3::Queue<ns3::QueueDiscItem>::ConstIterator ns3::Queue<ns3::QueueDiscItem>::begin() const [member function]
    cls.add_method('begin', 
                   'ns3::Queue< ns3::QueueDiscItem > ConstIterator', 
                   [], 
                   is_const=True, visibility='protected')
    ## queue.h (module 'network'): ns3::Queue<ns3::QueueDiscItem>::Iterator ns3::Queue<ns3::QueueDiscItem>::begin() [member function]
    cls.add_method('begin', 
                   'ns3::Queue< ns3::QueueDiscItem > Iterator', 
                   [], 
                   visibility='protected')
    ## queue.h (module 'network'): ns3::Queue<ns3::QueueDiscItem>::ConstIterator ns3::Queue<ns3::QueueDiscItem>::end() const [member function]
    cls.add_method('end', 
                   'ns3::Queue< ns3::QueueDiscItem > ConstIterator', 
                   [], 
                   is_const=True, visibility='protected')
    ## queue.h (module 'network'): ns3::Queue<ns3::QueueDiscItem>::Iterator ns3::Queue<ns3::QueueDiscItem>::end() [member function]
    cls.add_method('end', 
                   'ns3::Queue< ns3::QueueDiscItem > Iterator', 
                   [], 
                   visibility='protected')
    ## queue.h (module 'network'): bool ns3::Queue<ns3::QueueDiscItem>::DoEnqueue(ns3::Queue<ns3::QueueDiscItem>::ConstIterator pos, ns3::Ptr<ns3::QueueDiscItem> item) [member function]
    cls.add_method('DoEnqueue', 
                   'bool', 
                   [param('std::list< ns3::Ptr< ns3::QueueDiscItem > > const_iterator', 'pos'), param('ns3::Ptr< ns3::QueueDiscItem >', 'item')], 
                   visibility='protected')
    ## queue.h (module 'network'): bool ns3::Queue<ns3::QueueDiscItem>::DoEnqueue(ns3::Queue<ns3::QueueDiscItem>::ConstIterator pos, ns3::Ptr<ns3::QueueDiscItem> item, ns3::Queue<ns3::QueueDiscItem>::Iterator & ret) [member function]
    cls.add_method('DoEnqueue', 
                   'bool', 
                   [param('std::list< ns3::Ptr< ns3::QueueDiscItem > > const_iterator', 'pos'), param('ns3::Ptr< ns3::QueueDiscItem >', 'item'), param('ns3::Queue< ns3::QueueDiscItem > Iterator &', 'ret')], 
                   visibility='protected')
    ## queue.h (module 'network'): ns3::Ptr<ns3::QueueDiscItem> ns3::Queue<ns3::QueueDiscItem>::DoDequeue(ns3::Queue<ns3::QueueDiscItem>::ConstIterator pos) [member function]
    cls.add_method('DoDequeue', 
                   'ns3::Ptr< ns3::QueueDiscItem >', 
                   [param('std::list< ns3::Ptr< ns3::QueueDiscItem > > const_iterator', 'pos')], 
                   visibility='protected')
    ## queue.h (module 'network'): ns3::Ptr<ns3::QueueDiscItem> ns3::Queue<ns3::QueueDiscItem>::DoRemove(ns3::Queue<ns3::QueueDiscItem>::ConstIterator pos) [member function]
    cls.add_method('DoRemove', 
                   'ns3::Ptr< ns3::QueueDiscItem >', 
                   [param('std::list< ns3::Ptr< ns3::QueueDiscItem > > const_iterator', 'pos')], 
                   visibility='protected')
    ## queue.h (module 'network'): ns3::Ptr<const ns3::QueueDiscItem> ns3::Queue<ns3::QueueDiscItem>::DoPeek(ns3::Queue<ns3::QueueDiscItem>::ConstIterator pos) const [member function]
    cls.add_method('DoPeek', 
                   'ns3::Ptr< ns3::QueueDiscItem const >', 
                   [param('std::list< ns3::Ptr< ns3::QueueDiscItem > > const_iterator', 'pos')], 
                   is_const=True, visibility='protected')
    ## queue.h (module 'network'): void ns3::Queue<ns3::QueueDiscItem>::DropBeforeEnqueue(ns3::Ptr<ns3::QueueDiscItem> item) [member function]
    cls.add_method('DropBeforeEnqueue', 
                   'void', 
                   [param('ns3::Ptr< ns3::QueueDiscItem >', 'item')], 
                   visibility='protected')
    ## queue.h (module 'network'): void ns3::Queue<ns3::QueueDiscItem>::DropAfterDequeue(ns3::Ptr<ns3::QueueDiscItem> item) [member function]
    cls.add_method('DropAfterDequeue', 
                   'void', 
                   [param('ns3::Ptr< ns3::QueueDiscItem >', 'item')], 
                   visibility='protected')
    ## queue.h (module 'network'): void ns3::Queue<ns3::QueueDiscItem>::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3QueueItem_methods(root_module, cls):
    cls.add_output_stream_operator()
    ## queue-item.h (module 'network'): ns3::QueueItem::QueueItem(ns3::Ptr<ns3::Packet> p) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::Packet >', 'p')])
    ## queue-item.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::QueueItem::GetPacket() const [member function]
    cls.add_method('GetPacket', 
                   'ns3::Ptr< ns3::Packet >', 
                   [], 
                   is_const=True)
    ## queue-item.h (module 'network'): uint32_t ns3::QueueItem::GetSize() const [member function]
    cls.add_method('GetSize', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## queue-item.h (module 'network'): bool ns3::QueueItem::GetUint8Value(ns3::QueueItem::Uint8Values field, uint8_t & value) const [member function]
    cls.add_method('GetUint8Value', 
                   'bool', 
                   [param('ns3::QueueItem::Uint8Values', 'field'), param('uint8_t &', 'value')], 
                   is_const=True, is_virtual=True)
    ## queue-item.h (module 'network'): void ns3::QueueItem::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    return

def register_Ns3QueueSizeChecker_methods(root_module, cls):
    ## queue-size.h (module 'network'): ns3::QueueSizeChecker::QueueSizeChecker() [constructor]
    cls.add_constructor([])
    ## queue-size.h (module 'network'): ns3::QueueSizeChecker::QueueSizeChecker(ns3::QueueSizeChecker const & arg0) [constructor]
    cls.add_constructor([param('ns3::QueueSizeChecker const &', 'arg0')])
    return

def register_Ns3QueueSizeValue_methods(root_module, cls):
    ## queue-size.h (module 'network'): ns3::QueueSizeValue::QueueSizeValue() [constructor]
    cls.add_constructor([])
    ## queue-size.h (module 'network'): ns3::QueueSizeValue::QueueSizeValue(ns3::QueueSize const & value) [constructor]
    cls.add_constructor([param('ns3::QueueSize const &', 'value')])
    ## queue-size.h (module 'network'): ns3::QueueSizeValue::QueueSizeValue(ns3::QueueSizeValue const & arg0) [constructor]
    cls.add_constructor([param('ns3::QueueSizeValue const &', 'arg0')])
    ## queue-size.h (module 'network'): ns3::Ptr<ns3::AttributeValue> ns3::QueueSizeValue::Copy() const [member function]
    cls.add_method('Copy', 
                   'ns3::Ptr< ns3::AttributeValue >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## queue-size.h (module 'network'): bool ns3::QueueSizeValue::DeserializeFromString(std::string value, ns3::Ptr<const ns3::AttributeChecker> checker) [member function]
    cls.add_method('DeserializeFromString', 
                   'bool', 
                   [param('std::string', 'value'), param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_virtual=True)
    ## queue-size.h (module 'network'): ns3::QueueSize ns3::QueueSizeValue::Get() const [member function]
    cls.add_method('Get', 
                   'ns3::QueueSize', 
                   [], 
                   is_const=True)
    ## queue-size.h (module 'network'): std::string ns3::QueueSizeValue::SerializeToString(ns3::Ptr<const ns3::AttributeChecker> checker) const [member function]
    cls.add_method('SerializeToString', 
                   'std::string', 
                   [param('ns3::Ptr< ns3::AttributeChecker const >', 'checker')], 
                   is_const=True, is_virtual=True)
    ## queue-size.h (module 'network'): void ns3::QueueSizeValue::Set(ns3::QueueSize const & value) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('ns3::QueueSize const &', 'value')])
    return

def register_Ns3RateErrorModel_methods(root_module, cls):
    ## error-model.h (module 'network'): ns3::RateErrorModel::RateErrorModel(ns3::RateErrorModel const & arg0) [constructor]
    cls.add_constructor([param('ns3::RateErrorModel const &', 'arg0')])
    ## error-model.h (module 'network'): ns3::RateErrorModel::RateErrorModel() [constructor]
    cls.add_constructor([])
    ## error-model.h (module 'network'): int64_t ns3::RateErrorModel::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')])
    ## error-model.h (module 'network'): double ns3::RateErrorModel::GetRate() const [member function]
    cls.add_method('GetRate', 
                   'double', 
                   [], 
                   is_const=True)
    ## error-model.h (module 'network'): static ns3::TypeId ns3::RateErrorModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## error-model.h (module 'network'): ns3::RateErrorModel::ErrorUnit ns3::RateErrorModel::GetUnit() const [member function]
    cls.add_method('GetUnit', 
                   'ns3::RateErrorModel::ErrorUnit', 
                   [], 
                   is_const=True)
    ## error-model.h (module 'network'): void ns3::RateErrorModel::SetRandomVariable(ns3::Ptr<ns3::RandomVariableStream> arg0) [member function]
    cls.add_method('SetRandomVariable', 
                   'void', 
                   [param('ns3::Ptr< ns3::RandomVariableStream >', 'arg0')])
    ## error-model.h (module 'network'): void ns3::RateErrorModel::SetRate(double rate) [member function]
    cls.add_method('SetRate', 
                   'void', 
                   [param('double', 'rate')])
    ## error-model.h (module 'network'): void ns3::RateErrorModel::SetUnit(ns3::RateErrorModel::ErrorUnit error_unit) [member function]
    cls.add_method('SetUnit', 
                   'void', 
                   [param('ns3::RateErrorModel::ErrorUnit', 'error_unit')])
    ## error-model.h (module 'network'): bool ns3::RateErrorModel::DoCorrupt(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('DoCorrupt', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')], 
                   is_virtual=True, visibility='private')
    ## error-model.h (module 'network'): bool ns3::RateErrorModel::DoCorruptBit(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('DoCorruptBit', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')], 
                   is_virtual=True, visibility='private')
    ## error-model.h (module 'network'): bool ns3::RateErrorModel::DoCorruptByte(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('DoCorruptByte', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')], 
                   is_virtual=True, visibility='private')
    ## error-model.h (module 'network'): bool ns3::RateErrorModel::DoCorruptPkt(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('DoCorruptPkt', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')], 
                   is_virtual=True, visibility='private')
    ## error-model.h (module 'network'): void ns3::RateErrorModel::DoReset() [member function]
    cls.add_method('DoReset', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3ReceiveListErrorModel_methods(root_module, cls):
    ## error-model.h (module 'network'): ns3::ReceiveListErrorModel::ReceiveListErrorModel(ns3::ReceiveListErrorModel const & arg0) [constructor]
    cls.add_constructor([param('ns3::ReceiveListErrorModel const &', 'arg0')])
    ## error-model.h (module 'network'): ns3::ReceiveListErrorModel::ReceiveListErrorModel() [constructor]
    cls.add_constructor([])
    ## error-model.h (module 'network'): std::list<unsigned int, std::allocator<unsigned int>> ns3::ReceiveListErrorModel::GetList() const [member function]
    cls.add_method('GetList', 
                   'std::list< unsigned int >', 
                   [], 
                   is_const=True)
    ## error-model.h (module 'network'): static ns3::TypeId ns3::ReceiveListErrorModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## error-model.h (module 'network'): void ns3::ReceiveListErrorModel::SetList(std::list<unsigned int, std::allocator<unsigned int>> const & packetlist) [member function]
    cls.add_method('SetList', 
                   'void', 
                   [param('std::list< unsigned int > const &', 'packetlist')])
    ## error-model.h (module 'network'): bool ns3::ReceiveListErrorModel::DoCorrupt(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('DoCorrupt', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')], 
                   is_virtual=True, visibility='private')
    ## error-model.h (module 'network'): void ns3::ReceiveListErrorModel::DoReset() [member function]
    cls.add_method('DoReset', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3SimpleChannel_methods(root_module, cls):
    ## simple-channel.h (module 'network'): ns3::SimpleChannel::SimpleChannel(ns3::SimpleChannel const & arg0) [constructor]
    cls.add_constructor([param('ns3::SimpleChannel const &', 'arg0')])
    ## simple-channel.h (module 'network'): ns3::SimpleChannel::SimpleChannel() [constructor]
    cls.add_constructor([])
    ## simple-channel.h (module 'network'): void ns3::SimpleChannel::Add(ns3::Ptr<ns3::SimpleNetDevice> device) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::Ptr< ns3::SimpleNetDevice >', 'device')], 
                   is_virtual=True)
    ## simple-channel.h (module 'network'): void ns3::SimpleChannel::BlackList(ns3::Ptr<ns3::SimpleNetDevice> from, ns3::Ptr<ns3::SimpleNetDevice> to) [member function]
    cls.add_method('BlackList', 
                   'void', 
                   [param('ns3::Ptr< ns3::SimpleNetDevice >', 'from'), param('ns3::Ptr< ns3::SimpleNetDevice >', 'to')], 
                   is_virtual=True)
    ## simple-channel.h (module 'network'): ns3::Ptr<ns3::NetDevice> ns3::SimpleChannel::GetDevice(std::size_t i) const [member function]
    cls.add_method('GetDevice', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [param('std::size_t', 'i')], 
                   is_const=True, is_virtual=True)
    ## simple-channel.h (module 'network'): std::size_t ns3::SimpleChannel::GetNDevices() const [member function]
    cls.add_method('GetNDevices', 
                   'std::size_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-channel.h (module 'network'): static ns3::TypeId ns3::SimpleChannel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## simple-channel.h (module 'network'): void ns3::SimpleChannel::Send(ns3::Ptr<ns3::Packet> p, uint16_t protocol, ns3::Mac48Address to, ns3::Mac48Address from, ns3::Ptr<ns3::SimpleNetDevice> sender) [member function]
    cls.add_method('Send', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'p'), param('uint16_t', 'protocol'), param('ns3::Mac48Address', 'to'), param('ns3::Mac48Address', 'from'), param('ns3::Ptr< ns3::SimpleNetDevice >', 'sender')], 
                   is_virtual=True)
    ## simple-channel.h (module 'network'): void ns3::SimpleChannel::UnBlackList(ns3::Ptr<ns3::SimpleNetDevice> from, ns3::Ptr<ns3::SimpleNetDevice> to) [member function]
    cls.add_method('UnBlackList', 
                   'void', 
                   [param('ns3::Ptr< ns3::SimpleNetDevice >', 'from'), param('ns3::Ptr< ns3::SimpleNetDevice >', 'to')], 
                   is_virtual=True)
    return

def register_Ns3SimpleNetDevice_methods(root_module, cls):
    ## simple-net-device.h (module 'network'): ns3::SimpleNetDevice::SimpleNetDevice(ns3::SimpleNetDevice const & arg0) [constructor]
    cls.add_constructor([param('ns3::SimpleNetDevice const &', 'arg0')])
    ## simple-net-device.h (module 'network'): ns3::SimpleNetDevice::SimpleNetDevice() [constructor]
    cls.add_constructor([])
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::AddLinkChangeCallback(ns3::Callback<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> callback) [member function]
    cls.add_method('AddLinkChangeCallback', 
                   'void', 
                   [param('ns3::Callback< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'callback')], 
                   is_virtual=True)
    ## simple-net-device.h (module 'network'): ns3::Address ns3::SimpleNetDevice::GetAddress() const [member function]
    cls.add_method('GetAddress', 
                   'ns3::Address', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): ns3::Address ns3::SimpleNetDevice::GetBroadcast() const [member function]
    cls.add_method('GetBroadcast', 
                   'ns3::Address', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): ns3::Ptr<ns3::Channel> ns3::SimpleNetDevice::GetChannel() const [member function]
    cls.add_method('GetChannel', 
                   'ns3::Ptr< ns3::Channel >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): uint32_t ns3::SimpleNetDevice::GetIfIndex() const [member function]
    cls.add_method('GetIfIndex', 
                   'uint32_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): uint16_t ns3::SimpleNetDevice::GetMtu() const [member function]
    cls.add_method('GetMtu', 
                   'uint16_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): ns3::Address ns3::SimpleNetDevice::GetMulticast(ns3::Ipv4Address multicastGroup) const [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Address', 
                   [param('ns3::Ipv4Address', 'multicastGroup')], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): ns3::Address ns3::SimpleNetDevice::GetMulticast(ns3::Ipv6Address addr) const [member function]
    cls.add_method('GetMulticast', 
                   'ns3::Address', 
                   [param('ns3::Ipv6Address', 'addr')], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): ns3::Ptr<ns3::Node> ns3::SimpleNetDevice::GetNode() const [member function]
    cls.add_method('GetNode', 
                   'ns3::Ptr< ns3::Node >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): ns3::Ptr<ns3::Queue<ns3::Packet>> ns3::SimpleNetDevice::GetQueue() const [member function]
    cls.add_method('GetQueue', 
                   'ns3::Ptr< ns3::Queue< ns3::Packet > >', 
                   [], 
                   is_const=True)
    ## simple-net-device.h (module 'network'): static ns3::TypeId ns3::SimpleNetDevice::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## simple-net-device.h (module 'network'): bool ns3::SimpleNetDevice::IsBridge() const [member function]
    cls.add_method('IsBridge', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): bool ns3::SimpleNetDevice::IsBroadcast() const [member function]
    cls.add_method('IsBroadcast', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): bool ns3::SimpleNetDevice::IsLinkUp() const [member function]
    cls.add_method('IsLinkUp', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): bool ns3::SimpleNetDevice::IsMulticast() const [member function]
    cls.add_method('IsMulticast', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): bool ns3::SimpleNetDevice::IsPointToPoint() const [member function]
    cls.add_method('IsPointToPoint', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): bool ns3::SimpleNetDevice::NeedsArp() const [member function]
    cls.add_method('NeedsArp', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::Receive(ns3::Ptr<ns3::Packet> packet, uint16_t protocol, ns3::Mac48Address to, ns3::Mac48Address from) [member function]
    cls.add_method('Receive', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('uint16_t', 'protocol'), param('ns3::Mac48Address', 'to'), param('ns3::Mac48Address', 'from')])
    ## simple-net-device.h (module 'network'): bool ns3::SimpleNetDevice::Send(ns3::Ptr<ns3::Packet> packet, ns3::Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('Send', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_virtual=True)
    ## simple-net-device.h (module 'network'): bool ns3::SimpleNetDevice::SendFrom(ns3::Ptr<ns3::Packet> packet, ns3::Address const & source, ns3::Address const & dest, uint16_t protocolNumber) [member function]
    cls.add_method('SendFrom', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'packet'), param('ns3::Address const &', 'source'), param('ns3::Address const &', 'dest'), param('uint16_t', 'protocolNumber')], 
                   is_virtual=True)
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::SetAddress(ns3::Address address) [member function]
    cls.add_method('SetAddress', 
                   'void', 
                   [param('ns3::Address', 'address')], 
                   is_virtual=True)
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::SetChannel(ns3::Ptr<ns3::SimpleChannel> channel) [member function]
    cls.add_method('SetChannel', 
                   'void', 
                   [param('ns3::Ptr< ns3::SimpleChannel >', 'channel')])
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::SetIfIndex(uint32_t const index) [member function]
    cls.add_method('SetIfIndex', 
                   'void', 
                   [param('uint32_t const', 'index')], 
                   is_virtual=True)
    ## simple-net-device.h (module 'network'): bool ns3::SimpleNetDevice::SetMtu(uint16_t const mtu) [member function]
    cls.add_method('SetMtu', 
                   'bool', 
                   [param('uint16_t const', 'mtu')], 
                   is_virtual=True)
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::SetNode(ns3::Ptr<ns3::Node> node) [member function]
    cls.add_method('SetNode', 
                   'void', 
                   [param('ns3::Ptr< ns3::Node >', 'node')], 
                   is_virtual=True)
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::SetPromiscReceiveCallback(ns3::NetDevice::PromiscReceiveCallback cb) [member function]
    cls.add_method('SetPromiscReceiveCallback', 
                   'void', 
                   [param('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::Address const &, ns3::NetDevice::PacketType, ns3::empty, ns3::empty, ns3::empty >', 'cb')], 
                   is_virtual=True)
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::SetQueue(ns3::Ptr<ns3::Queue<ns3::Packet>> queue) [member function]
    cls.add_method('SetQueue', 
                   'void', 
                   [param('ns3::Ptr< ns3::Queue< ns3::Packet > >', 'queue')])
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::SetReceiveCallback(ns3::NetDevice::ReceiveCallback cb) [member function]
    cls.add_method('SetReceiveCallback', 
                   'void', 
                   [param('ns3::Callback< bool, ns3::Ptr< ns3::NetDevice >, ns3::Ptr< ns3::Packet const >, unsigned short, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >', 'cb')], 
                   is_virtual=True)
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::SetReceiveErrorModel(ns3::Ptr<ns3::ErrorModel> em) [member function]
    cls.add_method('SetReceiveErrorModel', 
                   'void', 
                   [param('ns3::Ptr< ns3::ErrorModel >', 'em')])
    ## simple-net-device.h (module 'network'): bool ns3::SimpleNetDevice::SupportsSendFrom() const [member function]
    cls.add_method('SupportsSendFrom', 
                   'bool', 
                   [], 
                   is_const=True, is_virtual=True)
    ## simple-net-device.h (module 'network'): void ns3::SimpleNetDevice::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
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

def register_Ns3BinaryErrorModel_methods(root_module, cls):
    ## error-model.h (module 'network'): ns3::BinaryErrorModel::BinaryErrorModel(ns3::BinaryErrorModel const & arg0) [constructor]
    cls.add_constructor([param('ns3::BinaryErrorModel const &', 'arg0')])
    ## error-model.h (module 'network'): ns3::BinaryErrorModel::BinaryErrorModel() [constructor]
    cls.add_constructor([])
    ## error-model.h (module 'network'): static ns3::TypeId ns3::BinaryErrorModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## error-model.h (module 'network'): bool ns3::BinaryErrorModel::DoCorrupt(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('DoCorrupt', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')], 
                   is_virtual=True, visibility='private')
    ## error-model.h (module 'network'): void ns3::BinaryErrorModel::DoReset() [member function]
    cls.add_method('DoReset', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
    return

def register_Ns3BurstErrorModel_methods(root_module, cls):
    ## error-model.h (module 'network'): ns3::BurstErrorModel::BurstErrorModel(ns3::BurstErrorModel const & arg0) [constructor]
    cls.add_constructor([param('ns3::BurstErrorModel const &', 'arg0')])
    ## error-model.h (module 'network'): ns3::BurstErrorModel::BurstErrorModel() [constructor]
    cls.add_constructor([])
    ## error-model.h (module 'network'): int64_t ns3::BurstErrorModel::AssignStreams(int64_t stream) [member function]
    cls.add_method('AssignStreams', 
                   'int64_t', 
                   [param('int64_t', 'stream')])
    ## error-model.h (module 'network'): double ns3::BurstErrorModel::GetBurstRate() const [member function]
    cls.add_method('GetBurstRate', 
                   'double', 
                   [], 
                   is_const=True)
    ## error-model.h (module 'network'): static ns3::TypeId ns3::BurstErrorModel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## error-model.h (module 'network'): void ns3::BurstErrorModel::SetBurstRate(double rate) [member function]
    cls.add_method('SetBurstRate', 
                   'void', 
                   [param('double', 'rate')])
    ## error-model.h (module 'network'): void ns3::BurstErrorModel::SetRandomBurstSize(ns3::Ptr<ns3::RandomVariableStream> burstSz) [member function]
    cls.add_method('SetRandomBurstSize', 
                   'void', 
                   [param('ns3::Ptr< ns3::RandomVariableStream >', 'burstSz')])
    ## error-model.h (module 'network'): void ns3::BurstErrorModel::SetRandomVariable(ns3::Ptr<ns3::RandomVariableStream> ranVar) [member function]
    cls.add_method('SetRandomVariable', 
                   'void', 
                   [param('ns3::Ptr< ns3::RandomVariableStream >', 'ranVar')])
    ## error-model.h (module 'network'): bool ns3::BurstErrorModel::DoCorrupt(ns3::Ptr<ns3::Packet> p) [member function]
    cls.add_method('DoCorrupt', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'p')], 
                   is_virtual=True, visibility='private')
    ## error-model.h (module 'network'): void ns3::BurstErrorModel::DoReset() [member function]
    cls.add_method('DoReset', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='private')
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

def register_Ns3CallbackImpl__Bool_Ns3Ptr__lt__ns3Socket__gt___Const_ns3Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<bool, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<bool, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< bool, ns3::Ptr< ns3::Socket >, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<bool, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<bool, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): bool ns3::CallbackImpl<bool, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<ns3::Socket> arg0, ns3::Address const & arg1) [member operator]
    cls.add_method('operator()', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Socket >', 'arg0'), param('ns3::Address const &', 'arg1')], 
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

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3Packet__gt___Const_ns3Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::Packet const >, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<const ns3::Packet>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<const ns3::Packet> arg0, ns3::Address const & arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'arg0'), param('ns3::Address const &', 'arg1')], 
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

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__const_ns3QueueDiscItem__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::QueueDiscItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<const ns3::QueueDiscItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<const ns3::QueueDiscItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::QueueDiscItem const >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::QueueDiscItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<const ns3::QueueDiscItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<const ns3::QueueDiscItem>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<const ns3::QueueDiscItem> arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::QueueDiscItem const >', 'arg0')], 
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

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3Socket__gt___Const_ns3Address___amp___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::Socket >, ns3::Address const &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, const ns3::Address &, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<ns3::Socket> arg0, ns3::Address const & arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::Socket >', 'arg0'), param('ns3::Address const &', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3Socket__gt___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::Socket >, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<ns3::Socket> arg0) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::Socket >', 'arg0')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Ptr__lt__ns3Socket__gt___Unsigned_int_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::Ptr< ns3::Socket >, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::Ptr<ns3::Socket>, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(ns3::Ptr<ns3::Socket> arg0, unsigned int arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('ns3::Ptr< ns3::Socket >', 'arg0'), param('unsigned int', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()() [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Unsigned_int_Unsigned_int_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(unsigned int arg0, unsigned int arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('unsigned int', 'arg0'), param('unsigned int', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CounterCalculator__Unsigned_int_methods(root_module, cls):
    ## basic-data-calculators.h (module 'stats'): ns3::CounterCalculator<unsigned int>::CounterCalculator(ns3::CounterCalculator<unsigned int> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CounterCalculator< unsigned int > const &', 'arg0')])
    ## basic-data-calculators.h (module 'stats'): ns3::CounterCalculator<unsigned int>::CounterCalculator() [constructor]
    cls.add_constructor([])
    ## basic-data-calculators.h (module 'stats'): unsigned int ns3::CounterCalculator<unsigned int>::GetCount() const [member function]
    cls.add_method('GetCount', 
                   'unsigned int', 
                   [], 
                   is_const=True)
    ## basic-data-calculators.h (module 'stats'): static ns3::TypeId ns3::CounterCalculator<unsigned int>::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## basic-data-calculators.h (module 'stats'): void ns3::CounterCalculator<unsigned int>::Output(ns3::DataOutputCallback & callback) const [member function]
    cls.add_method('Output', 
                   'void', 
                   [param('ns3::DataOutputCallback &', 'callback')], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): void ns3::CounterCalculator<unsigned int>::Update() [member function]
    cls.add_method('Update', 
                   'void', 
                   [])
    ## basic-data-calculators.h (module 'stats'): void ns3::CounterCalculator<unsigned int>::Update(unsigned int const i) [member function]
    cls.add_method('Update', 
                   'void', 
                   [param('unsigned int const', 'i')])
    ## basic-data-calculators.h (module 'stats'): void ns3::CounterCalculator<unsigned int>::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3DropTailQueue__Ns3Packet_methods(root_module, cls):
    ## drop-tail-queue.h (module 'network'): static ns3::TypeId ns3::DropTailQueue<ns3::Packet>::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## drop-tail-queue.h (module 'network'): ns3::DropTailQueue<ns3::Packet>::DropTailQueue() [constructor]
    cls.add_constructor([])
    ## drop-tail-queue.h (module 'network'): bool ns3::DropTailQueue<ns3::Packet>::Enqueue(ns3::Ptr<ns3::Packet> item) [member function]
    cls.add_method('Enqueue', 
                   'bool', 
                   [param('ns3::Ptr< ns3::Packet >', 'item')], 
                   is_virtual=True)
    ## drop-tail-queue.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::DropTailQueue<ns3::Packet>::Dequeue() [member function]
    cls.add_method('Dequeue', 
                   'ns3::Ptr< ns3::Packet >', 
                   [], 
                   is_virtual=True)
    ## drop-tail-queue.h (module 'network'): ns3::Ptr<ns3::Packet> ns3::DropTailQueue<ns3::Packet>::Remove() [member function]
    cls.add_method('Remove', 
                   'ns3::Ptr< ns3::Packet >', 
                   [], 
                   is_virtual=True)
    ## drop-tail-queue.h (module 'network'): ns3::Ptr<const ns3::Packet> ns3::DropTailQueue<ns3::Packet>::Peek() const [member function]
    cls.add_method('Peek', 
                   'ns3::Ptr< ns3::Packet const >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## drop-tail-queue.h (module 'network'): ns3::DropTailQueue<ns3::Packet>::DropTailQueue(ns3::DropTailQueue<ns3::Packet> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DropTailQueue< ns3::Packet > const &', 'arg0')])
    return

def register_Ns3DropTailQueue__Ns3QueueDiscItem_methods(root_module, cls):
    ## drop-tail-queue.h (module 'network'): static ns3::TypeId ns3::DropTailQueue<ns3::QueueDiscItem>::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## drop-tail-queue.h (module 'network'): ns3::DropTailQueue<ns3::QueueDiscItem>::DropTailQueue() [constructor]
    cls.add_constructor([])
    ## drop-tail-queue.h (module 'network'): bool ns3::DropTailQueue<ns3::QueueDiscItem>::Enqueue(ns3::Ptr<ns3::QueueDiscItem> item) [member function]
    cls.add_method('Enqueue', 
                   'bool', 
                   [param('ns3::Ptr< ns3::QueueDiscItem >', 'item')], 
                   is_virtual=True)
    ## drop-tail-queue.h (module 'network'): ns3::Ptr<ns3::QueueDiscItem> ns3::DropTailQueue<ns3::QueueDiscItem>::Dequeue() [member function]
    cls.add_method('Dequeue', 
                   'ns3::Ptr< ns3::QueueDiscItem >', 
                   [], 
                   is_virtual=True)
    ## drop-tail-queue.h (module 'network'): ns3::Ptr<ns3::QueueDiscItem> ns3::DropTailQueue<ns3::QueueDiscItem>::Remove() [member function]
    cls.add_method('Remove', 
                   'ns3::Ptr< ns3::QueueDiscItem >', 
                   [], 
                   is_virtual=True)
    ## drop-tail-queue.h (module 'network'): ns3::Ptr<const ns3::QueueDiscItem> ns3::DropTailQueue<ns3::QueueDiscItem>::Peek() const [member function]
    cls.add_method('Peek', 
                   'ns3::Ptr< ns3::QueueDiscItem const >', 
                   [], 
                   is_const=True, is_virtual=True)
    ## drop-tail-queue.h (module 'network'): ns3::DropTailQueue<ns3::QueueDiscItem>::DropTailQueue(ns3::DropTailQueue<ns3::QueueDiscItem> const & arg0) [constructor]
    cls.add_constructor([param('ns3::DropTailQueue< ns3::QueueDiscItem > const &', 'arg0')])
    return

def register_Ns3ErrorChannel_methods(root_module, cls):
    ## error-channel.h (module 'network'): ns3::ErrorChannel::ErrorChannel(ns3::ErrorChannel const & arg0) [constructor]
    cls.add_constructor([param('ns3::ErrorChannel const &', 'arg0')])
    ## error-channel.h (module 'network'): ns3::ErrorChannel::ErrorChannel() [constructor]
    cls.add_constructor([])
    ## error-channel.h (module 'network'): void ns3::ErrorChannel::Add(ns3::Ptr<ns3::SimpleNetDevice> device) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('ns3::Ptr< ns3::SimpleNetDevice >', 'device')], 
                   is_virtual=True)
    ## error-channel.h (module 'network'): ns3::Ptr<ns3::NetDevice> ns3::ErrorChannel::GetDevice(std::size_t i) const [member function]
    cls.add_method('GetDevice', 
                   'ns3::Ptr< ns3::NetDevice >', 
                   [param('std::size_t', 'i')], 
                   is_const=True, is_virtual=True)
    ## error-channel.h (module 'network'): std::size_t ns3::ErrorChannel::GetNDevices() const [member function]
    cls.add_method('GetNDevices', 
                   'std::size_t', 
                   [], 
                   is_const=True, is_virtual=True)
    ## error-channel.h (module 'network'): static ns3::TypeId ns3::ErrorChannel::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## error-channel.h (module 'network'): void ns3::ErrorChannel::Send(ns3::Ptr<ns3::Packet> p, uint16_t protocol, ns3::Mac48Address to, ns3::Mac48Address from, ns3::Ptr<ns3::SimpleNetDevice> sender) [member function]
    cls.add_method('Send', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet >', 'p'), param('uint16_t', 'protocol'), param('ns3::Mac48Address', 'to'), param('ns3::Mac48Address', 'from'), param('ns3::Ptr< ns3::SimpleNetDevice >', 'sender')], 
                   is_virtual=True)
    ## error-channel.h (module 'network'): void ns3::ErrorChannel::SetDuplicateMode(bool mode) [member function]
    cls.add_method('SetDuplicateMode', 
                   'void', 
                   [param('bool', 'mode')])
    ## error-channel.h (module 'network'): void ns3::ErrorChannel::SetDuplicateTime(ns3::Time delay) [member function]
    cls.add_method('SetDuplicateTime', 
                   'void', 
                   [param('ns3::Time', 'delay')])
    ## error-channel.h (module 'network'): void ns3::ErrorChannel::SetJumpingMode(bool mode) [member function]
    cls.add_method('SetJumpingMode', 
                   'void', 
                   [param('bool', 'mode')])
    ## error-channel.h (module 'network'): void ns3::ErrorChannel::SetJumpingTime(ns3::Time delay) [member function]
    cls.add_method('SetJumpingTime', 
                   'void', 
                   [param('ns3::Time', 'delay')])
    return

def register_Ns3PacketCounterCalculator_methods(root_module, cls):
    ## packet-data-calculators.h (module 'network'): ns3::PacketCounterCalculator::PacketCounterCalculator(ns3::PacketCounterCalculator const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketCounterCalculator const &', 'arg0')])
    ## packet-data-calculators.h (module 'network'): ns3::PacketCounterCalculator::PacketCounterCalculator() [constructor]
    cls.add_constructor([])
    ## packet-data-calculators.h (module 'network'): void ns3::PacketCounterCalculator::FrameUpdate(std::string path, ns3::Ptr<const ns3::Packet> packet, ns3::Mac48Address realto) [member function]
    cls.add_method('FrameUpdate', 
                   'void', 
                   [param('std::string', 'path'), param('ns3::Ptr< ns3::Packet const >', 'packet'), param('ns3::Mac48Address', 'realto')])
    ## packet-data-calculators.h (module 'network'): static ns3::TypeId ns3::PacketCounterCalculator::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## packet-data-calculators.h (module 'network'): void ns3::PacketCounterCalculator::PacketUpdate(std::string path, ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('PacketUpdate', 
                   'void', 
                   [param('std::string', 'path'), param('ns3::Ptr< ns3::Packet const >', 'packet')])
    ## packet-data-calculators.h (module 'network'): void ns3::PacketCounterCalculator::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3PacketProbe_methods(root_module, cls):
    ## packet-probe.h (module 'network'): ns3::PacketProbe::PacketProbe(ns3::PacketProbe const & arg0) [constructor]
    cls.add_constructor([param('ns3::PacketProbe const &', 'arg0')])
    ## packet-probe.h (module 'network'): ns3::PacketProbe::PacketProbe() [constructor]
    cls.add_constructor([])
    ## packet-probe.h (module 'network'): bool ns3::PacketProbe::ConnectByObject(std::string traceSource, ns3::Ptr<ns3::Object> obj) [member function]
    cls.add_method('ConnectByObject', 
                   'bool', 
                   [param('std::string', 'traceSource'), param('ns3::Ptr< ns3::Object >', 'obj')], 
                   is_virtual=True)
    ## packet-probe.h (module 'network'): void ns3::PacketProbe::ConnectByPath(std::string path) [member function]
    cls.add_method('ConnectByPath', 
                   'void', 
                   [param('std::string', 'path')], 
                   is_virtual=True)
    ## packet-probe.h (module 'network'): static ns3::TypeId ns3::PacketProbe::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## packet-probe.h (module 'network'): void ns3::PacketProbe::SetValue(ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('SetValue', 
                   'void', 
                   [param('ns3::Ptr< ns3::Packet const >', 'packet')])
    ## packet-probe.h (module 'network'): static void ns3::PacketProbe::SetValueByPath(std::string path, ns3::Ptr<const ns3::Packet> packet) [member function]
    cls.add_method('SetValueByPath', 
                   'void', 
                   [param('std::string', 'path'), param('ns3::Ptr< ns3::Packet const >', 'packet')], 
                   is_static=True)
    return

def register_Ns3PbbAddressTlv_methods(root_module, cls):
    ## packetbb.h (module 'network'): ns3::PbbAddressTlv::PbbAddressTlv() [constructor]
    cls.add_constructor([])
    ## packetbb.h (module 'network'): ns3::PbbAddressTlv::PbbAddressTlv(ns3::PbbAddressTlv const & arg0) [constructor]
    cls.add_constructor([param('ns3::PbbAddressTlv const &', 'arg0')])
    ## packetbb.h (module 'network'): uint8_t ns3::PbbAddressTlv::GetIndexStart() const [member function]
    cls.add_method('GetIndexStart', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): uint8_t ns3::PbbAddressTlv::GetIndexStop() const [member function]
    cls.add_method('GetIndexStop', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): bool ns3::PbbAddressTlv::HasIndexStart() const [member function]
    cls.add_method('HasIndexStart', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): bool ns3::PbbAddressTlv::HasIndexStop() const [member function]
    cls.add_method('HasIndexStop', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): bool ns3::PbbAddressTlv::IsMultivalue() const [member function]
    cls.add_method('IsMultivalue', 
                   'bool', 
                   [], 
                   is_const=True)
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlv::SetIndexStart(uint8_t index) [member function]
    cls.add_method('SetIndexStart', 
                   'void', 
                   [param('uint8_t', 'index')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlv::SetIndexStop(uint8_t index) [member function]
    cls.add_method('SetIndexStop', 
                   'void', 
                   [param('uint8_t', 'index')])
    ## packetbb.h (module 'network'): void ns3::PbbAddressTlv::SetMultivalue(bool isMultivalue) [member function]
    cls.add_method('SetMultivalue', 
                   'void', 
                   [param('bool', 'isMultivalue')])
    return

def register_Ns3QueueDiscItem_methods(root_module, cls):
    ## queue-item.h (module 'network'): ns3::QueueDiscItem::QueueDiscItem(ns3::Ptr<ns3::Packet> p, ns3::Address const & addr, uint16_t protocol) [constructor]
    cls.add_constructor([param('ns3::Ptr< ns3::Packet >', 'p'), param('ns3::Address const &', 'addr'), param('uint16_t', 'protocol')])
    ## queue-item.h (module 'network'): ns3::Address ns3::QueueDiscItem::GetAddress() const [member function]
    cls.add_method('GetAddress', 
                   'ns3::Address', 
                   [], 
                   is_const=True)
    ## queue-item.h (module 'network'): uint16_t ns3::QueueDiscItem::GetProtocol() const [member function]
    cls.add_method('GetProtocol', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## queue-item.h (module 'network'): uint8_t ns3::QueueDiscItem::GetTxQueueIndex() const [member function]
    cls.add_method('GetTxQueueIndex', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## queue-item.h (module 'network'): void ns3::QueueDiscItem::SetTxQueueIndex(uint8_t txq) [member function]
    cls.add_method('SetTxQueueIndex', 
                   'void', 
                   [param('uint8_t', 'txq')])
    ## queue-item.h (module 'network'): ns3::Time ns3::QueueDiscItem::GetTimeStamp() const [member function]
    cls.add_method('GetTimeStamp', 
                   'ns3::Time', 
                   [], 
                   is_const=True)
    ## queue-item.h (module 'network'): void ns3::QueueDiscItem::SetTimeStamp(ns3::Time t) [member function]
    cls.add_method('SetTimeStamp', 
                   'void', 
                   [param('ns3::Time', 't')])
    ## queue-item.h (module 'network'): void ns3::QueueDiscItem::AddHeader() [member function]
    cls.add_method('AddHeader', 
                   'void', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## queue-item.h (module 'network'): void ns3::QueueDiscItem::Print(std::ostream & os) const [member function]
    cls.add_method('Print', 
                   'void', 
                   [param('std::ostream &', 'os')], 
                   is_const=True, is_virtual=True)
    ## queue-item.h (module 'network'): bool ns3::QueueDiscItem::Mark() [member function]
    cls.add_method('Mark', 
                   'bool', 
                   [], 
                   is_pure_virtual=True, is_virtual=True)
    ## queue-item.h (module 'network'): uint32_t ns3::QueueDiscItem::Hash(uint32_t perturbation=0) const [member function]
    cls.add_method('Hash', 
                   'uint32_t', 
                   [param('uint32_t', 'perturbation', default_value='0')], 
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

def register_functions(root_module):
    module = root_module
    ## crc32.h (module 'network'): uint32_t ns3::CRC32Calculate(uint8_t const * data, int length) [free function]
    module.add_function('CRC32Calculate', 
                        'uint32_t', 
                        [param('uint8_t const *', 'data'), param('int', 'length')])
    ## address.h (module 'network'): ns3::Ptr<const ns3::AttributeChecker> ns3::MakeAddressChecker() [free function]
    module.add_function('MakeAddressChecker', 
                        'ns3::Ptr< ns3::AttributeChecker const >', 
                        [])
    ## data-rate.h (module 'network'): ns3::Ptr<const ns3::AttributeChecker> ns3::MakeDataRateChecker() [free function]
    module.add_function('MakeDataRateChecker', 
                        'ns3::Ptr< ns3::AttributeChecker const >', 
                        [])
    ## ipv4-address.h (module 'network'): ns3::Ptr<const ns3::AttributeChecker> ns3::MakeIpv4AddressChecker() [free function]
    module.add_function('MakeIpv4AddressChecker', 
                        'ns3::Ptr< ns3::AttributeChecker const >', 
                        [])
    ## ipv4-address.h (module 'network'): ns3::Ptr<const ns3::AttributeChecker> ns3::MakeIpv4MaskChecker() [free function]
    module.add_function('MakeIpv4MaskChecker', 
                        'ns3::Ptr< ns3::AttributeChecker const >', 
                        [])
    ## ipv6-address.h (module 'network'): ns3::Ptr<const ns3::AttributeChecker> ns3::MakeIpv6AddressChecker() [free function]
    module.add_function('MakeIpv6AddressChecker', 
                        'ns3::Ptr< ns3::AttributeChecker const >', 
                        [])
    ## ipv6-address.h (module 'network'): ns3::Ptr<const ns3::AttributeChecker> ns3::MakeIpv6PrefixChecker() [free function]
    module.add_function('MakeIpv6PrefixChecker', 
                        'ns3::Ptr< ns3::AttributeChecker const >', 
                        [])
    ## mac16-address.h (module 'network'): ns3::Ptr<const ns3::AttributeChecker> ns3::MakeMac16AddressChecker() [free function]
    module.add_function('MakeMac16AddressChecker', 
                        'ns3::Ptr< ns3::AttributeChecker const >', 
                        [])
    ## mac48-address.h (module 'network'): ns3::Ptr<const ns3::AttributeChecker> ns3::MakeMac48AddressChecker() [free function]
    module.add_function('MakeMac48AddressChecker', 
                        'ns3::Ptr< ns3::AttributeChecker const >', 
                        [])
    ## mac64-address.h (module 'network'): ns3::Ptr<const ns3::AttributeChecker> ns3::MakeMac64AddressChecker() [free function]
    module.add_function('MakeMac64AddressChecker', 
                        'ns3::Ptr< ns3::AttributeChecker const >', 
                        [])
    ## queue-size.h (module 'network'): ns3::Ptr<const ns3::AttributeAccessor> ns3::MakeQueueSizeAccessor(void ( ::ns3::QueueBase::* )( ::ns3::QueueSize ) a1, ns3::QueueSize ( ::ns3::QueueBase::* )(  )const a2) [free function]
    module.add_function('MakeQueueSizeAccessor', 
                        'ns3::Ptr< ns3::AttributeAccessor const >', 
                        [param('void ( ns3::QueueBase:: * ) ( ns3::QueueSize )', 'a1'), param('ns3::QueueSize ( ns3::QueueBase:: * ) (  ) const', 'a2')], 
                        template_parameters=['void (ns3::QueueBase::*)(ns3::QueueSize)', ' ns3::QueueSize (ns3::QueueBase::*)() const'])
    ## queue-size.h (module 'network'): ns3::Ptr<const ns3::AttributeChecker> ns3::MakeQueueSizeChecker() [free function]
    module.add_function('MakeQueueSizeChecker', 
                        'ns3::Ptr< ns3::AttributeChecker const >', 
                        [])
    ## address-utils.h (module 'network'): void ns3::ReadFrom(ns3::Buffer::Iterator & i, ns3::Address & ad, uint32_t len) [free function]
    module.add_function('ReadFrom', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Address &', 'ad'), param('uint32_t', 'len')])
    ## address-utils.h (module 'network'): void ns3::ReadFrom(ns3::Buffer::Iterator & i, ns3::Ipv4Address & ad) [free function]
    module.add_function('ReadFrom', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Ipv4Address &', 'ad')])
    ## address-utils.h (module 'network'): void ns3::ReadFrom(ns3::Buffer::Iterator & i, ns3::Ipv6Address & ad) [free function]
    module.add_function('ReadFrom', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Ipv6Address &', 'ad')])
    ## address-utils.h (module 'network'): void ns3::ReadFrom(ns3::Buffer::Iterator & i, ns3::Mac16Address & ad) [free function]
    module.add_function('ReadFrom', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Mac16Address &', 'ad')])
    ## address-utils.h (module 'network'): void ns3::ReadFrom(ns3::Buffer::Iterator & i, ns3::Mac48Address & ad) [free function]
    module.add_function('ReadFrom', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Mac48Address &', 'ad')])
    ## address-utils.h (module 'network'): void ns3::ReadFrom(ns3::Buffer::Iterator & i, ns3::Mac64Address & ad) [free function]
    module.add_function('ReadFrom', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Mac64Address &', 'ad')])
    ## address-utils.h (module 'network'): void ns3::WriteTo(ns3::Buffer::Iterator & i, ns3::Address const & ad) [free function]
    module.add_function('WriteTo', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Address const &', 'ad')])
    ## address-utils.h (module 'network'): void ns3::WriteTo(ns3::Buffer::Iterator & i, ns3::Ipv4Address ad) [free function]
    module.add_function('WriteTo', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Ipv4Address', 'ad')])
    ## address-utils.h (module 'network'): void ns3::WriteTo(ns3::Buffer::Iterator & i, ns3::Ipv6Address ad) [free function]
    module.add_function('WriteTo', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Ipv6Address', 'ad')])
    ## address-utils.h (module 'network'): void ns3::WriteTo(ns3::Buffer::Iterator & i, ns3::Mac16Address ad) [free function]
    module.add_function('WriteTo', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Mac16Address', 'ad')])
    ## address-utils.h (module 'network'): void ns3::WriteTo(ns3::Buffer::Iterator & i, ns3::Mac48Address ad) [free function]
    module.add_function('WriteTo', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Mac48Address', 'ad')])
    ## address-utils.h (module 'network'): void ns3::WriteTo(ns3::Buffer::Iterator & i, ns3::Mac64Address ad) [free function]
    module.add_function('WriteTo', 
                        'void', 
                        [param('ns3::Buffer::Iterator &', 'i'), param('ns3::Mac64Address', 'ad')])
    register_functions_ns3_FatalImpl(module.add_cpp_namespace('FatalImpl'), root_module)
    register_functions_ns3_Hash(module.add_cpp_namespace('Hash'), root_module)
    register_functions_ns3_TracedValueCallback(module.add_cpp_namespace('TracedValueCallback'), root_module)
    register_functions_ns3_addressUtils(module.add_cpp_namespace('addressUtils'), root_module)
    register_functions_ns3_internal(module.add_cpp_namespace('internal'), root_module)
    register_functions_ns3_tests(module.add_cpp_namespace('tests'), root_module)
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

def register_functions_ns3_addressUtils(module, root_module):
    ## address-utils.h (module 'network'): bool ns3::addressUtils::IsMulticast(ns3::Address const & ad) [free function]
    module.add_function('IsMulticast', 
                        'bool', 
                        [param('ns3::Address const &', 'ad')])
    return

def register_functions_ns3_internal(module, root_module):
    return

def register_functions_ns3_tests(module, root_module):
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

