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
    root_module = Module('ns.stats', cpp_namespace='::ns3')
    return root_module

def register_types(module):
    root_module = module.get_root()
    
    ## log.h (module 'core'): ns3::LogLevel [enumeration]
    module.add_enum('LogLevel', ['LOG_NONE', 'LOG_ERROR', 'LOG_LEVEL_ERROR', 'LOG_WARN', 'LOG_LEVEL_WARN', 'LOG_DEBUG', 'LOG_LEVEL_DEBUG', 'LOG_INFO', 'LOG_LEVEL_INFO', 'LOG_FUNCTION', 'LOG_LEVEL_FUNCTION', 'LOG_LOGIC', 'LOG_LEVEL_LOGIC', 'LOG_ALL', 'LOG_LEVEL_ALL', 'LOG_PREFIX_FUNC', 'LOG_PREFIX_TIME', 'LOG_PREFIX_NODE', 'LOG_PREFIX_LEVEL', 'LOG_PREFIX_ALL'], import_from_module='ns.core')
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList [class]
    module.add_class('AttributeConstructionList', import_from_module='ns.core')
    ## attribute-construction-list.h (module 'core'): ns3::AttributeConstructionList::Item [struct]
    module.add_class('Item', import_from_module='ns.core', outer_class=root_module['ns3::AttributeConstructionList'])
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator', 'ns3::AttributeConstructionList::CIterator')
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator*', 'ns3::AttributeConstructionList::CIterator*')
    typehandlers.add_type_alias('std::list< ns3::AttributeConstructionList::Item > const_iterator&', 'ns3::AttributeConstructionList::CIterator&')
    ## callback.h (module 'core'): ns3::CallbackBase [class]
    module.add_class('CallbackBase', import_from_module='ns.core')
    ## data-output-interface.h (module 'stats'): ns3::DataOutputCallback [class]
    module.add_class('DataOutputCallback', allow_subclassing=True)
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
    ## default-deleter.h (module 'core'): ns3::DefaultDeleter<ns3::TraceSourceAccessor> [struct]
    module.add_class('DefaultDeleter', import_from_module='ns.core', template_parameters=['ns3::TraceSourceAccessor'])
    ## event-id.h (module 'core'): ns3::EventId [class]
    module.add_class('EventId', import_from_module='ns.core')
    ## event-id.h (module 'core'): ns3::EventId::UID [enumeration]
    module.add_enum('UID', ['INVALID', 'NOW', 'DESTROY', 'RESERVED', 'VALID'], outer_class=root_module['ns3::EventId'], import_from_module='ns.core')
    ## file-helper.h (module 'stats'): ns3::FileHelper [class]
    module.add_class('FileHelper')
    ## gnuplot.h (module 'stats'): ns3::Gnuplot [class]
    module.add_class('Gnuplot')
    ## gnuplot.h (module 'stats'): ns3::GnuplotCollection [class]
    module.add_class('GnuplotCollection')
    ## gnuplot.h (module 'stats'): ns3::GnuplotDataset [class]
    module.add_class('GnuplotDataset')
    ## gnuplot-helper.h (module 'stats'): ns3::GnuplotHelper [class]
    module.add_class('GnuplotHelper')
    ## hash.h (module 'core'): ns3::Hasher [class]
    module.add_class('Hasher', import_from_module='ns.core')
    ## histogram.h (module 'stats'): ns3::Histogram [class]
    module.add_class('Histogram')
    ## log.h (module 'core'): ns3::LogComponent [class]
    module.add_class('LogComponent', import_from_module='ns.core')
    typehandlers.add_type_alias('std::map< std::string, ns3::LogComponent * >', 'ns3::LogComponent::ComponentList')
    typehandlers.add_type_alias('std::map< std::string, ns3::LogComponent * >*', 'ns3::LogComponent::ComponentList*')
    typehandlers.add_type_alias('std::map< std::string, ns3::LogComponent * >&', 'ns3::LogComponent::ComponentList&')
    ## object-base.h (module 'core'): ns3::ObjectBase [class]
    module.add_class('ObjectBase', allow_subclassing=True, import_from_module='ns.core')
    ## object.h (module 'core'): ns3::ObjectDeleter [struct]
    module.add_class('ObjectDeleter', import_from_module='ns.core')
    ## object-factory.h (module 'core'): ns3::ObjectFactory [class]
    module.add_class('ObjectFactory', import_from_module='ns.core')
    ## log.h (module 'core'): ns3::ParameterLogger [class]
    module.add_class('ParameterLogger', import_from_module='ns.core')
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::ObjectBase'], template_parameters=['ns3::Object', 'ns3::ObjectBase', 'ns3::ObjectDeleter'])
    ## simulator.h (module 'core'): ns3::Simulator [class]
    module.add_class('Simulator', destructor_visibility='private', import_from_module='ns.core')
    ## simulator.h (module 'core'): ns3::Simulator [enumeration]
    module.add_enum('', ['NO_CONTEXT'], outer_class=root_module['ns3::Simulator'], import_from_module='ns.core')
    ## data-calculator.h (module 'stats'): ns3::StatisticalSummary [class]
    module.add_class('StatisticalSummary', allow_subclassing=True)
    ## nstime.h (module 'core'): ns3::Time [class]
    module.add_class('Time', import_from_module='ns.core')
    ## nstime.h (module 'core'): ns3::Time::Unit [enumeration]
    module.add_enum('Unit', ['Y', 'D', 'H', 'MIN', 'S', 'MS', 'US', 'NS', 'PS', 'FS', 'LAST', 'AUTO'], outer_class=root_module['ns3::Time'], import_from_module='ns.core')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time )', 'ns3::Time::TracedCallback')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time )*', 'ns3::Time::TracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( ns3::Time )&', 'ns3::Time::TracedCallback&')
    ## nstime.h (module 'core'): ns3::TimeWithUnit [class]
    module.add_class('TimeWithUnit', import_from_module='ns.core')
    ## traced-value.h (module 'core'): ns3::TracedValue<bool> [class]
    module.add_class('TracedValue', import_from_module='ns.core', template_parameters=['bool'])
    ## traced-value.h (module 'core'): ns3::TracedValue<double> [class]
    module.add_class('TracedValue', import_from_module='ns.core', template_parameters=['double'])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned char> [class]
    module.add_class('TracedValue', import_from_module='ns.core', template_parameters=['unsigned char'])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned int> [class]
    module.add_class('TracedValue', import_from_module='ns.core', template_parameters=['unsigned int'])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned short> [class]
    module.add_class('TracedValue', import_from_module='ns.core', template_parameters=['unsigned short'])
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
    ## gnuplot.h (module 'stats'): ns3::Gnuplot2dDataset [class]
    module.add_class('Gnuplot2dDataset', parent=root_module['ns3::GnuplotDataset'])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot2dDataset::Style [enumeration]
    module.add_enum('Style', ['LINES', 'POINTS', 'LINES_POINTS', 'DOTS', 'IMPULSES', 'STEPS', 'FSTEPS', 'HISTEPS'], outer_class=root_module['ns3::Gnuplot2dDataset'])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot2dDataset::ErrorBars [enumeration]
    module.add_enum('ErrorBars', ['NONE', 'X', 'Y', 'XY'], outer_class=root_module['ns3::Gnuplot2dDataset'])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot2dFunction [class]
    module.add_class('Gnuplot2dFunction', parent=root_module['ns3::GnuplotDataset'])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot3dDataset [class]
    module.add_class('Gnuplot3dDataset', parent=root_module['ns3::GnuplotDataset'])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot3dFunction [class]
    module.add_class('Gnuplot3dFunction', parent=root_module['ns3::GnuplotDataset'])
    ## object.h (module 'core'): ns3::Object [class]
    module.add_class('Object', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter >'])
    ## object.h (module 'core'): ns3::Object::AggregateIterator [class]
    module.add_class('AggregateIterator', import_from_module='ns.core', outer_class=root_module['ns3::Object'])
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
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>> [class]
    module.add_class('SimpleRefCount', automatic_type_narrowing=True, import_from_module='ns.core', memory_policy=cppclass.ReferenceCountingMethodsPolicy(incref_method='Ref', decref_method='Unref', peekref_method='GetReferenceCount'), parent=root_module['ns3::empty'], template_parameters=['ns3::TraceSourceAccessor', 'ns3::empty', 'ns3::DefaultDeleter<ns3::TraceSourceAccessor>'])
    ## trace-source-accessor.h (module 'core'): ns3::TraceSourceAccessor [class]
    module.add_class('TraceSourceAccessor', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor> >'])
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
    ## data-calculator.h (module 'stats'): ns3::DataCalculator [class]
    module.add_class('DataCalculator', parent=root_module['ns3::Object'])
    ## data-collection-object.h (module 'stats'): ns3::DataCollectionObject [class]
    module.add_class('DataCollectionObject', parent=root_module['ns3::Object'])
    ## data-collector.h (module 'stats'): ns3::DataCollector [class]
    module.add_class('DataCollector', parent=root_module['ns3::Object'])
    ## data-output-interface.h (module 'stats'): ns3::DataOutputInterface [class]
    module.add_class('DataOutputInterface', parent=root_module['ns3::Object'])
    ## double.h (module 'core'): ns3::DoubleValue [class]
    module.add_class('DoubleValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
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
    ## event-impl.h (module 'core'): ns3::EventImpl [class]
    module.add_class('EventImpl', import_from_module='ns.core', parent=root_module['ns3::SimpleRefCount< ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl> >'])
    ## file-aggregator.h (module 'stats'): ns3::FileAggregator [class]
    module.add_class('FileAggregator', parent=root_module['ns3::DataCollectionObject'])
    ## file-aggregator.h (module 'stats'): ns3::FileAggregator::FileType [enumeration]
    module.add_enum('FileType', ['FORMATTED', 'SPACE_SEPARATED', 'COMMA_SEPARATED', 'TAB_SEPARATED'], outer_class=root_module['ns3::FileAggregator'])
    ## gnuplot-aggregator.h (module 'stats'): ns3::GnuplotAggregator [class]
    module.add_class('GnuplotAggregator', parent=root_module['ns3::DataCollectionObject'])
    ## gnuplot-aggregator.h (module 'stats'): ns3::GnuplotAggregator::KeyLocation [enumeration]
    module.add_enum('KeyLocation', ['NO_KEY', 'KEY_INSIDE', 'KEY_ABOVE', 'KEY_BELOW'], outer_class=root_module['ns3::GnuplotAggregator'])
    ## integer.h (module 'core'): ns3::IntegerValue [class]
    module.add_class('IntegerValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## basic-data-calculators.h (module 'stats'): ns3::MinMaxAvgTotalCalculator<double> [class]
    module.add_class('MinMaxAvgTotalCalculator', parent=[root_module['ns3::DataCalculator'], root_module['ns3::StatisticalSummary']], template_parameters=['double'])
    ## object-factory.h (module 'core'): ns3::ObjectFactoryChecker [class]
    module.add_class('ObjectFactoryChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## object-factory.h (module 'core'): ns3::ObjectFactoryValue [class]
    module.add_class('ObjectFactoryValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## omnet-data-output.h (module 'stats'): ns3::OmnetDataOutput [class]
    module.add_class('OmnetDataOutput', parent=root_module['ns3::DataOutputInterface'])
    ## probe.h (module 'stats'): ns3::Probe [class]
    module.add_class('Probe', parent=root_module['ns3::DataCollectionObject'])
    ## time-data-calculators.h (module 'stats'): ns3::TimeMinMaxAvgTotalCalculator [class]
    module.add_class('TimeMinMaxAvgTotalCalculator', parent=root_module['ns3::DataCalculator'])
    ## time-probe.h (module 'stats'): ns3::TimeProbe [class]
    module.add_class('TimeProbe', parent=root_module['ns3::Probe'])
    ## time-series-adaptor.h (module 'stats'): ns3::TimeSeriesAdaptor [class]
    module.add_class('TimeSeriesAdaptor', parent=root_module['ns3::DataCollectionObject'])
    typehandlers.add_type_alias('void ( * ) ( double const, double const )', 'ns3::TimeSeriesAdaptor::OutputTracedCallback')
    typehandlers.add_type_alias('void ( * ) ( double const, double const )*', 'ns3::TimeSeriesAdaptor::OutputTracedCallback*')
    typehandlers.add_type_alias('void ( * ) ( double const, double const )&', 'ns3::TimeSeriesAdaptor::OutputTracedCallback&')
    ## nstime.h (module 'core'): ns3::TimeValue [class]
    module.add_class('TimeValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## type-id.h (module 'core'): ns3::TypeIdChecker [class]
    module.add_class('TypeIdChecker', import_from_module='ns.core', parent=root_module['ns3::AttributeChecker'])
    ## type-id.h (module 'core'): ns3::TypeIdValue [class]
    module.add_class('TypeIdValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## uinteger-16-probe.h (module 'stats'): ns3::Uinteger16Probe [class]
    module.add_class('Uinteger16Probe', parent=root_module['ns3::Probe'])
    ## uinteger-32-probe.h (module 'stats'): ns3::Uinteger32Probe [class]
    module.add_class('Uinteger32Probe', parent=root_module['ns3::Probe'])
    ## uinteger-8-probe.h (module 'stats'): ns3::Uinteger8Probe [class]
    module.add_class('Uinteger8Probe', parent=root_module['ns3::Probe'])
    ## uinteger.h (module 'core'): ns3::UintegerValue [class]
    module.add_class('UintegerValue', import_from_module='ns.core', parent=root_module['ns3::AttributeValue'])
    ## boolean-probe.h (module 'stats'): ns3::BooleanProbe [class]
    module.add_class('BooleanProbe', parent=root_module['ns3::Probe'])
    ## callback.h (module 'core'): ns3::CallbackImpl<ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['ns3::ObjectBase *', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, bool, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'bool', 'bool', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, double, double, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'double', 'double', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, unsigned char, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'unsigned char', 'unsigned char', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'unsigned int', 'unsigned int', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned short, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> [class]
    module.add_class('CallbackImpl', import_from_module='ns.core', parent=root_module['ns3::CallbackImplBase'], template_parameters=['void', 'unsigned short', 'unsigned short', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty', 'ns3::empty'])
    ## double-probe.h (module 'stats'): ns3::DoubleProbe [class]
    module.add_class('DoubleProbe', parent=root_module['ns3::Probe'])
    module.add_container('std::map< std::string, ns3::LogComponent * >', ('std::basic_string< char, std::char_traits< char >, std::allocator< char > >', 'ns3::LogComponent *'), container_type='map')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::DataCalculator > >', 'ns3::DataCalculatorList')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::DataCalculator > >*', 'ns3::DataCalculatorList*')
    typehandlers.add_type_alias('std::list< ns3::Ptr< ns3::DataCalculator > >&', 'ns3::DataCalculatorList&')
    typehandlers.add_type_alias('std::list< std::pair< std::string, std::string > >', 'ns3::MetadataList')
    typehandlers.add_type_alias('std::list< std::pair< std::string, std::string > >*', 'ns3::MetadataList*')
    typehandlers.add_type_alias('std::list< std::pair< std::string, std::string > >&', 'ns3::MetadataList&')
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

def register_types_ns3_internal(module):
    root_module = module.get_root()
    

def register_methods(root_module):
    register_Ns3AttributeConstructionList_methods(root_module, root_module['ns3::AttributeConstructionList'])
    register_Ns3AttributeConstructionListItem_methods(root_module, root_module['ns3::AttributeConstructionList::Item'])
    register_Ns3CallbackBase_methods(root_module, root_module['ns3::CallbackBase'])
    register_Ns3DataOutputCallback_methods(root_module, root_module['ns3::DataOutputCallback'])
    register_Ns3DefaultDeleter__Ns3AttributeAccessor_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeAccessor >'])
    register_Ns3DefaultDeleter__Ns3AttributeChecker_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeChecker >'])
    register_Ns3DefaultDeleter__Ns3AttributeValue_methods(root_module, root_module['ns3::DefaultDeleter< ns3::AttributeValue >'])
    register_Ns3DefaultDeleter__Ns3CallbackImplBase_methods(root_module, root_module['ns3::DefaultDeleter< ns3::CallbackImplBase >'])
    register_Ns3DefaultDeleter__Ns3EventImpl_methods(root_module, root_module['ns3::DefaultDeleter< ns3::EventImpl >'])
    register_Ns3DefaultDeleter__Ns3HashImplementation_methods(root_module, root_module['ns3::DefaultDeleter< ns3::Hash::Implementation >'])
    register_Ns3DefaultDeleter__Ns3TraceSourceAccessor_methods(root_module, root_module['ns3::DefaultDeleter< ns3::TraceSourceAccessor >'])
    register_Ns3EventId_methods(root_module, root_module['ns3::EventId'])
    register_Ns3FileHelper_methods(root_module, root_module['ns3::FileHelper'])
    register_Ns3Gnuplot_methods(root_module, root_module['ns3::Gnuplot'])
    register_Ns3GnuplotCollection_methods(root_module, root_module['ns3::GnuplotCollection'])
    register_Ns3GnuplotDataset_methods(root_module, root_module['ns3::GnuplotDataset'])
    register_Ns3GnuplotHelper_methods(root_module, root_module['ns3::GnuplotHelper'])
    register_Ns3Hasher_methods(root_module, root_module['ns3::Hasher'])
    register_Ns3Histogram_methods(root_module, root_module['ns3::Histogram'])
    register_Ns3LogComponent_methods(root_module, root_module['ns3::LogComponent'])
    register_Ns3ObjectBase_methods(root_module, root_module['ns3::ObjectBase'])
    register_Ns3ObjectDeleter_methods(root_module, root_module['ns3::ObjectDeleter'])
    register_Ns3ObjectFactory_methods(root_module, root_module['ns3::ObjectFactory'])
    register_Ns3ParameterLogger_methods(root_module, root_module['ns3::ParameterLogger'])
    register_Ns3SimpleRefCount__Ns3Object_Ns3ObjectBase_Ns3ObjectDeleter_methods(root_module, root_module['ns3::SimpleRefCount< ns3::Object, ns3::ObjectBase, ns3::ObjectDeleter >'])
    register_Ns3Simulator_methods(root_module, root_module['ns3::Simulator'])
    register_Ns3StatisticalSummary_methods(root_module, root_module['ns3::StatisticalSummary'])
    register_Ns3Time_methods(root_module, root_module['ns3::Time'])
    register_Ns3TimeWithUnit_methods(root_module, root_module['ns3::TimeWithUnit'])
    register_Ns3TracedValue__Bool_methods(root_module, root_module['ns3::TracedValue< bool >'])
    register_Ns3TracedValue__Double_methods(root_module, root_module['ns3::TracedValue< double >'])
    register_Ns3TracedValue__Unsigned_char_methods(root_module, root_module['ns3::TracedValue< unsigned char >'])
    register_Ns3TracedValue__Unsigned_int_methods(root_module, root_module['ns3::TracedValue< unsigned int >'])
    register_Ns3TracedValue__Unsigned_short_methods(root_module, root_module['ns3::TracedValue< unsigned short >'])
    register_Ns3TypeId_methods(root_module, root_module['ns3::TypeId'])
    register_Ns3TypeIdAttributeInformation_methods(root_module, root_module['ns3::TypeId::AttributeInformation'])
    register_Ns3TypeIdTraceSourceInformation_methods(root_module, root_module['ns3::TypeId::TraceSourceInformation'])
    register_Ns3Empty_methods(root_module, root_module['ns3::empty'])
    register_Ns3Int64x64_t_methods(root_module, root_module['ns3::int64x64_t'])
    register_Ns3Gnuplot2dDataset_methods(root_module, root_module['ns3::Gnuplot2dDataset'])
    register_Ns3Gnuplot2dFunction_methods(root_module, root_module['ns3::Gnuplot2dFunction'])
    register_Ns3Gnuplot3dDataset_methods(root_module, root_module['ns3::Gnuplot3dDataset'])
    register_Ns3Gnuplot3dFunction_methods(root_module, root_module['ns3::Gnuplot3dFunction'])
    register_Ns3Object_methods(root_module, root_module['ns3::Object'])
    register_Ns3ObjectAggregateIterator_methods(root_module, root_module['ns3::Object::AggregateIterator'])
    register_Ns3SimpleRefCount__Ns3AttributeAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeAccessor__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeAccessor, ns3::empty, ns3::DefaultDeleter<ns3::AttributeAccessor> >'])
    register_Ns3SimpleRefCount__Ns3AttributeChecker_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeChecker__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeChecker, ns3::empty, ns3::DefaultDeleter<ns3::AttributeChecker> >'])
    register_Ns3SimpleRefCount__Ns3AttributeValue_Ns3Empty_Ns3DefaultDeleter__lt__ns3AttributeValue__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::AttributeValue, ns3::empty, ns3::DefaultDeleter<ns3::AttributeValue> >'])
    register_Ns3SimpleRefCount__Ns3CallbackImplBase_Ns3Empty_Ns3DefaultDeleter__lt__ns3CallbackImplBase__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::CallbackImplBase, ns3::empty, ns3::DefaultDeleter<ns3::CallbackImplBase> >'])
    register_Ns3SimpleRefCount__Ns3EventImpl_Ns3Empty_Ns3DefaultDeleter__lt__ns3EventImpl__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::EventImpl, ns3::empty, ns3::DefaultDeleter<ns3::EventImpl> >'])
    register_Ns3SimpleRefCount__Ns3HashImplementation_Ns3Empty_Ns3DefaultDeleter__lt__ns3HashImplementation__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::Hash::Implementation, ns3::empty, ns3::DefaultDeleter<ns3::Hash::Implementation> >'])
    register_Ns3SimpleRefCount__Ns3TraceSourceAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3TraceSourceAccessor__gt___methods(root_module, root_module['ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor> >'])
    register_Ns3TraceSourceAccessor_methods(root_module, root_module['ns3::TraceSourceAccessor'])
    register_Ns3AttributeAccessor_methods(root_module, root_module['ns3::AttributeAccessor'])
    register_Ns3AttributeChecker_methods(root_module, root_module['ns3::AttributeChecker'])
    register_Ns3AttributeValue_methods(root_module, root_module['ns3::AttributeValue'])
    register_Ns3BooleanChecker_methods(root_module, root_module['ns3::BooleanChecker'])
    register_Ns3BooleanValue_methods(root_module, root_module['ns3::BooleanValue'])
    register_Ns3CallbackChecker_methods(root_module, root_module['ns3::CallbackChecker'])
    register_Ns3CallbackImplBase_methods(root_module, root_module['ns3::CallbackImplBase'])
    register_Ns3CallbackValue_methods(root_module, root_module['ns3::CallbackValue'])
    register_Ns3DataCalculator_methods(root_module, root_module['ns3::DataCalculator'])
    register_Ns3DataCollectionObject_methods(root_module, root_module['ns3::DataCollectionObject'])
    register_Ns3DataCollector_methods(root_module, root_module['ns3::DataCollector'])
    register_Ns3DataOutputInterface_methods(root_module, root_module['ns3::DataOutputInterface'])
    register_Ns3DoubleValue_methods(root_module, root_module['ns3::DoubleValue'])
    register_Ns3EmptyAttributeAccessor_methods(root_module, root_module['ns3::EmptyAttributeAccessor'])
    register_Ns3EmptyAttributeChecker_methods(root_module, root_module['ns3::EmptyAttributeChecker'])
    register_Ns3EmptyAttributeValue_methods(root_module, root_module['ns3::EmptyAttributeValue'])
    register_Ns3EnumChecker_methods(root_module, root_module['ns3::EnumChecker'])
    register_Ns3EnumValue_methods(root_module, root_module['ns3::EnumValue'])
    register_Ns3EventImpl_methods(root_module, root_module['ns3::EventImpl'])
    register_Ns3FileAggregator_methods(root_module, root_module['ns3::FileAggregator'])
    register_Ns3GnuplotAggregator_methods(root_module, root_module['ns3::GnuplotAggregator'])
    register_Ns3IntegerValue_methods(root_module, root_module['ns3::IntegerValue'])
    register_Ns3MinMaxAvgTotalCalculator__Double_methods(root_module, root_module['ns3::MinMaxAvgTotalCalculator< double >'])
    register_Ns3ObjectFactoryChecker_methods(root_module, root_module['ns3::ObjectFactoryChecker'])
    register_Ns3ObjectFactoryValue_methods(root_module, root_module['ns3::ObjectFactoryValue'])
    register_Ns3OmnetDataOutput_methods(root_module, root_module['ns3::OmnetDataOutput'])
    register_Ns3Probe_methods(root_module, root_module['ns3::Probe'])
    register_Ns3TimeMinMaxAvgTotalCalculator_methods(root_module, root_module['ns3::TimeMinMaxAvgTotalCalculator'])
    register_Ns3TimeProbe_methods(root_module, root_module['ns3::TimeProbe'])
    register_Ns3TimeSeriesAdaptor_methods(root_module, root_module['ns3::TimeSeriesAdaptor'])
    register_Ns3TimeValue_methods(root_module, root_module['ns3::TimeValue'])
    register_Ns3TypeIdChecker_methods(root_module, root_module['ns3::TypeIdChecker'])
    register_Ns3TypeIdValue_methods(root_module, root_module['ns3::TypeIdValue'])
    register_Ns3Uinteger16Probe_methods(root_module, root_module['ns3::Uinteger16Probe'])
    register_Ns3Uinteger32Probe_methods(root_module, root_module['ns3::Uinteger32Probe'])
    register_Ns3Uinteger8Probe_methods(root_module, root_module['ns3::Uinteger8Probe'])
    register_Ns3UintegerValue_methods(root_module, root_module['ns3::UintegerValue'])
    register_Ns3BooleanProbe_methods(root_module, root_module['ns3::BooleanProbe'])
    register_Ns3CallbackImpl__Ns3ObjectBase___star___Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< ns3::ObjectBase *, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Bool_Bool_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, bool, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Double_Double_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, double, double, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Unsigned_char_Unsigned_char_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, unsigned char, unsigned char, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Unsigned_int_Unsigned_int_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, unsigned int, unsigned int, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3CallbackImpl__Void_Unsigned_short_Unsigned_short_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, root_module['ns3::CallbackImpl< void, unsigned short, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty >'])
    register_Ns3DoubleProbe_methods(root_module, root_module['ns3::DoubleProbe'])
    register_Ns3HashImplementation_methods(root_module, root_module['ns3::Hash::Implementation'])
    register_Ns3HashFunctionFnv1a_methods(root_module, root_module['ns3::Hash::Function::Fnv1a'])
    register_Ns3HashFunctionHash32_methods(root_module, root_module['ns3::Hash::Function::Hash32'])
    register_Ns3HashFunctionHash64_methods(root_module, root_module['ns3::Hash::Function::Hash64'])
    register_Ns3HashFunctionMurmur3_methods(root_module, root_module['ns3::Hash::Function::Murmur3'])
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

def register_Ns3FileHelper_methods(root_module, cls):
    ## file-helper.h (module 'stats'): ns3::FileHelper::FileHelper(ns3::FileHelper const & arg0) [constructor]
    cls.add_constructor([param('ns3::FileHelper const &', 'arg0')])
    ## file-helper.h (module 'stats'): ns3::FileHelper::FileHelper() [constructor]
    cls.add_constructor([])
    ## file-helper.h (module 'stats'): ns3::FileHelper::FileHelper(std::string const & outputFileNameWithoutExtension, ns3::FileAggregator::FileType fileType=::ns3::FileAggregator::FileType::SPACE_SEPARATED) [constructor]
    cls.add_constructor([param('std::string const &', 'outputFileNameWithoutExtension'), param('ns3::FileAggregator::FileType', 'fileType', default_value='::ns3::FileAggregator::FileType::SPACE_SEPARATED')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::AddAggregator(std::string const & aggregatorName, std::string const & outputFileName, bool onlyOneAggregator) [member function]
    cls.add_method('AddAggregator', 
                   'void', 
                   [param('std::string const &', 'aggregatorName'), param('std::string const &', 'outputFileName'), param('bool', 'onlyOneAggregator')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::AddTimeSeriesAdaptor(std::string const & adaptorName) [member function]
    cls.add_method('AddTimeSeriesAdaptor', 
                   'void', 
                   [param('std::string const &', 'adaptorName')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::ConfigureFile(std::string const & outputFileNameWithoutExtension, ns3::FileAggregator::FileType fileType=::ns3::FileAggregator::FileType::SPACE_SEPARATED) [member function]
    cls.add_method('ConfigureFile', 
                   'void', 
                   [param('std::string const &', 'outputFileNameWithoutExtension'), param('ns3::FileAggregator::FileType', 'fileType', default_value='::ns3::FileAggregator::FileType::SPACE_SEPARATED')])
    ## file-helper.h (module 'stats'): ns3::Ptr<ns3::FileAggregator> ns3::FileHelper::GetAggregatorMultiple(std::string const & aggregatorName, std::string const & outputFileName) [member function]
    cls.add_method('GetAggregatorMultiple', 
                   'ns3::Ptr< ns3::FileAggregator >', 
                   [param('std::string const &', 'aggregatorName'), param('std::string const &', 'outputFileName')])
    ## file-helper.h (module 'stats'): ns3::Ptr<ns3::FileAggregator> ns3::FileHelper::GetAggregatorSingle() [member function]
    cls.add_method('GetAggregatorSingle', 
                   'ns3::Ptr< ns3::FileAggregator >', 
                   [])
    ## file-helper.h (module 'stats'): ns3::Ptr<ns3::Probe> ns3::FileHelper::GetProbe(std::string probeName) const [member function]
    cls.add_method('GetProbe', 
                   'ns3::Ptr< ns3::Probe >', 
                   [param('std::string', 'probeName')], 
                   is_const=True)
    ## file-helper.h (module 'stats'): void ns3::FileHelper::Set10dFormat(std::string const & format) [member function]
    cls.add_method('Set10dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::Set1dFormat(std::string const & format) [member function]
    cls.add_method('Set1dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::Set2dFormat(std::string const & format) [member function]
    cls.add_method('Set2dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::Set3dFormat(std::string const & format) [member function]
    cls.add_method('Set3dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::Set4dFormat(std::string const & format) [member function]
    cls.add_method('Set4dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::Set5dFormat(std::string const & format) [member function]
    cls.add_method('Set5dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::Set6dFormat(std::string const & format) [member function]
    cls.add_method('Set6dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::Set7dFormat(std::string const & format) [member function]
    cls.add_method('Set7dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::Set8dFormat(std::string const & format) [member function]
    cls.add_method('Set8dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::Set9dFormat(std::string const & format) [member function]
    cls.add_method('Set9dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::SetHeading(std::string const & heading) [member function]
    cls.add_method('SetHeading', 
                   'void', 
                   [param('std::string const &', 'heading')])
    ## file-helper.h (module 'stats'): void ns3::FileHelper::WriteProbe(std::string const & typeId, std::string const & path, std::string const & probeTraceSource) [member function]
    cls.add_method('WriteProbe', 
                   'void', 
                   [param('std::string const &', 'typeId'), param('std::string const &', 'path'), param('std::string const &', 'probeTraceSource')])
    return

def register_Ns3Gnuplot_methods(root_module, cls):
    ## gnuplot.h (module 'stats'): ns3::Gnuplot::Gnuplot(ns3::Gnuplot const & arg0) [constructor]
    cls.add_constructor([param('ns3::Gnuplot const &', 'arg0')])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot::Gnuplot(std::string const & outputFilename="", std::string const & title="") [constructor]
    cls.add_constructor([param('std::string const &', 'outputFilename', default_value='""'), param('std::string const &', 'title', default_value='""')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot::AddDataset(ns3::GnuplotDataset const & dataset) [member function]
    cls.add_method('AddDataset', 
                   'void', 
                   [param('ns3::GnuplotDataset const &', 'dataset')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot::AppendExtra(std::string const & extra) [member function]
    cls.add_method('AppendExtra', 
                   'void', 
                   [param('std::string const &', 'extra')])
    ## gnuplot.h (module 'stats'): static std::string ns3::Gnuplot::DetectTerminal(std::string const & filename) [member function]
    cls.add_method('DetectTerminal', 
                   'std::string', 
                   [param('std::string const &', 'filename')], 
                   is_static=True)
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot::GenerateOutput(std::ostream & os) [member function]
    cls.add_method('GenerateOutput', 
                   'void', 
                   [param('std::ostream &', 'os')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot::GenerateOutput(std::ostream & osControl, std::ostream & osData, std::string dataFileName) [member function]
    cls.add_method('GenerateOutput', 
                   'void', 
                   [param('std::ostream &', 'osControl'), param('std::ostream &', 'osData'), param('std::string', 'dataFileName')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot::SetDataFileDatasetIndex(unsigned int index) [member function]
    cls.add_method('SetDataFileDatasetIndex', 
                   'void', 
                   [param('unsigned int', 'index')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot::SetExtra(std::string const & extra) [member function]
    cls.add_method('SetExtra', 
                   'void', 
                   [param('std::string const &', 'extra')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot::SetLegend(std::string const & xLegend, std::string const & yLegend) [member function]
    cls.add_method('SetLegend', 
                   'void', 
                   [param('std::string const &', 'xLegend'), param('std::string const &', 'yLegend')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot::SetOutputFilename(std::string const & outputFilename) [member function]
    cls.add_method('SetOutputFilename', 
                   'void', 
                   [param('std::string const &', 'outputFilename')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot::SetTerminal(std::string const & terminal) [member function]
    cls.add_method('SetTerminal', 
                   'void', 
                   [param('std::string const &', 'terminal')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot::SetTitle(std::string const & title) [member function]
    cls.add_method('SetTitle', 
                   'void', 
                   [param('std::string const &', 'title')])
    return

def register_Ns3GnuplotCollection_methods(root_module, cls):
    ## gnuplot.h (module 'stats'): ns3::GnuplotCollection::GnuplotCollection(ns3::GnuplotCollection const & arg0) [constructor]
    cls.add_constructor([param('ns3::GnuplotCollection const &', 'arg0')])
    ## gnuplot.h (module 'stats'): ns3::GnuplotCollection::GnuplotCollection(std::string const & outputFilename) [constructor]
    cls.add_constructor([param('std::string const &', 'outputFilename')])
    ## gnuplot.h (module 'stats'): void ns3::GnuplotCollection::AddPlot(ns3::Gnuplot const & plot) [member function]
    cls.add_method('AddPlot', 
                   'void', 
                   [param('ns3::Gnuplot const &', 'plot')])
    ## gnuplot.h (module 'stats'): void ns3::GnuplotCollection::GenerateOutput(std::ostream & os) [member function]
    cls.add_method('GenerateOutput', 
                   'void', 
                   [param('std::ostream &', 'os')])
    ## gnuplot.h (module 'stats'): void ns3::GnuplotCollection::GenerateOutput(std::ostream & osControl, std::ostream & osData, std::string dataFileName) [member function]
    cls.add_method('GenerateOutput', 
                   'void', 
                   [param('std::ostream &', 'osControl'), param('std::ostream &', 'osData'), param('std::string', 'dataFileName')])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot & ns3::GnuplotCollection::GetPlot(unsigned int id) [member function]
    cls.add_method('GetPlot', 
                   'ns3::Gnuplot &', 
                   [param('unsigned int', 'id')])
    ## gnuplot.h (module 'stats'): void ns3::GnuplotCollection::SetTerminal(std::string const & terminal) [member function]
    cls.add_method('SetTerminal', 
                   'void', 
                   [param('std::string const &', 'terminal')])
    return

def register_Ns3GnuplotDataset_methods(root_module, cls):
    ## gnuplot.h (module 'stats'): ns3::GnuplotDataset::GnuplotDataset(ns3::GnuplotDataset const & original) [constructor]
    cls.add_constructor([param('ns3::GnuplotDataset const &', 'original')])
    ## gnuplot.h (module 'stats'): static void ns3::GnuplotDataset::SetDefaultExtra(std::string const & extra) [member function]
    cls.add_method('SetDefaultExtra', 
                   'void', 
                   [param('std::string const &', 'extra')], 
                   is_static=True)
    ## gnuplot.h (module 'stats'): void ns3::GnuplotDataset::SetExtra(std::string const & extra) [member function]
    cls.add_method('SetExtra', 
                   'void', 
                   [param('std::string const &', 'extra')])
    ## gnuplot.h (module 'stats'): void ns3::GnuplotDataset::SetTitle(std::string const & title) [member function]
    cls.add_method('SetTitle', 
                   'void', 
                   [param('std::string const &', 'title')])
    ## gnuplot.h (module 'stats'): ns3::GnuplotDataset::GnuplotDataset(ns3::GnuplotDataset::Data * data) [constructor]
    cls.add_constructor([param('ns3::GnuplotDataset::Data *', 'data')], 
                        visibility='protected')
    return

def register_Ns3GnuplotHelper_methods(root_module, cls):
    ## gnuplot-helper.h (module 'stats'): ns3::GnuplotHelper::GnuplotHelper(ns3::GnuplotHelper const & arg0) [constructor]
    cls.add_constructor([param('ns3::GnuplotHelper const &', 'arg0')])
    ## gnuplot-helper.h (module 'stats'): ns3::GnuplotHelper::GnuplotHelper() [constructor]
    cls.add_constructor([])
    ## gnuplot-helper.h (module 'stats'): ns3::GnuplotHelper::GnuplotHelper(std::string const & outputFileNameWithoutExtension, std::string const & title, std::string const & xLegend, std::string const & yLegend, std::string const & terminalType="png") [constructor]
    cls.add_constructor([param('std::string const &', 'outputFileNameWithoutExtension'), param('std::string const &', 'title'), param('std::string const &', 'xLegend'), param('std::string const &', 'yLegend'), param('std::string const &', 'terminalType', default_value='"png"')])
    ## gnuplot-helper.h (module 'stats'): void ns3::GnuplotHelper::AddTimeSeriesAdaptor(std::string const & adaptorName) [member function]
    cls.add_method('AddTimeSeriesAdaptor', 
                   'void', 
                   [param('std::string const &', 'adaptorName')])
    ## gnuplot-helper.h (module 'stats'): void ns3::GnuplotHelper::ConfigurePlot(std::string const & outputFileNameWithoutExtension, std::string const & title, std::string const & xLegend, std::string const & yLegend, std::string const & terminalType="png") [member function]
    cls.add_method('ConfigurePlot', 
                   'void', 
                   [param('std::string const &', 'outputFileNameWithoutExtension'), param('std::string const &', 'title'), param('std::string const &', 'xLegend'), param('std::string const &', 'yLegend'), param('std::string const &', 'terminalType', default_value='"png"')])
    ## gnuplot-helper.h (module 'stats'): ns3::Ptr<ns3::GnuplotAggregator> ns3::GnuplotHelper::GetAggregator() [member function]
    cls.add_method('GetAggregator', 
                   'ns3::Ptr< ns3::GnuplotAggregator >', 
                   [])
    ## gnuplot-helper.h (module 'stats'): ns3::Ptr<ns3::Probe> ns3::GnuplotHelper::GetProbe(std::string probeName) const [member function]
    cls.add_method('GetProbe', 
                   'ns3::Ptr< ns3::Probe >', 
                   [param('std::string', 'probeName')], 
                   is_const=True)
    ## gnuplot-helper.h (module 'stats'): void ns3::GnuplotHelper::PlotProbe(std::string const & typeId, std::string const & path, std::string const & probeTraceSource, std::string const & title, ns3::GnuplotAggregator::KeyLocation keyLocation=::ns3::GnuplotAggregator::KeyLocation::KEY_INSIDE) [member function]
    cls.add_method('PlotProbe', 
                   'void', 
                   [param('std::string const &', 'typeId'), param('std::string const &', 'path'), param('std::string const &', 'probeTraceSource'), param('std::string const &', 'title'), param('ns3::GnuplotAggregator::KeyLocation', 'keyLocation', default_value='::ns3::GnuplotAggregator::KeyLocation::KEY_INSIDE')])
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

def register_Ns3Histogram_methods(root_module, cls):
    ## histogram.h (module 'stats'): ns3::Histogram::Histogram(ns3::Histogram const & arg0) [constructor]
    cls.add_constructor([param('ns3::Histogram const &', 'arg0')])
    ## histogram.h (module 'stats'): ns3::Histogram::Histogram(double binWidth) [constructor]
    cls.add_constructor([param('double', 'binWidth')])
    ## histogram.h (module 'stats'): ns3::Histogram::Histogram() [constructor]
    cls.add_constructor([])
    ## histogram.h (module 'stats'): void ns3::Histogram::AddValue(double value) [member function]
    cls.add_method('AddValue', 
                   'void', 
                   [param('double', 'value')])
    ## histogram.h (module 'stats'): uint32_t ns3::Histogram::GetBinCount(uint32_t index) [member function]
    cls.add_method('GetBinCount', 
                   'uint32_t', 
                   [param('uint32_t', 'index')])
    ## histogram.h (module 'stats'): double ns3::Histogram::GetBinEnd(uint32_t index) [member function]
    cls.add_method('GetBinEnd', 
                   'double', 
                   [param('uint32_t', 'index')])
    ## histogram.h (module 'stats'): double ns3::Histogram::GetBinStart(uint32_t index) [member function]
    cls.add_method('GetBinStart', 
                   'double', 
                   [param('uint32_t', 'index')])
    ## histogram.h (module 'stats'): double ns3::Histogram::GetBinWidth(uint32_t index) const [member function]
    cls.add_method('GetBinWidth', 
                   'double', 
                   [param('uint32_t', 'index')], 
                   is_const=True)
    ## histogram.h (module 'stats'): uint32_t ns3::Histogram::GetNBins() const [member function]
    cls.add_method('GetNBins', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## histogram.h (module 'stats'): void ns3::Histogram::SerializeToXmlStream(std::ostream & os, uint16_t indent, std::string elementName) const [member function]
    cls.add_method('SerializeToXmlStream', 
                   'void', 
                   [param('std::ostream &', 'os'), param('uint16_t', 'indent'), param('std::string', 'elementName')], 
                   is_const=True)
    ## histogram.h (module 'stats'): void ns3::Histogram::SetDefaultBinWidth(double binWidth) [member function]
    cls.add_method('SetDefaultBinWidth', 
                   'void', 
                   [param('double', 'binWidth')])
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

def register_Ns3TracedValue__Bool_methods(root_module, cls):
    ## traced-value.h (module 'core'): ns3::TracedValue<bool>::TracedValue() [constructor]
    cls.add_constructor([])
    ## traced-value.h (module 'core'): ns3::TracedValue<bool>::TracedValue(ns3::TracedValue<bool> const & o) [constructor]
    cls.add_constructor([param('ns3::TracedValue< bool > const &', 'o')])
    ## traced-value.h (module 'core'): ns3::TracedValue<bool>::TracedValue(bool const & v) [constructor]
    cls.add_constructor([param('bool const &', 'v')])
    ## traced-value.h (module 'core'): ns3::TracedValue<bool>::TracedValue(ns3::TracedValue<bool> const & other) [constructor]
    cls.add_constructor([param('ns3::TracedValue< bool > const &', 'other')])
    ## traced-value.h (module 'core'): ns3::TracedValue<bool>::TracedValue(ns3::TracedValue<bool> const & other) [constructor]
    cls.add_constructor([param('ns3::TracedValue< bool > const &', 'other')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<bool>::Connect(ns3::CallbackBase const & cb, std::string path) [member function]
    cls.add_method('Connect', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb'), param('std::string', 'path')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<bool>::ConnectWithoutContext(ns3::CallbackBase const & cb) [member function]
    cls.add_method('ConnectWithoutContext', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<bool>::Disconnect(ns3::CallbackBase const & cb, std::string path) [member function]
    cls.add_method('Disconnect', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb'), param('std::string', 'path')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<bool>::DisconnectWithoutContext(ns3::CallbackBase const & cb) [member function]
    cls.add_method('DisconnectWithoutContext', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb')])
    ## traced-value.h (module 'core'): bool ns3::TracedValue<bool>::Get() const [member function]
    cls.add_method('Get', 
                   'bool', 
                   [], 
                   is_const=True)
    ## traced-value.h (module 'core'): void ns3::TracedValue<bool>::Set(bool const & v) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('bool const &', 'v')])
    return

def register_Ns3TracedValue__Double_methods(root_module, cls):
    ## traced-value.h (module 'core'): ns3::TracedValue<double>::TracedValue() [constructor]
    cls.add_constructor([])
    ## traced-value.h (module 'core'): ns3::TracedValue<double>::TracedValue(ns3::TracedValue<double> const & o) [constructor]
    cls.add_constructor([param('ns3::TracedValue< double > const &', 'o')])
    ## traced-value.h (module 'core'): ns3::TracedValue<double>::TracedValue(double const & v) [constructor]
    cls.add_constructor([param('double const &', 'v')])
    ## traced-value.h (module 'core'): ns3::TracedValue<double>::TracedValue(ns3::TracedValue<double> const & other) [constructor]
    cls.add_constructor([param('ns3::TracedValue< double > const &', 'other')])
    ## traced-value.h (module 'core'): ns3::TracedValue<double>::TracedValue(ns3::TracedValue<double> const & other) [constructor]
    cls.add_constructor([param('ns3::TracedValue< double > const &', 'other')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<double>::Connect(ns3::CallbackBase const & cb, std::string path) [member function]
    cls.add_method('Connect', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb'), param('std::string', 'path')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<double>::ConnectWithoutContext(ns3::CallbackBase const & cb) [member function]
    cls.add_method('ConnectWithoutContext', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<double>::Disconnect(ns3::CallbackBase const & cb, std::string path) [member function]
    cls.add_method('Disconnect', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb'), param('std::string', 'path')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<double>::DisconnectWithoutContext(ns3::CallbackBase const & cb) [member function]
    cls.add_method('DisconnectWithoutContext', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb')])
    ## traced-value.h (module 'core'): double ns3::TracedValue<double>::Get() const [member function]
    cls.add_method('Get', 
                   'double', 
                   [], 
                   is_const=True)
    ## traced-value.h (module 'core'): void ns3::TracedValue<double>::Set(double const & v) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('double const &', 'v')])
    return

def register_Ns3TracedValue__Unsigned_char_methods(root_module, cls):
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned char>::TracedValue() [constructor]
    cls.add_constructor([])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned char>::TracedValue(ns3::TracedValue<unsigned char> const & o) [constructor]
    cls.add_constructor([param('ns3::TracedValue< unsigned char > const &', 'o')])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned char>::TracedValue(unsigned char const & v) [constructor]
    cls.add_constructor([param('unsigned char const &', 'v')])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned char>::TracedValue(ns3::TracedValue<unsigned char> const & other) [constructor]
    cls.add_constructor([param('ns3::TracedValue< unsigned char > const &', 'other')])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned char>::TracedValue(ns3::TracedValue<unsigned char> const & other) [constructor]
    cls.add_constructor([param('ns3::TracedValue< unsigned char > const &', 'other')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned char>::Connect(ns3::CallbackBase const & cb, std::string path) [member function]
    cls.add_method('Connect', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb'), param('std::string', 'path')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned char>::ConnectWithoutContext(ns3::CallbackBase const & cb) [member function]
    cls.add_method('ConnectWithoutContext', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned char>::Disconnect(ns3::CallbackBase const & cb, std::string path) [member function]
    cls.add_method('Disconnect', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb'), param('std::string', 'path')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned char>::DisconnectWithoutContext(ns3::CallbackBase const & cb) [member function]
    cls.add_method('DisconnectWithoutContext', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb')])
    ## traced-value.h (module 'core'): unsigned char ns3::TracedValue<unsigned char>::Get() const [member function]
    cls.add_method('Get', 
                   'unsigned char', 
                   [], 
                   is_const=True)
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned char>::Set(unsigned char const & v) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('unsigned char const &', 'v')])
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

def register_Ns3TracedValue__Unsigned_short_methods(root_module, cls):
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned short>::TracedValue() [constructor]
    cls.add_constructor([])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned short>::TracedValue(ns3::TracedValue<unsigned short> const & o) [constructor]
    cls.add_constructor([param('ns3::TracedValue< unsigned short > const &', 'o')])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned short>::TracedValue(short unsigned int const & v) [constructor]
    cls.add_constructor([param('short unsigned int const &', 'v')])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned short>::TracedValue(ns3::TracedValue<unsigned short> const & other) [constructor]
    cls.add_constructor([param('ns3::TracedValue< unsigned short > const &', 'other')])
    ## traced-value.h (module 'core'): ns3::TracedValue<unsigned short>::TracedValue(ns3::TracedValue<unsigned short> const & other) [constructor]
    cls.add_constructor([param('ns3::TracedValue< unsigned short > const &', 'other')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned short>::Connect(ns3::CallbackBase const & cb, std::string path) [member function]
    cls.add_method('Connect', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb'), param('std::string', 'path')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned short>::ConnectWithoutContext(ns3::CallbackBase const & cb) [member function]
    cls.add_method('ConnectWithoutContext', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned short>::Disconnect(ns3::CallbackBase const & cb, std::string path) [member function]
    cls.add_method('Disconnect', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb'), param('std::string', 'path')])
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned short>::DisconnectWithoutContext(ns3::CallbackBase const & cb) [member function]
    cls.add_method('DisconnectWithoutContext', 
                   'void', 
                   [param('ns3::CallbackBase const &', 'cb')])
    ## traced-value.h (module 'core'): short unsigned int ns3::TracedValue<unsigned short>::Get() const [member function]
    cls.add_method('Get', 
                   'short unsigned int', 
                   [], 
                   is_const=True)
    ## traced-value.h (module 'core'): void ns3::TracedValue<unsigned short>::Set(short unsigned int const & v) [member function]
    cls.add_method('Set', 
                   'void', 
                   [param('short unsigned int const &', 'v')])
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

def register_Ns3Gnuplot2dDataset_methods(root_module, cls):
    ## gnuplot.h (module 'stats'): ns3::Gnuplot2dDataset::Gnuplot2dDataset(ns3::Gnuplot2dDataset const & arg0) [constructor]
    cls.add_constructor([param('ns3::Gnuplot2dDataset const &', 'arg0')])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot2dDataset::Gnuplot2dDataset(std::string const & title="Untitled") [constructor]
    cls.add_constructor([param('std::string const &', 'title', default_value='"Untitled"')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot2dDataset::Add(double x, double y) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('double', 'x'), param('double', 'y')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot2dDataset::Add(double x, double y, double errorDelta) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('double', 'x'), param('double', 'y'), param('double', 'errorDelta')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot2dDataset::Add(double x, double y, double xErrorDelta, double yErrorDelta) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('double', 'x'), param('double', 'y'), param('double', 'xErrorDelta'), param('double', 'yErrorDelta')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot2dDataset::AddEmptyLine() [member function]
    cls.add_method('AddEmptyLine', 
                   'void', 
                   [])
    ## gnuplot.h (module 'stats'): static void ns3::Gnuplot2dDataset::SetDefaultErrorBars(ns3::Gnuplot2dDataset::ErrorBars errorBars) [member function]
    cls.add_method('SetDefaultErrorBars', 
                   'void', 
                   [param('ns3::Gnuplot2dDataset::ErrorBars', 'errorBars')], 
                   is_static=True)
    ## gnuplot.h (module 'stats'): static void ns3::Gnuplot2dDataset::SetDefaultStyle(ns3::Gnuplot2dDataset::Style style) [member function]
    cls.add_method('SetDefaultStyle', 
                   'void', 
                   [param('ns3::Gnuplot2dDataset::Style', 'style')], 
                   is_static=True)
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot2dDataset::SetErrorBars(ns3::Gnuplot2dDataset::ErrorBars errorBars) [member function]
    cls.add_method('SetErrorBars', 
                   'void', 
                   [param('ns3::Gnuplot2dDataset::ErrorBars', 'errorBars')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot2dDataset::SetStyle(ns3::Gnuplot2dDataset::Style style) [member function]
    cls.add_method('SetStyle', 
                   'void', 
                   [param('ns3::Gnuplot2dDataset::Style', 'style')])
    return

def register_Ns3Gnuplot2dFunction_methods(root_module, cls):
    ## gnuplot.h (module 'stats'): ns3::Gnuplot2dFunction::Gnuplot2dFunction(ns3::Gnuplot2dFunction const & arg0) [constructor]
    cls.add_constructor([param('ns3::Gnuplot2dFunction const &', 'arg0')])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot2dFunction::Gnuplot2dFunction(std::string const & title="Untitled", std::string const & function="") [constructor]
    cls.add_constructor([param('std::string const &', 'title', default_value='"Untitled"'), param('std::string const &', 'function', default_value='""')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot2dFunction::SetFunction(std::string const & function) [member function]
    cls.add_method('SetFunction', 
                   'void', 
                   [param('std::string const &', 'function')])
    return

def register_Ns3Gnuplot3dDataset_methods(root_module, cls):
    ## gnuplot.h (module 'stats'): ns3::Gnuplot3dDataset::Gnuplot3dDataset(ns3::Gnuplot3dDataset const & arg0) [constructor]
    cls.add_constructor([param('ns3::Gnuplot3dDataset const &', 'arg0')])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot3dDataset::Gnuplot3dDataset(std::string const & title="Untitled") [constructor]
    cls.add_constructor([param('std::string const &', 'title', default_value='"Untitled"')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot3dDataset::Add(double x, double y, double z) [member function]
    cls.add_method('Add', 
                   'void', 
                   [param('double', 'x'), param('double', 'y'), param('double', 'z')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot3dDataset::AddEmptyLine() [member function]
    cls.add_method('AddEmptyLine', 
                   'void', 
                   [])
    ## gnuplot.h (module 'stats'): static void ns3::Gnuplot3dDataset::SetDefaultStyle(std::string const & style) [member function]
    cls.add_method('SetDefaultStyle', 
                   'void', 
                   [param('std::string const &', 'style')], 
                   is_static=True)
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot3dDataset::SetStyle(std::string const & style) [member function]
    cls.add_method('SetStyle', 
                   'void', 
                   [param('std::string const &', 'style')])
    return

def register_Ns3Gnuplot3dFunction_methods(root_module, cls):
    ## gnuplot.h (module 'stats'): ns3::Gnuplot3dFunction::Gnuplot3dFunction(ns3::Gnuplot3dFunction const & arg0) [constructor]
    cls.add_constructor([param('ns3::Gnuplot3dFunction const &', 'arg0')])
    ## gnuplot.h (module 'stats'): ns3::Gnuplot3dFunction::Gnuplot3dFunction(std::string const & title="Untitled", std::string const & function="") [constructor]
    cls.add_constructor([param('std::string const &', 'title', default_value='"Untitled"'), param('std::string const &', 'function', default_value='""')])
    ## gnuplot.h (module 'stats'): void ns3::Gnuplot3dFunction::SetFunction(std::string const & function) [member function]
    cls.add_method('SetFunction', 
                   'void', 
                   [param('std::string const &', 'function')])
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

def register_Ns3SimpleRefCount__Ns3TraceSourceAccessor_Ns3Empty_Ns3DefaultDeleter__lt__ns3TraceSourceAccessor__gt___methods(root_module, cls):
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>>::SimpleRefCount() [constructor]
    cls.add_constructor([])
    ## simple-ref-count.h (module 'core'): ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>>::SimpleRefCount(ns3::SimpleRefCount<ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter<ns3::TraceSourceAccessor>> const & o) [constructor]
    cls.add_constructor([param('ns3::SimpleRefCount< ns3::TraceSourceAccessor, ns3::empty, ns3::DefaultDeleter< ns3::TraceSourceAccessor > > const &', 'o')])
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
                   is_static=True, template_parameters=['bool'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['double'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['unsigned short'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['unsigned int'], visibility='protected')
    ## callback.h (module 'core'): static std::string ns3::CallbackImplBase::GetCppTypeid() [member function]
    cls.add_method('GetCppTypeid', 
                   'std::string', 
                   [], 
                   is_static=True, template_parameters=['unsigned char'], visibility='protected')
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

def register_Ns3DataCollector_methods(root_module, cls):
    ## data-collector.h (module 'stats'): ns3::DataCollector::DataCollector(ns3::DataCollector const & arg0) [constructor]
    cls.add_constructor([param('ns3::DataCollector const &', 'arg0')])
    ## data-collector.h (module 'stats'): ns3::DataCollector::DataCollector() [constructor]
    cls.add_constructor([])
    ## data-collector.h (module 'stats'): void ns3::DataCollector::AddDataCalculator(ns3::Ptr<ns3::DataCalculator> datac) [member function]
    cls.add_method('AddDataCalculator', 
                   'void', 
                   [param('ns3::Ptr< ns3::DataCalculator >', 'datac')])
    ## data-collector.h (module 'stats'): void ns3::DataCollector::AddMetadata(std::string key, std::string value) [member function]
    cls.add_method('AddMetadata', 
                   'void', 
                   [param('std::string', 'key'), param('std::string', 'value')])
    ## data-collector.h (module 'stats'): void ns3::DataCollector::AddMetadata(std::string key, double value) [member function]
    cls.add_method('AddMetadata', 
                   'void', 
                   [param('std::string', 'key'), param('double', 'value')])
    ## data-collector.h (module 'stats'): void ns3::DataCollector::AddMetadata(std::string key, uint32_t value) [member function]
    cls.add_method('AddMetadata', 
                   'void', 
                   [param('std::string', 'key'), param('uint32_t', 'value')])
    ## data-collector.h (module 'stats'): std::list<ns3::Ptr<ns3::DataCalculator>, std::allocator<ns3::Ptr<ns3::DataCalculator>>>::iterator ns3::DataCollector::DataCalculatorBegin() [member function]
    cls.add_method('DataCalculatorBegin', 
                   'std::list< ns3::Ptr< ns3::DataCalculator > > iterator', 
                   [])
    ## data-collector.h (module 'stats'): std::list<ns3::Ptr<ns3::DataCalculator>, std::allocator<ns3::Ptr<ns3::DataCalculator>>>::iterator ns3::DataCollector::DataCalculatorEnd() [member function]
    cls.add_method('DataCalculatorEnd', 
                   'std::list< ns3::Ptr< ns3::DataCalculator > > iterator', 
                   [])
    ## data-collector.h (module 'stats'): void ns3::DataCollector::DescribeRun(std::string experiment, std::string strategy, std::string input, std::string runID, std::string description="") [member function]
    cls.add_method('DescribeRun', 
                   'void', 
                   [param('std::string', 'experiment'), param('std::string', 'strategy'), param('std::string', 'input'), param('std::string', 'runID'), param('std::string', 'description', default_value='""')])
    ## data-collector.h (module 'stats'): std::string ns3::DataCollector::GetDescription() const [member function]
    cls.add_method('GetDescription', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## data-collector.h (module 'stats'): std::string ns3::DataCollector::GetExperimentLabel() const [member function]
    cls.add_method('GetExperimentLabel', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## data-collector.h (module 'stats'): std::string ns3::DataCollector::GetInputLabel() const [member function]
    cls.add_method('GetInputLabel', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## data-collector.h (module 'stats'): std::string ns3::DataCollector::GetRunLabel() const [member function]
    cls.add_method('GetRunLabel', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## data-collector.h (module 'stats'): std::string ns3::DataCollector::GetStrategyLabel() const [member function]
    cls.add_method('GetStrategyLabel', 
                   'std::string', 
                   [], 
                   is_const=True)
    ## data-collector.h (module 'stats'): static ns3::TypeId ns3::DataCollector::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## data-collector.h (module 'stats'): std::list<std::pair<std::basic_string<char>, std::basic_string<char>>, std::allocator<std::pair<std::basic_string<char>, std::basic_string<char>>>>::iterator ns3::DataCollector::MetadataBegin() [member function]
    cls.add_method('MetadataBegin', 
                   'std::list< std::pair< std::string, std::string > > iterator', 
                   [])
    ## data-collector.h (module 'stats'): std::list<std::pair<std::basic_string<char>, std::basic_string<char>>, std::allocator<std::pair<std::basic_string<char>, std::basic_string<char>>>>::iterator ns3::DataCollector::MetadataEnd() [member function]
    cls.add_method('MetadataEnd', 
                   'std::list< std::pair< std::string, std::string > > iterator', 
                   [])
    ## data-collector.h (module 'stats'): void ns3::DataCollector::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
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

def register_Ns3FileAggregator_methods(root_module, cls):
    ## file-aggregator.h (module 'stats'): static ns3::TypeId ns3::FileAggregator::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## file-aggregator.h (module 'stats'): ns3::FileAggregator::FileAggregator(std::string const & outputFileName, ns3::FileAggregator::FileType fileType=::ns3::FileAggregator::FileType::SPACE_SEPARATED) [constructor]
    cls.add_constructor([param('std::string const &', 'outputFileName'), param('ns3::FileAggregator::FileType', 'fileType', default_value='::ns3::FileAggregator::FileType::SPACE_SEPARATED')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::SetFileType(ns3::FileAggregator::FileType fileType) [member function]
    cls.add_method('SetFileType', 
                   'void', 
                   [param('ns3::FileAggregator::FileType', 'fileType')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::SetHeading(std::string const & heading) [member function]
    cls.add_method('SetHeading', 
                   'void', 
                   [param('std::string const &', 'heading')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Set1dFormat(std::string const & format) [member function]
    cls.add_method('Set1dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Set2dFormat(std::string const & format) [member function]
    cls.add_method('Set2dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Set3dFormat(std::string const & format) [member function]
    cls.add_method('Set3dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Set4dFormat(std::string const & format) [member function]
    cls.add_method('Set4dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Set5dFormat(std::string const & format) [member function]
    cls.add_method('Set5dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Set6dFormat(std::string const & format) [member function]
    cls.add_method('Set6dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Set7dFormat(std::string const & format) [member function]
    cls.add_method('Set7dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Set8dFormat(std::string const & format) [member function]
    cls.add_method('Set8dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Set9dFormat(std::string const & format) [member function]
    cls.add_method('Set9dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Set10dFormat(std::string const & format) [member function]
    cls.add_method('Set10dFormat', 
                   'void', 
                   [param('std::string const &', 'format')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Write1d(std::string context, double v1) [member function]
    cls.add_method('Write1d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'v1')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Write2d(std::string context, double v1, double v2) [member function]
    cls.add_method('Write2d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'v1'), param('double', 'v2')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Write3d(std::string context, double v1, double v2, double v3) [member function]
    cls.add_method('Write3d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'v1'), param('double', 'v2'), param('double', 'v3')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Write4d(std::string context, double v1, double v2, double v3, double v4) [member function]
    cls.add_method('Write4d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'v1'), param('double', 'v2'), param('double', 'v3'), param('double', 'v4')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Write5d(std::string context, double v1, double v2, double v3, double v4, double v5) [member function]
    cls.add_method('Write5d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'v1'), param('double', 'v2'), param('double', 'v3'), param('double', 'v4'), param('double', 'v5')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Write6d(std::string context, double v1, double v2, double v3, double v4, double v5, double v6) [member function]
    cls.add_method('Write6d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'v1'), param('double', 'v2'), param('double', 'v3'), param('double', 'v4'), param('double', 'v5'), param('double', 'v6')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Write7d(std::string context, double v1, double v2, double v3, double v4, double v5, double v6, double v7) [member function]
    cls.add_method('Write7d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'v1'), param('double', 'v2'), param('double', 'v3'), param('double', 'v4'), param('double', 'v5'), param('double', 'v6'), param('double', 'v7')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Write8d(std::string context, double v1, double v2, double v3, double v4, double v5, double v6, double v7, double v8) [member function]
    cls.add_method('Write8d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'v1'), param('double', 'v2'), param('double', 'v3'), param('double', 'v4'), param('double', 'v5'), param('double', 'v6'), param('double', 'v7'), param('double', 'v8')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Write9d(std::string context, double v1, double v2, double v3, double v4, double v5, double v6, double v7, double v8, double v9) [member function]
    cls.add_method('Write9d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'v1'), param('double', 'v2'), param('double', 'v3'), param('double', 'v4'), param('double', 'v5'), param('double', 'v6'), param('double', 'v7'), param('double', 'v8'), param('double', 'v9')])
    ## file-aggregator.h (module 'stats'): void ns3::FileAggregator::Write10d(std::string context, double v1, double v2, double v3, double v4, double v5, double v6, double v7, double v8, double v9, double v10) [member function]
    cls.add_method('Write10d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'v1'), param('double', 'v2'), param('double', 'v3'), param('double', 'v4'), param('double', 'v5'), param('double', 'v6'), param('double', 'v7'), param('double', 'v8'), param('double', 'v9'), param('double', 'v10')])
    return

def register_Ns3GnuplotAggregator_methods(root_module, cls):
    ## gnuplot-aggregator.h (module 'stats'): ns3::GnuplotAggregator::GnuplotAggregator(ns3::GnuplotAggregator const & arg0) [constructor]
    cls.add_constructor([param('ns3::GnuplotAggregator const &', 'arg0')])
    ## gnuplot-aggregator.h (module 'stats'): ns3::GnuplotAggregator::GnuplotAggregator(std::string const & outputFileNameWithoutExtension) [constructor]
    cls.add_constructor([param('std::string const &', 'outputFileNameWithoutExtension')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::Add2dDataset(std::string const & dataset, std::string const & title) [member function]
    cls.add_method('Add2dDataset', 
                   'void', 
                   [param('std::string const &', 'dataset'), param('std::string const &', 'title')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::AppendExtra(std::string const & extra) [member function]
    cls.add_method('AppendExtra', 
                   'void', 
                   [param('std::string const &', 'extra')])
    ## gnuplot-aggregator.h (module 'stats'): static ns3::TypeId ns3::GnuplotAggregator::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## gnuplot-aggregator.h (module 'stats'): static void ns3::GnuplotAggregator::Set2dDatasetDefaultErrorBars(ns3::Gnuplot2dDataset::ErrorBars errorBars) [member function]
    cls.add_method('Set2dDatasetDefaultErrorBars', 
                   'void', 
                   [param('ns3::Gnuplot2dDataset::ErrorBars', 'errorBars')], 
                   is_static=True)
    ## gnuplot-aggregator.h (module 'stats'): static void ns3::GnuplotAggregator::Set2dDatasetDefaultExtra(std::string const & extra) [member function]
    cls.add_method('Set2dDatasetDefaultExtra', 
                   'void', 
                   [param('std::string const &', 'extra')], 
                   is_static=True)
    ## gnuplot-aggregator.h (module 'stats'): static void ns3::GnuplotAggregator::Set2dDatasetDefaultStyle(ns3::Gnuplot2dDataset::Style style) [member function]
    cls.add_method('Set2dDatasetDefaultStyle', 
                   'void', 
                   [param('ns3::Gnuplot2dDataset::Style', 'style')], 
                   is_static=True)
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::Set2dDatasetErrorBars(std::string const & dataset, ns3::Gnuplot2dDataset::ErrorBars errorBars) [member function]
    cls.add_method('Set2dDatasetErrorBars', 
                   'void', 
                   [param('std::string const &', 'dataset'), param('ns3::Gnuplot2dDataset::ErrorBars', 'errorBars')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::Set2dDatasetExtra(std::string const & dataset, std::string const & extra) [member function]
    cls.add_method('Set2dDatasetExtra', 
                   'void', 
                   [param('std::string const &', 'dataset'), param('std::string const &', 'extra')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::Set2dDatasetStyle(std::string const & dataset, ns3::Gnuplot2dDataset::Style style) [member function]
    cls.add_method('Set2dDatasetStyle', 
                   'void', 
                   [param('std::string const &', 'dataset'), param('ns3::Gnuplot2dDataset::Style', 'style')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::SetExtra(std::string const & extra) [member function]
    cls.add_method('SetExtra', 
                   'void', 
                   [param('std::string const &', 'extra')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::SetKeyLocation(ns3::GnuplotAggregator::KeyLocation keyLocation) [member function]
    cls.add_method('SetKeyLocation', 
                   'void', 
                   [param('ns3::GnuplotAggregator::KeyLocation', 'keyLocation')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::SetLegend(std::string const & xLegend, std::string const & yLegend) [member function]
    cls.add_method('SetLegend', 
                   'void', 
                   [param('std::string const &', 'xLegend'), param('std::string const &', 'yLegend')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::SetTerminal(std::string const & terminal) [member function]
    cls.add_method('SetTerminal', 
                   'void', 
                   [param('std::string const &', 'terminal')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::SetTitle(std::string const & title) [member function]
    cls.add_method('SetTitle', 
                   'void', 
                   [param('std::string const &', 'title')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::Write2d(std::string context, double x, double y) [member function]
    cls.add_method('Write2d', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'x'), param('double', 'y')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::Write2dDatasetEmptyLine(std::string const & dataset) [member function]
    cls.add_method('Write2dDatasetEmptyLine', 
                   'void', 
                   [param('std::string const &', 'dataset')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::Write2dWithXErrorDelta(std::string context, double x, double y, double errorDelta) [member function]
    cls.add_method('Write2dWithXErrorDelta', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'x'), param('double', 'y'), param('double', 'errorDelta')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::Write2dWithXYErrorDelta(std::string context, double x, double y, double xErrorDelta, double yErrorDelta) [member function]
    cls.add_method('Write2dWithXYErrorDelta', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'x'), param('double', 'y'), param('double', 'xErrorDelta'), param('double', 'yErrorDelta')])
    ## gnuplot-aggregator.h (module 'stats'): void ns3::GnuplotAggregator::Write2dWithYErrorDelta(std::string context, double x, double y, double errorDelta) [member function]
    cls.add_method('Write2dWithYErrorDelta', 
                   'void', 
                   [param('std::string', 'context'), param('double', 'x'), param('double', 'y'), param('double', 'errorDelta')])
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

def register_Ns3MinMaxAvgTotalCalculator__Double_methods(root_module, cls):
    ## basic-data-calculators.h (module 'stats'): ns3::MinMaxAvgTotalCalculator<double>::MinMaxAvgTotalCalculator(ns3::MinMaxAvgTotalCalculator<double> const & arg0) [constructor]
    cls.add_constructor([param('ns3::MinMaxAvgTotalCalculator< double > const &', 'arg0')])
    ## basic-data-calculators.h (module 'stats'): ns3::MinMaxAvgTotalCalculator<double>::MinMaxAvgTotalCalculator() [constructor]
    cls.add_constructor([])
    ## basic-data-calculators.h (module 'stats'): static ns3::TypeId ns3::MinMaxAvgTotalCalculator<double>::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## basic-data-calculators.h (module 'stats'): void ns3::MinMaxAvgTotalCalculator<double>::Output(ns3::DataOutputCallback & callback) const [member function]
    cls.add_method('Output', 
                   'void', 
                   [param('ns3::DataOutputCallback &', 'callback')], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): void ns3::MinMaxAvgTotalCalculator<double>::Reset() [member function]
    cls.add_method('Reset', 
                   'void', 
                   [])
    ## basic-data-calculators.h (module 'stats'): void ns3::MinMaxAvgTotalCalculator<double>::Update(double const i) [member function]
    cls.add_method('Update', 
                   'void', 
                   [param('double const', 'i')])
    ## basic-data-calculators.h (module 'stats'): long int ns3::MinMaxAvgTotalCalculator<double>::getCount() const [member function]
    cls.add_method('getCount', 
                   'long int', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<double>::getMax() const [member function]
    cls.add_method('getMax', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<double>::getMean() const [member function]
    cls.add_method('getMean', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<double>::getMin() const [member function]
    cls.add_method('getMin', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<double>::getSqrSum() const [member function]
    cls.add_method('getSqrSum', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<double>::getStddev() const [member function]
    cls.add_method('getStddev', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<double>::getSum() const [member function]
    cls.add_method('getSum', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): double ns3::MinMaxAvgTotalCalculator<double>::getVariance() const [member function]
    cls.add_method('getVariance', 
                   'double', 
                   [], 
                   is_const=True, is_virtual=True)
    ## basic-data-calculators.h (module 'stats'): void ns3::MinMaxAvgTotalCalculator<double>::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
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

def register_Ns3OmnetDataOutput_methods(root_module, cls):
    ## omnet-data-output.h (module 'stats'): ns3::OmnetDataOutput::OmnetDataOutput(ns3::OmnetDataOutput const & arg0) [constructor]
    cls.add_constructor([param('ns3::OmnetDataOutput const &', 'arg0')])
    ## omnet-data-output.h (module 'stats'): ns3::OmnetDataOutput::OmnetDataOutput() [constructor]
    cls.add_constructor([])
    ## omnet-data-output.h (module 'stats'): static ns3::TypeId ns3::OmnetDataOutput::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## omnet-data-output.h (module 'stats'): void ns3::OmnetDataOutput::Output(ns3::DataCollector & dc) [member function]
    cls.add_method('Output', 
                   'void', 
                   [param('ns3::DataCollector &', 'dc')], 
                   is_virtual=True)
    ## omnet-data-output.h (module 'stats'): void ns3::OmnetDataOutput::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
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

def register_Ns3TimeMinMaxAvgTotalCalculator_methods(root_module, cls):
    ## time-data-calculators.h (module 'stats'): ns3::TimeMinMaxAvgTotalCalculator::TimeMinMaxAvgTotalCalculator(ns3::TimeMinMaxAvgTotalCalculator const & arg0) [constructor]
    cls.add_constructor([param('ns3::TimeMinMaxAvgTotalCalculator const &', 'arg0')])
    ## time-data-calculators.h (module 'stats'): ns3::TimeMinMaxAvgTotalCalculator::TimeMinMaxAvgTotalCalculator() [constructor]
    cls.add_constructor([])
    ## time-data-calculators.h (module 'stats'): static ns3::TypeId ns3::TimeMinMaxAvgTotalCalculator::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## time-data-calculators.h (module 'stats'): void ns3::TimeMinMaxAvgTotalCalculator::Output(ns3::DataOutputCallback & callback) const [member function]
    cls.add_method('Output', 
                   'void', 
                   [param('ns3::DataOutputCallback &', 'callback')], 
                   is_const=True, is_virtual=True)
    ## time-data-calculators.h (module 'stats'): void ns3::TimeMinMaxAvgTotalCalculator::Update(ns3::Time const i) [member function]
    cls.add_method('Update', 
                   'void', 
                   [param('ns3::Time const', 'i')])
    ## time-data-calculators.h (module 'stats'): void ns3::TimeMinMaxAvgTotalCalculator::DoDispose() [member function]
    cls.add_method('DoDispose', 
                   'void', 
                   [], 
                   is_virtual=True, visibility='protected')
    return

def register_Ns3TimeProbe_methods(root_module, cls):
    ## time-probe.h (module 'stats'): ns3::TimeProbe::TimeProbe(ns3::TimeProbe const & arg0) [constructor]
    cls.add_constructor([param('ns3::TimeProbe const &', 'arg0')])
    ## time-probe.h (module 'stats'): ns3::TimeProbe::TimeProbe() [constructor]
    cls.add_constructor([])
    ## time-probe.h (module 'stats'): bool ns3::TimeProbe::ConnectByObject(std::string traceSource, ns3::Ptr<ns3::Object> obj) [member function]
    cls.add_method('ConnectByObject', 
                   'bool', 
                   [param('std::string', 'traceSource'), param('ns3::Ptr< ns3::Object >', 'obj')], 
                   is_virtual=True)
    ## time-probe.h (module 'stats'): void ns3::TimeProbe::ConnectByPath(std::string path) [member function]
    cls.add_method('ConnectByPath', 
                   'void', 
                   [param('std::string', 'path')], 
                   is_virtual=True)
    ## time-probe.h (module 'stats'): static ns3::TypeId ns3::TimeProbe::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## time-probe.h (module 'stats'): double ns3::TimeProbe::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_const=True)
    ## time-probe.h (module 'stats'): void ns3::TimeProbe::SetValue(ns3::Time value) [member function]
    cls.add_method('SetValue', 
                   'void', 
                   [param('ns3::Time', 'value')])
    ## time-probe.h (module 'stats'): static void ns3::TimeProbe::SetValueByPath(std::string path, ns3::Time value) [member function]
    cls.add_method('SetValueByPath', 
                   'void', 
                   [param('std::string', 'path'), param('ns3::Time', 'value')], 
                   is_static=True)
    return

def register_Ns3TimeSeriesAdaptor_methods(root_module, cls):
    ## time-series-adaptor.h (module 'stats'): ns3::TimeSeriesAdaptor::TimeSeriesAdaptor(ns3::TimeSeriesAdaptor const & arg0) [constructor]
    cls.add_constructor([param('ns3::TimeSeriesAdaptor const &', 'arg0')])
    ## time-series-adaptor.h (module 'stats'): ns3::TimeSeriesAdaptor::TimeSeriesAdaptor() [constructor]
    cls.add_constructor([])
    ## time-series-adaptor.h (module 'stats'): static ns3::TypeId ns3::TimeSeriesAdaptor::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## time-series-adaptor.h (module 'stats'): void ns3::TimeSeriesAdaptor::TraceSinkBoolean(bool oldData, bool newData) [member function]
    cls.add_method('TraceSinkBoolean', 
                   'void', 
                   [param('bool', 'oldData'), param('bool', 'newData')])
    ## time-series-adaptor.h (module 'stats'): void ns3::TimeSeriesAdaptor::TraceSinkDouble(double oldData, double newData) [member function]
    cls.add_method('TraceSinkDouble', 
                   'void', 
                   [param('double', 'oldData'), param('double', 'newData')])
    ## time-series-adaptor.h (module 'stats'): void ns3::TimeSeriesAdaptor::TraceSinkUinteger16(uint16_t oldData, uint16_t newData) [member function]
    cls.add_method('TraceSinkUinteger16', 
                   'void', 
                   [param('uint16_t', 'oldData'), param('uint16_t', 'newData')])
    ## time-series-adaptor.h (module 'stats'): void ns3::TimeSeriesAdaptor::TraceSinkUinteger32(uint32_t oldData, uint32_t newData) [member function]
    cls.add_method('TraceSinkUinteger32', 
                   'void', 
                   [param('uint32_t', 'oldData'), param('uint32_t', 'newData')])
    ## time-series-adaptor.h (module 'stats'): void ns3::TimeSeriesAdaptor::TraceSinkUinteger8(uint8_t oldData, uint8_t newData) [member function]
    cls.add_method('TraceSinkUinteger8', 
                   'void', 
                   [param('uint8_t', 'oldData'), param('uint8_t', 'newData')])
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

def register_Ns3Uinteger16Probe_methods(root_module, cls):
    ## uinteger-16-probe.h (module 'stats'): ns3::Uinteger16Probe::Uinteger16Probe(ns3::Uinteger16Probe const & arg0) [constructor]
    cls.add_constructor([param('ns3::Uinteger16Probe const &', 'arg0')])
    ## uinteger-16-probe.h (module 'stats'): ns3::Uinteger16Probe::Uinteger16Probe() [constructor]
    cls.add_constructor([])
    ## uinteger-16-probe.h (module 'stats'): bool ns3::Uinteger16Probe::ConnectByObject(std::string traceSource, ns3::Ptr<ns3::Object> obj) [member function]
    cls.add_method('ConnectByObject', 
                   'bool', 
                   [param('std::string', 'traceSource'), param('ns3::Ptr< ns3::Object >', 'obj')], 
                   is_virtual=True)
    ## uinteger-16-probe.h (module 'stats'): void ns3::Uinteger16Probe::ConnectByPath(std::string path) [member function]
    cls.add_method('ConnectByPath', 
                   'void', 
                   [param('std::string', 'path')], 
                   is_virtual=True)
    ## uinteger-16-probe.h (module 'stats'): static ns3::TypeId ns3::Uinteger16Probe::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## uinteger-16-probe.h (module 'stats'): uint16_t ns3::Uinteger16Probe::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'uint16_t', 
                   [], 
                   is_const=True)
    ## uinteger-16-probe.h (module 'stats'): void ns3::Uinteger16Probe::SetValue(uint16_t value) [member function]
    cls.add_method('SetValue', 
                   'void', 
                   [param('uint16_t', 'value')])
    ## uinteger-16-probe.h (module 'stats'): static void ns3::Uinteger16Probe::SetValueByPath(std::string path, uint16_t value) [member function]
    cls.add_method('SetValueByPath', 
                   'void', 
                   [param('std::string', 'path'), param('uint16_t', 'value')], 
                   is_static=True)
    return

def register_Ns3Uinteger32Probe_methods(root_module, cls):
    ## uinteger-32-probe.h (module 'stats'): ns3::Uinteger32Probe::Uinteger32Probe(ns3::Uinteger32Probe const & arg0) [constructor]
    cls.add_constructor([param('ns3::Uinteger32Probe const &', 'arg0')])
    ## uinteger-32-probe.h (module 'stats'): ns3::Uinteger32Probe::Uinteger32Probe() [constructor]
    cls.add_constructor([])
    ## uinteger-32-probe.h (module 'stats'): bool ns3::Uinteger32Probe::ConnectByObject(std::string traceSource, ns3::Ptr<ns3::Object> obj) [member function]
    cls.add_method('ConnectByObject', 
                   'bool', 
                   [param('std::string', 'traceSource'), param('ns3::Ptr< ns3::Object >', 'obj')], 
                   is_virtual=True)
    ## uinteger-32-probe.h (module 'stats'): void ns3::Uinteger32Probe::ConnectByPath(std::string path) [member function]
    cls.add_method('ConnectByPath', 
                   'void', 
                   [param('std::string', 'path')], 
                   is_virtual=True)
    ## uinteger-32-probe.h (module 'stats'): static ns3::TypeId ns3::Uinteger32Probe::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## uinteger-32-probe.h (module 'stats'): uint32_t ns3::Uinteger32Probe::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'uint32_t', 
                   [], 
                   is_const=True)
    ## uinteger-32-probe.h (module 'stats'): void ns3::Uinteger32Probe::SetValue(uint32_t value) [member function]
    cls.add_method('SetValue', 
                   'void', 
                   [param('uint32_t', 'value')])
    ## uinteger-32-probe.h (module 'stats'): static void ns3::Uinteger32Probe::SetValueByPath(std::string path, uint32_t value) [member function]
    cls.add_method('SetValueByPath', 
                   'void', 
                   [param('std::string', 'path'), param('uint32_t', 'value')], 
                   is_static=True)
    return

def register_Ns3Uinteger8Probe_methods(root_module, cls):
    ## uinteger-8-probe.h (module 'stats'): ns3::Uinteger8Probe::Uinteger8Probe(ns3::Uinteger8Probe const & arg0) [constructor]
    cls.add_constructor([param('ns3::Uinteger8Probe const &', 'arg0')])
    ## uinteger-8-probe.h (module 'stats'): ns3::Uinteger8Probe::Uinteger8Probe() [constructor]
    cls.add_constructor([])
    ## uinteger-8-probe.h (module 'stats'): bool ns3::Uinteger8Probe::ConnectByObject(std::string traceSource, ns3::Ptr<ns3::Object> obj) [member function]
    cls.add_method('ConnectByObject', 
                   'bool', 
                   [param('std::string', 'traceSource'), param('ns3::Ptr< ns3::Object >', 'obj')], 
                   is_virtual=True)
    ## uinteger-8-probe.h (module 'stats'): void ns3::Uinteger8Probe::ConnectByPath(std::string path) [member function]
    cls.add_method('ConnectByPath', 
                   'void', 
                   [param('std::string', 'path')], 
                   is_virtual=True)
    ## uinteger-8-probe.h (module 'stats'): static ns3::TypeId ns3::Uinteger8Probe::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## uinteger-8-probe.h (module 'stats'): uint8_t ns3::Uinteger8Probe::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'uint8_t', 
                   [], 
                   is_const=True)
    ## uinteger-8-probe.h (module 'stats'): void ns3::Uinteger8Probe::SetValue(uint8_t value) [member function]
    cls.add_method('SetValue', 
                   'void', 
                   [param('uint8_t', 'value')])
    ## uinteger-8-probe.h (module 'stats'): static void ns3::Uinteger8Probe::SetValueByPath(std::string path, uint8_t value) [member function]
    cls.add_method('SetValueByPath', 
                   'void', 
                   [param('std::string', 'path'), param('uint8_t', 'value')], 
                   is_static=True)
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

def register_Ns3BooleanProbe_methods(root_module, cls):
    ## boolean-probe.h (module 'stats'): ns3::BooleanProbe::BooleanProbe(ns3::BooleanProbe const & arg0) [constructor]
    cls.add_constructor([param('ns3::BooleanProbe const &', 'arg0')])
    ## boolean-probe.h (module 'stats'): ns3::BooleanProbe::BooleanProbe() [constructor]
    cls.add_constructor([])
    ## boolean-probe.h (module 'stats'): bool ns3::BooleanProbe::ConnectByObject(std::string traceSource, ns3::Ptr<ns3::Object> obj) [member function]
    cls.add_method('ConnectByObject', 
                   'bool', 
                   [param('std::string', 'traceSource'), param('ns3::Ptr< ns3::Object >', 'obj')], 
                   is_virtual=True)
    ## boolean-probe.h (module 'stats'): void ns3::BooleanProbe::ConnectByPath(std::string path) [member function]
    cls.add_method('ConnectByPath', 
                   'void', 
                   [param('std::string', 'path')], 
                   is_virtual=True)
    ## boolean-probe.h (module 'stats'): static ns3::TypeId ns3::BooleanProbe::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## boolean-probe.h (module 'stats'): bool ns3::BooleanProbe::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'bool', 
                   [], 
                   is_const=True)
    ## boolean-probe.h (module 'stats'): void ns3::BooleanProbe::SetValue(bool value) [member function]
    cls.add_method('SetValue', 
                   'void', 
                   [param('bool', 'value')])
    ## boolean-probe.h (module 'stats'): static void ns3::BooleanProbe::SetValueByPath(std::string path, bool value) [member function]
    cls.add_method('SetValueByPath', 
                   'void', 
                   [param('std::string', 'path'), param('bool', 'value')], 
                   is_static=True)
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

def register_Ns3CallbackImpl__Void_Bool_Bool_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, bool, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, bool, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, bool, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, bool, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, bool, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, bool, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, bool, bool, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(bool arg0, bool arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('bool', 'arg0'), param('bool', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Double_Double_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, double, double, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, double, double, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, double, double, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, double, double, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, double, double, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, double, double, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, double, double, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(double arg0, double arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('double', 'arg0'), param('double', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3CallbackImpl__Void_Unsigned_char_Unsigned_char_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, unsigned char, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned char, unsigned char, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, unsigned char, unsigned char, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, unsigned char, unsigned char, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, unsigned char, unsigned char, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, unsigned char, unsigned char, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, unsigned char, unsigned char, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(unsigned char arg0, unsigned char arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('unsigned char', 'arg0'), param('unsigned char', 'arg1')], 
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

def register_Ns3CallbackImpl__Void_Unsigned_short_Unsigned_short_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_Ns3Empty_methods(root_module, cls):
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned short, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl() [constructor]
    cls.add_constructor([])
    ## callback.h (module 'core'): ns3::CallbackImpl<void, unsigned short, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::CallbackImpl(ns3::CallbackImpl<void, unsigned short, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty> const & arg0) [constructor]
    cls.add_constructor([param('ns3::CallbackImpl< void, unsigned short, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty > const &', 'arg0')])
    ## callback.h (module 'core'): static std::string ns3::CallbackImpl<void, unsigned short, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::DoGetTypeid() [member function]
    cls.add_method('DoGetTypeid', 
                   'std::string', 
                   [], 
                   is_static=True)
    ## callback.h (module 'core'): std::string ns3::CallbackImpl<void, unsigned short, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::GetTypeid() const [member function]
    cls.add_method('GetTypeid', 
                   'std::string', 
                   [], 
                   is_const=True, is_virtual=True)
    ## callback.h (module 'core'): void ns3::CallbackImpl<void, unsigned short, unsigned short, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty, ns3::empty>::operator()(short unsigned int arg0, short unsigned int arg1) [member operator]
    cls.add_method('operator()', 
                   'void', 
                   [param('short unsigned int', 'arg0'), param('short unsigned int', 'arg1')], 
                   custom_name='__call__', is_pure_virtual=True, is_virtual=True)
    return

def register_Ns3DoubleProbe_methods(root_module, cls):
    ## double-probe.h (module 'stats'): ns3::DoubleProbe::DoubleProbe(ns3::DoubleProbe const & arg0) [constructor]
    cls.add_constructor([param('ns3::DoubleProbe const &', 'arg0')])
    ## double-probe.h (module 'stats'): ns3::DoubleProbe::DoubleProbe() [constructor]
    cls.add_constructor([])
    ## double-probe.h (module 'stats'): bool ns3::DoubleProbe::ConnectByObject(std::string traceSource, ns3::Ptr<ns3::Object> obj) [member function]
    cls.add_method('ConnectByObject', 
                   'bool', 
                   [param('std::string', 'traceSource'), param('ns3::Ptr< ns3::Object >', 'obj')], 
                   is_virtual=True)
    ## double-probe.h (module 'stats'): void ns3::DoubleProbe::ConnectByPath(std::string path) [member function]
    cls.add_method('ConnectByPath', 
                   'void', 
                   [param('std::string', 'path')], 
                   is_virtual=True)
    ## double-probe.h (module 'stats'): static ns3::TypeId ns3::DoubleProbe::GetTypeId() [member function]
    cls.add_method('GetTypeId', 
                   'ns3::TypeId', 
                   [], 
                   is_static=True)
    ## double-probe.h (module 'stats'): double ns3::DoubleProbe::GetValue() const [member function]
    cls.add_method('GetValue', 
                   'double', 
                   [], 
                   is_const=True)
    ## double-probe.h (module 'stats'): void ns3::DoubleProbe::SetValue(double value) [member function]
    cls.add_method('SetValue', 
                   'void', 
                   [param('double', 'value')])
    ## double-probe.h (module 'stats'): static void ns3::DoubleProbe::SetValueByPath(std::string path, double value) [member function]
    cls.add_method('SetValueByPath', 
                   'void', 
                   [param('std::string', 'path'), param('double', 'value')], 
                   is_static=True)
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
    ## get-wildcard-matches.h (module 'stats'): std::string ns3::GetWildcardMatches(std::string const & configPath, std::string const & matchedPath, std::string const & wildcardSeparator=" ") [free function]
    module.add_function('GetWildcardMatches', 
                        'std::string', 
                        [param('std::string const &', 'configPath'), param('std::string const &', 'matchedPath'), param('std::string const &', 'wildcardSeparator', default_value='" "')])
    ## data-calculator.h (module 'stats'): bool ns3::isNaN(double x) [free function]
    module.add_function('isNaN', 
                        'bool', 
                        [param('double', 'x')])
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

