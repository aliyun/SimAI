import re
import os
import sys

from pybindgen.typehandlers import base as typehandlers
from pybindgen import ReturnValue, Parameter
from pybindgen.cppmethod import CustomCppMethodWrapper, CustomCppConstructorWrapper
from pybindgen.typehandlers.codesink import MemoryCodeSink
from pybindgen.typehandlers import ctypeparser
from pybindgen import cppclass, param, retval
import warnings

from pybindgen.typehandlers.base import CodeGenerationError


class ArgvParam(Parameter):
    """
    Converts a python list-of-strings argument to a pair of 'int argc,
    char *argv[]' arguments to pass into C.

    One Python argument becomes two C function arguments -> it's a miracle!

    Note: this parameter type handler is not registered by any name;
    must be used explicitly.
    """

    DIRECTIONS = [Parameter.DIRECTION_IN]
    CTYPES = []
    
    def convert_c_to_python(self, wrapper):
        raise NotImplementedError

    def convert_python_to_c(self, wrapper):
        py_name = wrapper.declarations.declare_variable('PyObject*', 'py_' + self.name)
        argc_var = wrapper.declarations.declare_variable('int', 'argc')
        name = wrapper.declarations.declare_variable('char**', self.name)
        idx = wrapper.declarations.declare_variable('Py_ssize_t', 'idx')
        wrapper.parse_params.add_parameter('O!', ['&PyList_Type', '&'+py_name], self.name)

        #wrapper.before_call.write_error_check('!PyList_Check(%s)' % py_name) # XXX

        wrapper.before_call.write_code("%s = (char **) malloc(sizeof(char*)*PyList_Size(%s));"
                                       % (name, py_name))
        wrapper.before_call.add_cleanup_code('free(%s);' % name)
        wrapper.before_call.write_code('''
for (%(idx)s = 0; %(idx)s < PyList_Size(%(py_name)s); %(idx)s++)
{
''' % vars())
        wrapper.before_call.sink.indent()
        wrapper.before_call.write_code('''
PyObject *item = PyList_GET_ITEM(%(py_name)s, %(idx)s);
''' % vars())
        #wrapper.before_call.write_error_check('item == NULL')
        wrapper.before_call.write_error_check(
            '\n'
            '#if PY_VERSION_HEX >= 0x03000000\n'
            '!PyUnicode_Check(item)\n'
            '#else\n'
            '!PyString_Check(item)\n'
            '#endif\n',
            failure_cleanup=('PyErr_SetString(PyExc_TypeError, '
                             '"argument %s must be a list of strings");') % self.name)
        wrapper.before_call.write_code(
            '#if PY_VERSION_HEX >= 0x03000000\n'
            '{var}[{idx}] = const_cast<char *> (PyUnicode_AsUTF8(item));\n'
            '#else\n'
            '{var}[{idx}] = PyString_AsString(item);\n'
            '#endif\n'
            .format(var=name, idx=idx))
        wrapper.before_call.sink.unindent()
        wrapper.before_call.write_code('}')
        wrapper.before_call.write_code('%s = PyList_Size(%s);' % (argc_var, py_name))
        
        wrapper.call_params.append(argc_var)
        wrapper.call_params.append(name)



def Simulator_customizations(module):
    Simulator = module['ns3::Simulator']

    ## Simulator::Schedule(delay, callback, ...user..args...)
    Simulator.add_custom_method_wrapper("Schedule", "_wrap_Simulator_Schedule",
                                        flags=["METH_VARARGS", "METH_KEYWORDS", "METH_STATIC"])

    ## Simulator::ScheduleNow(callback, ...user..args...)
    Simulator.add_custom_method_wrapper("ScheduleNow", "_wrap_Simulator_ScheduleNow",
                                        flags=["METH_VARARGS", "METH_KEYWORDS", "METH_STATIC"])

    ## Simulator::ScheduleWithContext(delay, callback, ...user..args...)
    Simulator.add_custom_method_wrapper("ScheduleWithContext", "_wrap_Simulator_ScheduleWithContext",
                                        flags=["METH_VARARGS", "METH_KEYWORDS", "METH_STATIC"])

    ## Simulator::ScheduleDestroy(callback, ...user..args...)
    Simulator.add_custom_method_wrapper("ScheduleDestroy", "_wrap_Simulator_ScheduleDestroy",
                                        flags=["METH_VARARGS", "METH_KEYWORDS", "METH_STATIC"])

    Simulator.add_custom_method_wrapper("Run", "_wrap_Simulator_Run",
                                        flags=["METH_VARARGS", "METH_KEYWORDS", "METH_STATIC"])


def CommandLine_customizations(module):
    CommandLine = module['ns3::CommandLine']
    CommandLine.add_method('Parse', None, [ArgvParam(None, 'argv')],
                           is_static=False)
    CommandLine.add_custom_method_wrapper("AddValue", "_wrap_CommandLine_AddValue",
                                          flags=["METH_VARARGS", "METH_KEYWORDS"])


def TypeId_customizations(module):
    TypeId = module['ns3::TypeId']
    TypeId.add_custom_method_wrapper("LookupByNameFailSafe", "_wrap_TypeId_LookupByNameFailSafe",
                                     flags=["METH_VARARGS", "METH_KEYWORDS", "METH_STATIC"])


def add_std_ofstream(module):
    module.add_include('<fstream>')
    ostream = module.add_class('ostream', foreign_cpp_namespace='::std')
    ostream.set_cannot_be_constructed("abstract base class")
    ofstream = module.add_class('ofstream', foreign_cpp_namespace='::std', parent=ostream)
    ofstream.add_enum('openmode', [
            ('app', 'std::ios_base::app'),
            ('ate', 'std::ios_base::ate'),
            ('binary', 'std::ios_base::binary'),
            ('in', 'std::ios_base::in'),
            ('out', 'std::ios_base::out'),
            ('trunc', 'std::ios_base::trunc'),
            ])
    ofstream.add_constructor([Parameter.new("const char *", 'filename'),
                              Parameter.new("::std::ofstream::openmode", 'mode', default_value="std::ios_base::out")])
    ofstream.add_method('close', None, [])

    import pybindgen.typehandlers.base
    for alias in "std::_Ios_Openmode", "std::ios::openmode":
        pybindgen.typehandlers.base.param_type_matcher.add_type_alias(alias, "int")

    for flag in 'in', 'out', 'ate', 'app', 'trunc', 'binary':
        module.after_init.write_code('PyModule_AddIntConstant(m, (char *) "STD_IOS_%s", std::ios::%s);'
                                     % (flag.upper(), flag))



def add_ipv4_address_tp_hash(module):
    module.body.writeln('''
long
_ns3_Ipv4Address_tp_hash (PyObject *obj)
{
   PyNs3Ipv4Address *addr = reinterpret_cast<PyNs3Ipv4Address *> (obj);
   return static_cast<long> (ns3::Ipv4AddressHash () (*addr->obj));
}
''')
    module.header.writeln('long _ns3_Ipv4Address_tp_hash (PyObject *obj);')
    module['Ipv4Address'].pytype.slots['tp_hash'] = "_ns3_Ipv4Address_tp_hash"


def post_register_types(root_module):
    Simulator_customizations(root_module)
    CommandLine_customizations(root_module)
    TypeId_customizations(root_module)
    add_std_ofstream(root_module)

    enabled_features = os.environ['NS3_ENABLED_FEATURES'].split(',')    

    if 'RealTime' not in enabled_features:
        for clsname in ['WallClockSynchronizer', 'RealtimeSimulatorImpl']:
            root_module.classes.remove(root_module['ns3::%s' % clsname])
        root_module.enums.remove(root_module['ns3::RealtimeSimulatorImpl::SynchronizationMode'])

    # Issue 318: PyEval_InitThreads deprecated after Python 3.6
    if sys.hexversion < 0x03070000:
        root_module.after_init.write_code("PyEval_InitThreads();")

# these are already in the main script, so commented out here
# Object_customizations(root_module)
# Attribute_customizations(root_module)
#def post_register_functions(root_module):
#    pass


