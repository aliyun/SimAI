from __future__ import print_function
import warnings
import sys
import os
import pybindgen.settings
from pybindgen import Module, FileCodeSink, param, retval, cppclass, typehandlers
from pybindgen.module import MultiSectionFactory
import ns3modulegen_core_customizations

import logging

pybindgen.settings.wrapper_registry = pybindgen.settings.StdMapWrapperRegistry

import traceback

class ErrorHandler(pybindgen.settings.ErrorHandler):

    def __init__(self, apidefs_file):
        self.apidefs_file = apidefs_file

    def handle_error(self, wrapper, exception, traceback_):
        stack = getattr(wrapper, 'stack_where_defined', [])
        stack.reverse()
        for l in stack:
            if l[0] == self.apidefs_file:
                warnings.warn_explicit("exception %r in wrapper %s" % (exception, wrapper),
                                       Warning, l[0], l[1])
                break
        else:
            warnings.warn("exception %r in wrapper %s" % (exception, wrapper))
        return True


#print >> sys.stderr, ">>>>>>>>>>>>>>>>>>>>>>>>>>>> ", bool(eval(os.environ["GCC_RTTI_ABI_COMPLETE"]))
pybindgen.settings.gcc_rtti_abi_complete = bool(eval(os.environ["GCC_RTTI_ABI_COMPLETE"]))

class MyMultiSectionFactory(MultiSectionFactory):
    def __init__(self, main_file_name):
        super(MyMultiSectionFactory, self).__init__()
        self.main_file_name = main_file_name
        self.main_sink = FileCodeSink(open(main_file_name, "wt"))
        self.header_name = "ns3module.h"
        header_file_name = os.path.join(os.path.dirname(self.main_file_name), self.header_name)
        #print >> sys.stderr, ">>>>>>>>>>>>>>>>>", header_file_name, main_file_name
        self.header_sink = FileCodeSink(open(header_file_name, "wt"))
    def get_section_code_sink(self, section_name):
        return self.main_sink
    def get_main_code_sink(self):
        return self.main_sink
    def get_common_header_code_sink(self):
        return self.header_sink
    def get_common_header_include(self):
        return '"%s"' % self.header_name
    def close(self):
        self.header_sink.file.close()
        self.main_sink.file.close()



def main(argv):
    logging.basicConfig()
    logging.getLogger("pybindgen.typehandlers").setLevel(logging.DEBUG)

    module_abs_src_path, target, extension_name, output_cc_file_name = argv[1:]
    module_name = os.path.basename(module_abs_src_path)
    out = MyMultiSectionFactory(output_cc_file_name)

    sys.path.insert(0, os.path.join(module_abs_src_path, "bindings"))
    try:
        module_apidefs = __import__("modulegen__%s" % target)
        del sys.modules["modulegen__%s" % target]
        try:
            module_customization = __import__("modulegen_customizations")
            del sys.modules["modulegen_customizations"]
        except ImportError:
            module_customization = object()

        try:
            from callbacks_list import callback_classes
        except ImportError as ex:
            print("***************", repr(ex), file=sys.stderr)
            callback_classes = []
        else:
            print(">>>>>>>>>>>>>>>>", repr(callback_classes), file=sys.stderr)

    finally:
        sys.path.pop(0)

    apidefs_file, dummy = os.path.splitext(module_apidefs.__file__)
    apidefs_file += '.py'
    pybindgen.settings.error_handler = ErrorHandler(apidefs_file)

    root_module = module_apidefs.module_init()
    root_module.set_name(extension_name)
    root_module.add_include('"ns3/%s-module.h"' % module_name)

    ns3modulegen_core_customizations.add_std_ios_openmode(root_module)

    # -----------
    module_apidefs.register_types(root_module)

    if hasattr(module_customization, 'post_register_types'):
        module_customization.post_register_types(root_module)

    # register Callback<...> type handlers
    ns3modulegen_core_customizations.register_callback_classes(root_module.after_forward_declarations,
                                                               callback_classes)

    # -----------
    module_apidefs.register_methods(root_module)

    if hasattr(module_customization, 'post_register_methods'):
        module_customization.post_register_methods(root_module)

    ns3modulegen_core_customizations.Object_customizations(root_module)
    ns3modulegen_core_customizations.Attribute_customizations(root_module)
    ns3modulegen_core_customizations.generate_callback_classes(root_module,
                                                               callback_classes)

    # -----------
    module_apidefs.register_functions(root_module)

    if hasattr(module_customization, 'post_register_functions'):
        module_customization.post_register_functions(root_module)

    # -----------
    root_module.generate(out)

if __name__ == '__main__':
    import sys
    main(sys.argv)
