#! /usr/bin/env python3

import sys
import os.path

import pybindgen.settings
from pybindgen.castxmlparser import ModuleParser, PygenClassifier, PygenSection, WrapperWarning, find_declaration_from_name
from pybindgen.typehandlers.codesink import FileCodeSink
from pygccxml.declarations import templates
from pygccxml.declarations.enumeration import enumeration_t
from pygccxml.declarations.class_declaration import class_t
from pygccxml.declarations.free_calldef import free_function_t
from pygccxml.declarations.calldef_members import constructor_t, member_function_t
from pygccxml.declarations.calldef import calldef_t

## we need the smart pointer type transformation to be active even
## during castxml scanning.
import ns3modulegen_core_customizations


## silence castxmlparser errors; we only want error handling in the
## generated python script, not while scanning.
class ErrorHandler(pybindgen.settings.ErrorHandler):
    def handle_error(self, dummy_wrapper, dummy_exception, dummy_traceback_):
        return True
pybindgen.settings.error_handler = ErrorHandler()
import warnings
warnings.filterwarnings(category=WrapperWarning, action='ignore')


import ns3modulescan
type_annotations = ns3modulescan.type_annotations


def get_ns3_relative_path(path):
    l = []
    head = path
    if not path:
        return
    while head:
        new_head, tail = os.path.split(head)
        if new_head == head:
            raise ValueError
        head = new_head
        if tail == 'ns3':
            return os.path.join(*l)
        l.insert(0, tail)
    raise AssertionError("is the path %r inside ns3?!" % path)

class PreScanHook:

    def __init__(self, headers_map, module):
        self.headers_map = headers_map
        self.module = module

    def __call__(self, module_parser,
                 pygccxml_definition,
                 global_annotations,
                 parameter_annotations):
        try:
            ns3_header = get_ns3_relative_path(pygccxml_definition.location.file_name)
        except ValueError: # the header is not from ns3
            return # ignore the definition, it's not ns-3 def.
        if not ns3_header:
            return
        definition_module = self.headers_map[ns3_header]

        ## Note: we don't include line numbers in the comments because
        ## those numbers are very likely to change frequently, which would
        ## cause needless changes, since the generated python files are
        ## kept under version control.

        #global_annotations['pygen_comment'] = "%s:%i: %s" % \
        #    (ns3_header, pygccxml_definition.location.line, pygccxml_definition)
        global_annotations['pygen_comment'] = "%s (module %r): %s" % \
            (ns3_header, definition_module, pygccxml_definition)


        ## handle ns3::Object::GetObject (left to its own devices,
        ## pybindgen will generate a mangled name containing the template
        ## argument type name).
        if isinstance(pygccxml_definition, member_function_t) \
                and pygccxml_definition.parent.name == 'Object' \
                and pygccxml_definition.name == 'GetObject':
            template_args = templates.args(str(pygccxml_definition))
            if template_args == ['ns3::Object']:
                global_annotations['template_instance_names'] = 'ns3::Object=>GetObject'

        ## Don't wrap Simulator::Schedule* (manually wrapped)
        if isinstance(pygccxml_definition, member_function_t) \
                and pygccxml_definition.parent.name == 'Simulator' \
                and pygccxml_definition.name.startswith('Schedule'):
            global_annotations['ignore'] = None

        # manually wrapped
        if isinstance(pygccxml_definition, member_function_t) \
                and pygccxml_definition.parent.name == 'Simulator' \
                and pygccxml_definition.name == 'Run':
            global_annotations['ignore'] = True

        ## http://www.gccxml.org/Bug/view.php?id=9915
        if isinstance(pygccxml_definition, calldef_t):
            for arg in pygccxml_definition.arguments:
                if arg.default_value is None:
                    continue
                elif arg.default_value == "ns3::MilliSeconds( )":
                    arg.default_value = "ns3::MilliSeconds(0)"
                elif arg.default_value == "ns3::Seconds( )":
                    arg.default_value = "ns3::Seconds(0)"

        ## classes
        if isinstance(pygccxml_definition, class_t):
            print(pygccxml_definition, file=sys.stderr)
            # no need for helper classes to allow subclassing in Python, I think...
            #if pygccxml_definition.name.endswith('Helper'):
            #    global_annotations['allow_subclassing'] = 'false'

            #
            # If a class is template instantiation, even if the
            # template was defined in some other module, if a template
            # argument belongs to this module then the template
            # instantiation will belong to this module.
            # 
            if templates.is_instantiation(pygccxml_definition.decl_string):
                cls_name, template_parameters = templates.split(pygccxml_definition.name)
                template_parameters_decls = [find_declaration_from_name(module_parser.global_ns, templ_param)
                                             for templ_param in template_parameters]
                #print >> sys.stderr, "********************", cls_name, repr(template_parameters_decls)
                
                template_parameters_modules = []
                for templ in template_parameters_decls:
                    if not hasattr(templ, 'location'):
                        continue
                    try:
                        h = get_ns3_relative_path(templ.location.file_name)
                    except ValueError:
                        continue
                    template_parameters_modules.append(self.headers_map[h])

                for templ_mod in template_parameters_modules:
                    if templ_mod == self.module:
                        definition_module = templ_mod
                        break
                #print >> sys.stderr, "********************", cls_name, repr(template_parameters_modules)


            if definition_module != self.module:
                global_annotations['import_from_module'] = 'ns.%s' % (definition_module.replace('-', '_'),)

            if pygccxml_definition.decl_string.startswith('::ns3::SimpleRefCount<'):
                global_annotations['incref_method'] = 'Ref'
                global_annotations['decref_method'] = 'Unref'
                global_annotations['peekref_method'] = 'GetReferenceCount'
                global_annotations['automatic_type_narrowing'] = 'true'
                return

            if pygccxml_definition.decl_string.startswith('::ns3::Callback<'):
                # manually handled in ns3modulegen_core_customizations.py
                global_annotations['ignore'] = None
                return

            if pygccxml_definition.decl_string.startswith('::ns3::TracedCallback<'):
                global_annotations['ignore'] = None
                return

            if pygccxml_definition.decl_string.startswith('::ns3::Ptr<'):
                # handled by pybindgen "type transformation"
                global_annotations['ignore'] = None
                return

            # table driven class customization
            try:
                annotations = type_annotations[pygccxml_definition.decl_string]
            except KeyError:
                pass
            else:
                global_annotations.update(annotations)

        ## enums
        if isinstance(pygccxml_definition, enumeration_t):
            if definition_module != self.module:
                global_annotations['import_from_module'] = 'ns.%s' % definition_module

        ## free functions
        if isinstance(pygccxml_definition, free_function_t):

            if definition_module != self.module:
                global_annotations['ignore'] = None
                return

            if pygccxml_definition.name == 'PeekPointer':
                global_annotations['ignore'] = None
                return

        ## table driven methods/constructors/functions customization
        if isinstance(pygccxml_definition, (free_function_t, member_function_t, constructor_t)):
            try:
                annotations = type_annotations[str(pygccxml_definition)]
            except KeyError:
                pass
            else:
                for key,value in list(annotations.items()):
                    if key == 'params':
                        parameter_annotations.update (value)
                        del annotations['params']
                global_annotations.update(annotations)


# def post_scan_hook(dummy_module_parser, dummy_pygccxml_definition, pybindgen_wrapper):
#     ## classes
#     if isinstance(pybindgen_wrapper, CppClass):
#         if pybindgen_wrapper.name.endswith('Checker'):
#             print >> sys.stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", pybindgen_wrapper
#             #pybindgen_wrapper.set_instance_creation_function(AttributeChecker_instance_creation_function)


def scan_callback_classes(module_parser, callback_classes_file):
    callback_classes_file.write("callback_classes = [\n")
    for cls in module_parser.module_namespace.classes(function=module_parser.location_filter,
                                                      recursive=False):
        if not cls.name.startswith("Callback<"):
            continue
        assert templates.is_instantiation(cls.decl_string), "%s is not a template instantiation" % cls
        dummy_cls_name, template_parameters = templates.split(cls.decl_string)
        callback_classes_file.write("    %r,\n" % template_parameters)
    callback_classes_file.write("]\n")


def ns3_module_scan(top_builddir, module_name, headers_map, output_file_name, cflags):
    module_parser = ModuleParser('ns.%s' % module_name.replace('-', '_'), 'ns3')
    module_parser.add_pre_scan_hook(PreScanHook(headers_map, module_name))
    #module_parser.add_post_scan_hook(post_scan_hook)

    castxml_options = dict(
        include_paths=[top_builddir, os.path.join(top_builddir, "include")],
         define_symbols={
            #'NS3_ASSERT_ENABLE': None,
            #'NS3_LOG_ENABLE': None,
            },
        cflags=('-std=c++17 %s' % cflags)
        )

    try:
        os.unlink(output_file_name)
    except OSError:
        pass
    try:
        os.makedirs(os.path.dirname(output_file_name))
    except OSError:
        pass
    output_file = open(output_file_name, "wt")
    output_sink = FileCodeSink(output_file)

    # if there exists a scan-header.h file in src/<module>/bindings or contrib/<module>/bindings,
    # scan it, otherwise scan ns3/xxxx-module.h.
    scan_header = os.path.join(os.path.dirname(output_file_name), "scan-header.h")
    if not os.path.exists(scan_header):
        scan_header = os.path.join(top_builddir, "ns3", "%s-module.h" % module_name)
    if not os.path.exists(scan_header):
        scan_header = os.path.join(top_builddir, "include", "ns3", "%s-module.h" % module_name)

    module_parser.parse_init([scan_header],
                             None, whitelist_paths=[top_builddir],
                             pygen_sink=output_sink,
                             castxml_options=castxml_options)
    module_parser.scan_types()

    callback_classes_file = open(os.path.join(os.path.dirname(output_file_name), "callbacks_list.py"), "wt")
    scan_callback_classes(module_parser, callback_classes_file)
    callback_classes_file.close()


    module_parser.scan_methods()
    module_parser.scan_functions()
    module_parser.parse_finalize()

    output_file.close()
    os.chmod(output_file_name, 0o400)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("ns3modulescan-modular.py top_builddir module_path module_headers output_file_name cflags")
        sys.exit(1)
    if os.path.exists(sys.argv[3]):
        import json
        with open(sys.argv[3], "r") as f:
            module_headers = json.load(f)
    else:
        module_headers = eval(sys.argv[3])
    ns3_module_scan(sys.argv[1], sys.argv[2], module_headers, sys.argv[4], sys.argv[5])
    sys.exit(0)
