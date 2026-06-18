from pybindgen import Module, FileCodeSink, write_preamble, param, retval

def register_types(module):
    module.add_class('MyClass')
    
def register_methods(root_module):
    MyClass = root_module['MyClass']
    MyClass.add_constructor([], visibility='public')
    MyClass.add_constructor([param('double', 's'), param('double', 'l'), param('double', 'mean')], visibility='public')

def register_functions(module):
    module.add_function('SomeFunction', 'int', [param('int', 'xpto')])

