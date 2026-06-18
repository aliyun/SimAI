#! /usr/bin/env python3

# These methods are used by test.py to look for and read the
# .ns3rc configuration file, which is used to specify the modules that
# should be enabled

import os
import sys

def get_list_from_file(file_path, list_name):
    '''Looks for a Python list called list_name in the file specified
    by file_path and returns it.

    If the file or list name aren't found, this function will return
    an empty list.

    '''

    list = []

    # Read in the file if it exists.
    if os.path.exists(file_path):
        with open(file_path, "r") as file_in:

            # Look for the list.
            list_string = ""
            parsing_multiline_list = False
            for line in file_in:

                # Remove any comments.
                if '#' in line:
                    (line, comment) = line.split('#', 1)

                # Parse the line.
                if list_name in line or parsing_multiline_list:
                    list_string += line

                    # Handle multiline lists.
                    if ']' not in list_string:
                        parsing_multiline_list = True
                    else:
                        # Evaluate the list once its end is reached.
                        # Make the split function only split it once.
                        list = eval(list_string.split('=', 1)[1].strip())
                        break

    return list


def get_bool_from_file(file_path, bool_name, value_if_missing):
    '''Looks for a Python boolean variable called bool_name in the
    file specified by file_path and returns its value.

    If the file or boolean variable aren't found, this function will
    return value_if_missing.

    '''

    # Read in the file if it exists.
    if os.path.exists(file_path):
        with open(file_path, "r") as file_in:

            # Look for the boolean variable.
            bool_found = False
            for line in file_in:

                # Remove any comments.
                if '#' in line:
                    (line, comment) = line.split('#', 1)

                # Parse the line.
                if bool_name in line:
                    # Evaluate the variable's line once it is found.  Make
                    # the split function only split it once.
                    bool = eval(line.split('=', 1)[1].strip())
                    bool_found = True
                    break

    if bool_found:
        return bool
    else:
        return value_if_missing


# Reads the NS-3 configuration file and returns a list of enabled modules.
#
# This function first looks for the ns3 configuration file (.ns3rc) in
# the current working directory and then looks in the ~ directory.
def read_config_file():
    # By default, all modules will be enabled, examples will be disabled,
    # and tests will be disabled.
    modules_enabled  = ['all_modules']
    examples_enabled = False
    tests_enabled    = False

    # See if the ns3 configuration file exists in the current working
    # directory and then look for it in the ~ directory.
    config_file_exists = False
    dot_ns3rc_name = '.ns3rc'
    dot_ns3rc_path = dot_ns3rc_name
    if not os.path.exists(dot_ns3rc_path):
        dot_ns3rc_path = os.path.expanduser('~/') + dot_ns3rc_name
        if not os.path.exists(dot_ns3rc_path):
            # Return all of the default values if the .ns3rc file can't be found.
            return (config_file_exists, modules_enabled, examples_enabled, tests_enabled)

    config_file_exists = True

    # Read in the enabled modules.
    modules_enabled = get_list_from_file(dot_ns3rc_path, 'modules_enabled')
    if not modules_enabled:
        # Enable all modules if the modules_enabled line can't be found.
        modules_enabled = ['all_modules']

    # Read in whether examples should be enabled or not.
    value_if_missing = False
    examples_enabled = get_bool_from_file(dot_ns3rc_path, 'examples_enabled', value_if_missing)

    # Read in whether tests should be enabled or not.
    value_if_missing = False
    tests_enabled = get_bool_from_file(dot_ns3rc_path, 'tests_enabled', value_if_missing)

    return (config_file_exists, modules_enabled, examples_enabled, tests_enabled)

