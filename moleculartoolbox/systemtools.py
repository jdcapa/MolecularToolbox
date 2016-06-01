# -*- coding: UTF-8 -*-
"""This is a collection of system routines for a Linux machine (Python 2.7)."""
from __future__ import print_function
from __future__ import division
import sys
import os
import subprocess


def query_process(command):
    """Retrieve an output of a Unix command."""
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    if err:
        sys.exit(err)
    return out


def run_process(command, silent=False):
    """
    Run a Unix system command.

    This is a os.system substitute, to better understand eventual errors
    """
    try:
        retcode = subprocess.call(command, shell=True)
        if retcode < 0:
            errorprint("Program terminated", -retcode)
        else:
            if not silent:
                errorprint("\t...done.")
    except OSError, e:
        errorprint("Execution failed:", e)


def errorprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_memory(modifier=0.8, unit="m"):
    """
    Retrieve the available memory and return a fraction.

    Available units:
        b: bytes
        k: kilo     1024 b
        m: mega     1048576 b
        g: giga     1073741824 b
    """
    total_query = "/usr/bin/free -b | awk '/Mem/ { print $2 } '"
    used_query = "/usr/bin/free -b | awk '/buffers\/cache/ { print $3 } '"
    try:
        total_ram = float(query_process(total_query)[:-1])
        used_ram = float(query_process(used_query)[:-1])
        available_ram = (total_ram - used_ram) * modifier
    except:
        print("systemtools.get_memory(): Could not read RAM usage.")
        available_ram = 2048.0 * 1048576
    if unit == "g":
        unit_modifier = 1073741824
    elif unit == "m":
        unit_modifier = 1048576
    elif unit == "k":
        unit_modifier = 1024
    else:  # unit = "b"
        unit_modifier = 1
    return available_ram / unit_modifier


def write_string_to_file(string, filename, path=''):
    """Write a string to a file with an optional path parameter."""
    with open(os.path.join(path, filename), 'w') as write_to_file:
        write_to_file.write(string)


def file2list(path):
    """Convert file contents to a list."""
    list_of_lines = []
    if os.path.exists(path):
        with open(path) as input_file:
            for line in input_file:
                list_of_lines.append(line.strip())
        return list_of_lines
    else:
        sys.exit('{0} does not exist.'.format(path))


def list2file(output_list, output_file_path):
    """Write a list to a file, where list elements are separated by newline."""
    dirname = os.path.dirname(output_file_path)
    if not os.path.exists(dirname):
        try:
            print("Creating {0}".format(dirname))
            os.makedirs(dirname)
        except Exception, e:
            raise e
    with open(output_file_path, 'w') as output_file:
        output_file.write("\n".join(output_list))


def find_file_in_PP(file_name):
    """
    Return the full path of a file, if it's found somewhere in the $PYTHONPATH.
    """
    for path in sys.path:
        path_to_file = os.path.join(path, file_name)
        if os.path.exists(path_to_file):
            return path_to_file
    sys.exit("Could not find {} in $PYTHONPATH.".format(file_name))
