"""
This is a modest collection of file utilities we regularly use.
"""

import os
import subprocess
import sys


def print_memory_usage():
    # this prints the total memory usage of the machine
    # TODO get the memory usage of this script only
    command = "free -h | head -n 2"
    os.system(command)


def runCommand(command, show_output=True, quit=True, error_message=""):
    """ Small function to handle running bash subprocess
    *command*: The bash command to run on the system
    *show_output*: Default True. If False, output of the command will be hidden
    *quit*: Default True. If False, the script will not quit if the subprocess has an error
    *error_message*: Error message to show if the command failed
    """
    if show_output:
        print("Running: %s" % (command))
    try:
        if show_output:
            subprocess.check_call(command.split())
        else:
            subprocess.check_call(command.split(), stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
    except subprocess.CalledProcessError:
        if error_message:
            sys.stderr.write(error_message)
        if quit:
            raise


def checkDir(directory):
    """ Analagous to mkdir -p directory from the command line
    """
    if not os.path.isdir(directory):
        print("Dir %s doesn't exist. Creating it" % (directory))
        try:
            os.makedirs(directory)
        except OSError:
            # if multiple parallel processes are running, they could be trying to create the same directory at the same time
            print("Dir %s already exists" % (directory))


def readDict(f, fromCol=1, toCol=2, sep='\t'):
    """
    Read the dict from the given tab-delimited file. The dict
    maps items in the fromCol to items in the toCol (1-based column index).
    """
    itemMap = {}
    for line in open(f, 'r').readlines():
        if line=='':
            continue
        if line[0]=='#':
            continue
        items = line.rstrip().split(sep)
        if len(items)<max(fromCol, toCol):
            continue
        key = items[fromCol-1]
        val = items[toCol-1]
        if key=='':
            continue
        if val=='':
            continue
        itemMap[key] = val
    return itemMap


def readColumnsSep(f, sep='\t', *cols):
    """
    Read multiple columns and return the items from those columns
    in each line as a tuple.

    foo.txt:
        a b c
        d e f
        g h i

    Calling "readColumnsSep('foo.txt', ' ',1, 3,)" will return:
        [(a, c), (d, f), (g, i)]

    """
    if len(cols)==0:
        return []
    rows = []
    for line in open(f, 'r').readlines():
        if line=='':
            continue
        if line[0]=='#':
            continue
        items = line.rstrip().split(sep)
        if len(items)<max(cols):
            continue
        rows.append(tuple([items[c-1] for c in cols]))
    return rows


def readColumns(f, *cols):
    """
    Read multiple columns and return the items from those columns
    in each line as a tuple.

    foo.txt:
        a b c
        d e f
        g h i

    Calling "readColumns('foo.txt', 1, 3)" will return:
        [(a, c), (d, f), (g, i)]

    """
    if len(cols)==0:
        return []
    rows = []
    for line in open(f, 'r').readlines():
        if line=='':
            continue
        if line[0]=='#':
            continue
        items = line.rstrip().split('\t')
        if len(items)<max(cols):
            continue
        rows.append(tuple([items[c-1] for c in cols]))
    return rows


def readItemList(f, col=1, sep='\t', var_type='str'):
    """
    Read the given column of the tab-delimited file f
    and return it as a list. Col is the 1-based column index.

    *var_type*: variable type to which items will be cast.
        Can be 'str', 'int', or 'float'
    """
    itemlist = []
    for line in open(f, 'r').readlines():
        if line=='':
            continue
        if line[0]=='#':
            continue
        items = line.rstrip().split(sep)
        if len(items)<col:
            continue
        if var_type == 'str':
            itemlist.append(items[col-1])
        elif var_type == 'int':
            itemlist.append(int(items[col-1]))
        elif var_type == 'float':
            itemlist.append(float(items[col-1]))
        else:
            print("Error: variable type '%s' not recognized!" % (var_type))
            return 
    return itemlist


def readItemSet(f, col=1, sep='\t', var_type='str'):
    """
    Read the given column of the tab-delimited file f
    and return it as a set. Col is the 1-based column index.

    A wrapper to readItemList, returning a set instead of a list.
    """
    return set(readItemList(f, col=col, sep=sep, var_type='str'))
