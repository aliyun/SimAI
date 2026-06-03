# Copyright (c) 2007 RADLogic
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Provide various handy Python functions.

Running this script directly will execute the doctests.

Functions:
int2bin(i, n) -- Convert integer to binary string.
bin2int(bin_string) -- Convert binary string to integer.
reverse(input_string) -- Reverse a string.
transpose(matrix) -- Transpose a list of lists.
polygon_area(points_list) -- Calculate the area of an arbitrary polygon.
timestamp() -- Return string containing current time stamp.
pt2str(point) -- Return prettier string version of point tuple.
gcf(a, b) -- Return the greatest common factor of two numbers.
lcm(a, b) -- Return the least common multiple of two numbers.
permutations(input_list) -- Generate all permutations of a list of items.
reduce_fraction(fraction) -- Reduce fraction (num, denom) to simplest form.
quantile(l, p) -- Return p quantile of list l. E.g. p=0.25 for q1.
trim(l) -- Discard values in list more than 1.5*IQR outside IQR.
nice_units(value) -- Return value converted to human readable units.
uniquify(seq) -- Return sequence with duplicate items in sequence seq removed.
reverse_dict(d) -- Return the dictionary with the items as keys and vice-versa.
lsb(x, n) -- Return the n least significant bits of x.
gray_encode(i) -- Gray encode the given integer.
random_vec(bits, max_value=None) -- Return a random binary vector.
binary_range(bits) -- Return list of all possible binary numbers width=bits.
float_range([start], stop, [step]) -- Return range of floats.
find_common_fixes(s1, s2) -- Find common (prefix, suffix) of two strings.
is_rotated(seq1, seq2) -- Return true if the list is a rotation of other list.
getmodule(obj) -- Return the module that contains the object definition of obj.
                  (use inspect.getmodule instead, though)
get_args(argv) -- Store command-line args in a dictionary.

This module requires Python >= 2.2

"""
__author__ = 'Tim Wegener <twegener@radlogic.com.au>'
__date__ = '$Date: 2007/03/27 03:15:06 $'
__version__ = '$Revision: 0.45 $'
__credits__ = """
              David Chandler, for polygon area algorithm.
               (http://www.davidchandler.com/AreaOfAGeneralPolygon.pdf)
              """

import re
import sys
import time
import random

try:
    True, False
except NameError:
    True, False = (1==1, 0==1)


def int2bin(i, n):
    """Convert decimal integer i to n-bit binary number (string).

    >>> int2bin(0, 8)
    '00000000'

    >>> int2bin(123, 8)
    '01111011'

    >>> int2bin(123L, 8)
    '01111011'

    >>> int2bin(15, 2)
    Traceback (most recent call last):
    ValueError: Value too large for given number of bits.

    """
    hex2bin = {'0': '0000', '1': '0001', '2': '0010', '3': '0011',
               '4': '0100', '5': '0101', '6': '0110', '7': '0111',
               '8': '1000', '9': '1001', 'a': '1010', 'b': '1011',
               'c': '1100', 'd': '1101', 'e': '1110', 'f': '1111'}
    # Convert to hex then map each hex digit to binary equivalent.
    result = ''.join([hex2bin[x] for x in hex(i).lower().replace('l','')[2:]])
                      
    # Shrink result to appropriate length.
    # Raise an error if the value is changed by the truncation.
    if '1' in result[:-n]:
        raise ValueError("Value too large for given number of bits.")
    result = result[-n:]
    # Zero-pad if length longer than mapped result.
    result = '0'*(n-len(result)) + result
    return result


def bin2int(bin_string):
    """Convert binary number string to decimal integer.
    
    Note: Python > v2 has int(bin_string, 2)

    >>> bin2int('1111')
    15

    >>> bin2int('0101')
    5

    """
##     result = 0
##     bin_list = list(bin_string)
##     if len(filter(lambda x: x in ('1','0'), bin_list)) < len(bin_list):
##         raise Exception ("bin2int: Error - not a binary number: %s"
##                          % bin_string)
##     bit_list = map(int, bin_list)
##     bit_list.reverse()  # Make most significant bit have highest index.
##     for bit_place in range(len(bit_list)):
##         result = result + ((2**bit_place) * bit_list[bit_place])
##     return result
    return int(bin_string, 2)


def reverse(input_string):
    """Reverse a string. Useful for strings of binary numbers.

    >>> reverse('abc')
    'cba'

    """
    str_list = list(input_string)
    str_list.reverse()
    return ''.join(str_list)


def transpose(matrix):
    """Transpose a list of lists.

    >>> transpose([['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']])
    [['a', 'd', 'g'], ['b', 'e', 'h'], ['c', 'f', 'i']]

    >>> transpose([['a', 'b', 'c'], ['d', 'e', 'f']])
    [['a', 'd'], ['b', 'e'], ['c', 'f']]

    >>> transpose([['a', 'b'], ['d', 'e'], ['g', 'h']])
    [['a', 'd', 'g'], ['b', 'e', 'h']]

    """
    result = zip(*matrix)
    # Convert list of tuples to list of lists.
    # map is faster than a list comprehension since it is being used with
    # a built-in function as an argument.
    result = map(list, result)
    return result


def polygon_area(points_list, precision=100):
    """Calculate area of an arbitrary polygon using an algorithm from the web.

    Return the area of the polygon as a positive float. 

    Arguments:
    points_list -- list of point tuples [(x0, y0), (x1, y1), (x2, y2), ...]
                   (Unclosed polygons will be closed automatically.
    precision -- Internal arithmetic precision (integer arithmetic).

    >>> polygon_area([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0), (0, 0)])
    3.0

    Credits:
    Area of a General Polygon by David Chandler
    http://www.davidchandler.com/AreaOfAGeneralPolygon.pdf
    
    """
    # Scale up co-ordinates and convert them to integers.
    for i in range(len(points_list)):
        points_list[i] = (int(points_list[i][0] * precision),
                          int(points_list[i][1] * precision))
    # Close polygon if not closed.
    if points_list[-1] != points_list[0]:
        points_list.append(points_list[0])
    # Calculate area.
    area = 0
    for i in range(len(points_list)-1):
        (x_i, y_i) = points_list[i]
        (x_i_plus_1, y_i_plus_1) = points_list[i+1]
        area = area + (x_i_plus_1 * y_i) - (y_i_plus_1 * x_i)
    area = abs(area / 2)
    # Unscale area.
    area = float(area)/(precision**2)
    return area


def timestamp():
    """Return string containing current time stamp.

    Note: In Python 2 onwards can use time.asctime() with no arguments.

    """

    return time.asctime()


def pt2str(point):
    """Return prettier string version of point tuple.

    >>> pt2str((1.8, 1.9))
    '(1.8, 1.9)'

    """
    return "(%s, %s)" % (str(point[0]), str(point[1]))


def gcf(a, b, epsilon=1e-16):
    """Return the greatest common factor of a and b, using Euclidean algorithm.

    Arguments:
    a, b -- two numbers
            If both numbers are integers return an integer result, 
            otherwise return a float result.
    epsilon -- floats less than this magnitude are considered to be zero
               (default: 1e-16)

    Examples:

    >>> gcf(12, 34)
    2

    >>> gcf(13.5, 4)
    0.5

    >>> gcf(-2, 4)
    2

    >>> gcf(5, 0)
    5

    By (a convenient) definition:
    >>> gcf(0, 0)
    0

    """
    result = max(a, b)
    remainder = min(a, b)
    while remainder and abs(remainder) > epsilon:
        new_remainder = result % remainder
        result = remainder
        remainder = new_remainder
    return abs(result)

def lcm(a, b, precision=None):
    """Return the least common multiple of a and b, using the gcf function.

    Arguments:
    a, b -- two numbers. If both are integers return an integer result, 
            otherwise a return a float result.
    precision -- scaling factor if a and/or b are floats.

    >>> lcm(21, 6)
    42

    >>> lcm(2.5, 3.5)
    17.5

    >>> str(lcm(1.5e-8, 2.5e-8, precision=1e9))
    '7.5e-08'

    By (an arbitrary) definition:
    >>> lcm(0, 0)
    0

    """
    # Note: Dummy precision argument is for backwards compatibility.
    # Do the division first.
    # (See http://en.wikipedia.org/wiki/Least_common_multiple )
    denom = gcf(a, b)
    if denom == 0:
        result = 0
    else:
        result = a * (b / denom)
    return result


def permutations(input_list):
    """Return a list containing all permutations of the input list.

    Note: This is a recursive function.

    >>> perms = permutations(['a', 'b', 'c'])
    >>> perms.sort()
    >>> for perm in perms:
    ...     print perm
    ['a', 'b', 'c']
    ['a', 'c', 'b']
    ['b', 'a', 'c']
    ['b', 'c', 'a']
    ['c', 'a', 'b']
    ['c', 'b', 'a']

    """
    out_lists = []
    if len(input_list) > 1:
        # Extract first item in list.
        item = input_list[0]
        # Find all permutations of remainder of list. (Recursive call.)
        sub_lists = permutations(input_list[1:])
        # For every permutation of the sub list...
        for sub_list in sub_lists:
            # Insert the extracted first item at every position of the list.
            for i in range(len(input_list)):
                new_list = sub_list[:]
                new_list.insert(i, item)
                out_lists.append(new_list)
    else:
        # Termination condition: only one item in input list.
        out_lists = [input_list]
    return out_lists


def reduce_fraction(fraction):
    """Reduce fraction tuple to simplest form. fraction=(num, denom)
    
    >>> reduce_fraction((14, 7))
    (2, 1)

    >>> reduce_fraction((-2, 4))
    (-1, 2)

    >>> reduce_fraction((0, 4))
    (0, 1)

    >>> reduce_fraction((4, 0))
    (1, 0)
    
    """
    (numerator, denominator) = fraction
    common_factor = abs(gcf(numerator, denominator))
    result = (numerator/common_factor, denominator/common_factor)
    return result


def quantile(l, p):
    """Return p quantile of list l. E.g. p=0.25 for q1.

    See:
    http://rweb.stat.umn.edu/R/library/base/html/quantile.html

    """
    l_sort = l[:]
    l_sort.sort()
    n = len(l)
    r = 1 + ((n - 1) * p)
    i = int(r)
    f = r - i
    if i < n:
        result =  (1-f)*l_sort[i-1] + f*l_sort[i]
    else:
        result = l_sort[i-1]
    return result


def trim(l):
    """Discard values in list more than 1.5*IQR outside IQR.

    (IQR is inter-quartile-range)

    This function uses rad_util.quantile

    1.5*IQR -- mild outlier
    3*IQR -- extreme outlier

    See:
    http://wind.cc.whecn.edu/~pwildman/statnew/section_7_-_exploratory_data_analysis.htm

    """
    l_sort = l[:]
    l_sort.sort()
    # Calculate medianscore  (based on stats.py lmedianscore by Gary Strangman)
    if len(l_sort) % 2 == 0:
        # If even number of scores, average middle 2.
        index = int(len(l_sort) / 2)  # Integer division correct
        median = float(l_sort[index] + l_sort[index-1]) / 2
    else:
        # int division gives mid value when count from 0
        index = int(len(l_sort) / 2)
        median = l_sort[index]
    # Calculate IQR.
    q1 = quantile(l_sort, 0.25)
    q3 = quantile(l_sort, 0.75)
    iqr = q3 - q1
    iqr_extra = iqr * 1.5
    def in_interval(x, i=iqr_extra, q1=q1, q3=q3):
        return (x >= q1-i and x <= q3+i)
    l_trimmed = [x for x in l_sort if in_interval(x)]
    return l_trimmed


def nice_units(value, dp=0, sigfigs=None, suffix='', space=' ',
               use_extra_prefixes=False, use_full_name=False, mode='si'):
    """Return value converted to human readable units eg milli, micro, etc.

    Arguments:
    value -- number in base units
    dp -- number of decimal places to display (rounded)
    sigfigs -- number of significant figures to display (rounded)
               This overrides dp if set.
    suffix -- optional unit suffix to append to unit multiplier
    space -- separator between value and unit multiplier (default: ' ')
    use_extra_prefixes -- use hecto, deka, deci and centi as well if set.
                          (default: False)
    use_full_name -- use full name for multiplier symbol,
                     e.g. milli instead of m
                     (default: False)
    mode -- 'si' for SI prefixes, 'bin' for binary multipliers (1024, etc.)
            (Default: 'si')

    SI prefixes from:
    http://physics.nist.gov/cuu/Units/prefixes.html
    (Greek mu changed to u.)
    Binary prefixes based on:
    http://physics.nist.gov/cuu/Units/binary.html

    >>> nice_units(2e-11)
    '20 p'

    >>> nice_units(2e-11, space='')
    '20p'

    """
    si_prefixes = {1e24:  ('Y', 'yotta'),
                   1e21:  ('Z', 'zetta'),
                   1e18:  ('E', 'exa'),
                   1e15:  ('P', 'peta'),
                   1e12:  ('T', 'tera'),
                   1e9:   ('G', 'giga'),
                   1e6:   ('M', 'mega'),
                   1e3:   ('k', 'kilo'),
                   1e-3:  ('m', 'milli'),
                   1e-6:  ('u', 'micro'),
                   1e-9:  ('n', 'nano'),
                   1e-12: ('p', 'pico'),
                   1e-15: ('f', 'femto'),
                   1e-18: ('a', 'atto'),
                   1e-21: ('z', 'zepto'),
                   1e-24: ('y', 'yocto')
                   }
    if use_extra_prefixes:
        si_prefixes.update({1e2:  ('h', 'hecto'),
                            1e1:  ('da', 'deka'),
                            1e-1: ('d', 'deci'),
                            1e-2: ('c', 'centi')
                            })
    bin_prefixes = {2**10: ('K', 'kilo'),
                    2**20: ('M', 'mega'),
                    2**30: ('G', 'mega'),
                    2**40: ('T', 'tera'),
                    2**50: ('P', 'peta'),
                    2**60: ('E', 'exa')
                    }
    if mode == 'bin':
        prefixes = bin_prefixes
    else:
        prefixes = si_prefixes
    prefixes[1] = ('', '')  # Unity.
    # Determine appropriate multiplier.
    multipliers = prefixes.keys()
    multipliers.sort()
    mult = None
    for i in range(len(multipliers) - 1):
        lower_mult = multipliers[i]
        upper_mult = multipliers[i+1]
        if lower_mult <= value < upper_mult:
            mult_i = i
            break
    if mult is None:
        if value < multipliers[0]:
            mult_i = 0
        elif value >= multipliers[-1]:
            mult_i = len(multipliers) - 1
    mult = multipliers[mult_i]
    # Convert value for this multiplier.
    new_value = value / mult
    # Deal with special case due to rounding.
    if sigfigs is None:
        if mult_i < (len(multipliers) - 1) and \
               round(new_value, dp) == \
               round((multipliers[mult_i+1] / mult), dp):
            mult = multipliers[mult_i + 1]
            new_value = value / mult
    # Concatenate multiplier symbol.
    if use_full_name:
        label_type = 1
    else:
        label_type = 0
    # Round and truncate to appropriate precision.
    if sigfigs is None:
        str_value = eval('"%.'+str(dp)+'f" % new_value', locals(), {})
    else:
        str_value = eval('"%.'+str(sigfigs)+'g" % new_value', locals(), {})
    return str_value + space + prefixes[mult][label_type] + suffix


def uniquify(seq, preserve_order=False):
    """Return sequence with duplicate items in sequence seq removed.

    The code is based on usenet post by Tim Peters.

    This code is O(N) if the sequence items are hashable, O(N**2) if not.
    
    Peter Bengtsson has a blog post with an empirical comparison of other
    approaches:
    http://www.peterbe.com/plog/uniqifiers-benchmark

    If order is not important and the sequence items are hashable then
    list(set(seq)) is readable and efficient.

    If order is important and the sequence items are hashable generator
    expressions can be used (in py >= 2.4) (useful for large sequences):
    seen = set()
    do_something(x for x in seq if x not in seen or seen.add(x))

    Arguments:
    seq -- sequence
    preserve_order -- if not set the order will be arbitrary
                      Using this option will incur a speed penalty.
                      (default: False)

    Example showing order preservation:

    >>> uniquify(['a', 'aa', 'b', 'b', 'ccc', 'ccc', 'd'], preserve_order=True)
    ['a', 'aa', 'b', 'ccc', 'd']

    Example using a sequence of un-hashable items:

    >>> uniquify([['z'], ['x'], ['y'], ['z']], preserve_order=True)
    [['z'], ['x'], ['y']]

    The sorted output or the non-order-preserving approach should equal
    that of the sorted order-preserving approach output:
    
    >>> unordered = uniquify([3, 3, 1, 2], preserve_order=False)
    >>> unordered.sort()
    >>> ordered = uniquify([3, 3, 1, 2], preserve_order=True)
    >>> ordered.sort()
    >>> ordered
    [1, 2, 3]
    >>> int(ordered == unordered)
    1

    """
    try:
        # Attempt fast algorithm.
        d = {}
        if preserve_order:
            # This is based on Dave Kirby's method (f8) noted in the post:
            # http://www.peterbe.com/plog/uniqifiers-benchmark
            return [x for x in seq if (x not in d) and not d.__setitem__(x, 0)]
        else:
            for x in seq:
                d[x] = 0
            return d.keys()
    except TypeError:
        # Have an unhashable object, so use slow algorithm.
        result = []
        app = result.append
        for x in seq:
            if x not in result:
                app(x)
        return result

# Alias to noun form for backward compatibility.
unique = uniquify


def reverse_dict(d):
    """Reverse a dictionary so the items become the keys and vice-versa.

    Note: The results will be arbitrary if the items are not unique.

    >>> d = reverse_dict({'a': 1, 'b': 2})
    >>> d_items = d.items()
    >>> d_items.sort()
    >>> d_items
    [(1, 'a'), (2, 'b')]

    """
    result = {}
    for key, value in d.items():
        result[value] = key
    return result

    
def lsb(x, n):
    """Return the n least significant bits of x.

    >>> lsb(13, 3)
    5

    """
    return x & ((2 ** n) - 1)


def gray_encode(i):
    """Gray encode the given integer."""

    return i ^ (i >> 1)


def random_vec(bits, max_value=None):
    """Generate a random binary vector of length bits and given max value."""

    vector = ""
    for _ in range(int(bits / 10) + 1):
        i = int((2**10) * random.random())
        vector += int2bin(i, 10)

    if max_value and (max_value < 2 ** bits - 1):
        vector = int2bin((int(vector, 2) / (2 ** bits - 1)) * max_value, bits)
    
    return vector[0:bits]


def binary_range(bits):
    """Return a list of all possible binary numbers in order with width=bits. 
    
    It would be nice to extend it to match the
    functionality of python's range() built-in function.
    
    """
    l = []
    v = ['0'] * bits

    toggle = [1] + [0] * bits
    
    while toggle[bits] != 1:
        v_copy = v[:]
        v_copy.reverse()
        l.append(''.join(v_copy))
        
        toggle = [1] + [0]*bits
        i = 0
        while i < bits and toggle[i] == 1:
            if toggle[i]:
                if v[i] == '0':
                    v[i] = '1'
                    toggle[i+1] = 0
                else:
                    v[i] = '0'
                    toggle[i+1] = 1
            i += 1
    return l


def float_range(start, stop=None, step=None):
    """Return a list containing an arithmetic progression of floats.

    Return a list of floats between 0.0 (or start) and stop with an
    increment of step. 

    This is in functionality to python's range() built-in function 
    but can accept float increments.

    As with range(), stop is omitted from the list.

    """
    if stop is None:
        stop = float(start)
        start = 0.0

    if step is None:
        step = 1.0

    cur = float(start)
    l = []
    while cur < stop:
        l.append(cur)
        cur += step

    return l


def find_common_fixes(s1, s2):
    """Find common (prefix, suffix) of two strings.

    >>> find_common_fixes('abc', 'def')
    ('', '')

    >>> find_common_fixes('abcelephantdef', 'abccowdef')
    ('abc', 'def')

    >>> find_common_fixes('abcelephantdef', 'abccow')
    ('abc', '')

    >>> find_common_fixes('elephantdef', 'abccowdef')
    ('', 'def')

    """
    prefix = []
    suffix = []

    i = 0
    common_len = min(len(s1), len(s2))
    while i < common_len:
        if s1[i] != s2[i]:
            break

        prefix.append(s1[i])
        i += 1

    i = 1
    while i < (common_len + 1):
        if s1[-i] != s2[-i]:
            break
        
        suffix.append(s1[-i])
        i += 1

    suffix.reverse()

    prefix = ''.join(prefix)
    suffix = ''.join(suffix)
        
    return (prefix, suffix)


def is_rotated(seq1, seq2):
    """Return true if the first sequence is a rotation of the second sequence.

    >>> seq1 = ['A', 'B', 'C', 'D']
    >>> seq2 = ['C', 'D', 'A', 'B']
    >>> int(is_rotated(seq1, seq2))
    1

    >>> seq2 = ['C', 'D', 'B', 'A']
    >>> int(is_rotated(seq1, seq2))
    0

    >>> seq1 = ['A', 'B', 'C', 'A']
    >>> seq2 = ['A', 'A', 'B', 'C']
    >>> int(is_rotated(seq1, seq2))
    1

    >>> seq2 = ['A', 'B', 'C', 'A']
    >>> int(is_rotated(seq1, seq2))
    1

    >>> seq2 = ['A', 'A', 'C', 'B']
    >>> int(is_rotated(seq1, seq2))
    0

    """
    # Do a sanity check.
    if len(seq1) != len(seq2):
        return False
    # Look for occurrences of second sequence head item in first sequence.
    start_indexes = []
    head_item = seq2[0]
    for index1 in range(len(seq1)):
        if seq1[index1] == head_item:
            start_indexes.append(index1)
    # Check that wrapped sequence matches.
    double_seq1 = seq1 + seq1
    for index1 in start_indexes:
        if double_seq1[index1:index1+len(seq1)] == seq2:
            return True
    return False

def getmodule(obj):
    """Return the module that contains the object definition of obj.

    Note: Use inspect.getmodule instead.

    Arguments:
    obj -- python obj, generally a class or a function

    Examples:
    
    A function:
    >>> module = getmodule(random.choice)
    >>> module.__name__
    'random'
    >>> module is random
    1

    A class:
    >>> module = getmodule(random.Random)
    >>> module.__name__
    'random'
    >>> module is random
    1

    A class inheriting from a class in another module:
    (note: The inheriting class must define at least one function.)
    >>> class MyRandom(random.Random):
    ...     def play(self):
    ...         pass
    >>> module = getmodule(MyRandom)
    >>> if __name__ == '__main__':
    ...     name = 'rad_util'
    ... else:
    ...     name = module.__name__
    >>> name
    'rad_util'
    >>> module is sys.modules[__name__]
    1

    Discussion:
    This approach is slightly hackish, and won't work in various situations.
    However, this was the approach recommended by GvR, so it's as good as
    you'll get.

    See GvR's post in this thread:
    http://groups.google.com.au/group/comp.lang.python/browse_thread/thread/966a7bdee07e3b34/c3cab3f41ea84236?lnk=st&q=python+determine+class+module&rnum=4&hl=en#c3cab3f41ea84236
    
    """
    if hasattr(obj, 'func_globals'):
        func = obj
    else:
        # Handle classes.
        func = None
        for item in obj.__dict__.values():
            if hasattr(item, 'func_globals'):
                func = item
                break
        if func is None:
            raise ValueError("No functions attached to object: %r" % obj)
    module_name = func.func_globals['__name__']
    # Get module.
    module = sys.modules[module_name]
    return module


def round_grid(value, grid, mode=0):
    """Round off the given value to the given grid size.

    Arguments:
    value -- value to be roudne
    grid -- result must be a multiple of this
    mode -- 0 nearest, 1 up, -1 down

    Examples:
    
    >>> round_grid(7.5, 5)
    10

    >>> round_grid(7.5, 5, mode=-1)
    5

    >>> round_grid(7.3, 5, mode=1)
    10

    >>> round_grid(7.3, 5.0, mode=1)
    10.0

    """
    off_grid = value % grid
    if mode == 0:
        add_one = int(off_grid >= (grid / 2.0))
    elif mode == 1 and off_grid:
        add_one = 1
    elif mode == -1 and off_grid:
        add_one = 0
    result = ((int(value / grid) + add_one) * grid)
    return result


def get_args(argv):
    """Store command-line args in a dictionary.
    
    -, -- prefixes are removed
    Items not prefixed with - or -- are stored as a list, indexed by 'args'

    For options that take a value use --option=value

    Consider using optparse or getopt (in Python standard library) instead.

    """
    d = {}
    args = []
    
    for arg in argv:
            
        if arg.startswith('-'):
            parts = re.sub(r'^-+', '', arg).split('=')
            if len(parts) == 2:
                d[parts[0]] = parts[1]
            else:
                d[parts[0]] = None
        else:
            args.append(arg)

    d['args'] = args
    
    return d


if __name__ == '__main__':
    import doctest
    doctest.testmod(sys.modules['__main__'])

