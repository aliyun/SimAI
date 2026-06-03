# topsort - dependency (topological) sorting and cycle finding functions
# Copyright (C) 2007 RADLogic
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; 
# version 2.1 of the License.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# See http://www.fsf.org/licensing/licenses/lgpl.txt for full license text.
"""Provide toplogical sorting (i.e. dependency sorting) functions.

The topsort function is based on code posted on Usenet by Tim Peters.

Modifications:
- added doctests
- changed some bits to use current Python idioms
  (listcomp instead of filter, +=/-=, inherit from Exception)
- added a topsort_levels version that ports items in each dependency level
  into a sub-list
- added find_cycles to aid in cycle debugging

Run this module directly to run the doctests (unittests).
Make sure they all pass before checking in any modifications.

Requires Python >= 2.2
(For Python 2.2 also requires separate sets.py module)

This requires the rad_util.py module.

"""

# Provide support for Python 2.2*
from __future__ import generators

__version__ = '$Revision: 0.9 $'
__date__ = '$Date: 2007/03/27 04:15:26 $'
__credits__ = '''Tim Peters -- original topsort code
Tim Wegener -- doctesting, updating to current idioms, topsort_levels,
               find_cycles
'''

# Make Python 2.3 sets look like Python 2.4 sets.
try:
    set
except NameError:
    from sets import Set as set

from rad_util import is_rotated


class CycleError(Exception):
    """Cycle Error"""
    pass


def topsort(pairlist):
    """Topologically sort a list of (parent, child) pairs.

    Return a list of the elements in dependency order (parent to child order).

    >>> print topsort( [(1,2), (3,4), (5,6), (1,3), (1,5), (1,6), (2,5)] ) 
    [1, 2, 3, 5, 4, 6]

    >>> print topsort( [(1,2), (1,3), (2,4), (3,4), (5,6), (4,5)] )
    [1, 2, 3, 4, 5, 6]

    >>> print topsort( [(1,2), (2,3), (3,2)] )
    Traceback (most recent call last):
    CycleError: ([1], {2: 1, 3: 1}, {2: [3], 3: [2]})
    
    """
    num_parents = {}  # element -> # of predecessors 
    children = {}  # element -> list of successors 
    for parent, child in pairlist: 
        # Make sure every element is a key in num_parents.
        if not num_parents.has_key( parent ): 
            num_parents[parent] = 0 
        if not num_parents.has_key( child ): 
            num_parents[child] = 0 

        # Since child has a parent, increment child's num_parents count.
        num_parents[child] += 1

        # ... and parent gains a child.
        children.setdefault(parent, []).append(child)

    # Suck up everything without a parent.
    answer = [x for x in num_parents.keys() if num_parents[x] == 0]

    # For everything in answer, knock down the parent count on its children.
    # Note that answer grows *in* the loop.
    for parent in answer: 
        del num_parents[parent]
        if children.has_key( parent ): 
            for child in children[parent]: 
                num_parents[child] -= 1
                if num_parents[child] == 0: 
                    answer.append( child ) 
            # Following "del" isn't needed; just makes 
            # CycleError details easier to grasp.
            del children[parent]

    if num_parents: 
        # Everything in num_parents has at least one child -> 
        # there's a cycle.
        raise CycleError(answer, num_parents, children)
    return answer 

def topsort_levels(pairlist):
    """Topologically sort a list of (parent, child) pairs into depth levels.

    This returns a generator. 
    Turn this into a an iterator using the iter built-in function.
    (if you iterate over the iterator, each element gets generated when
    it is asked for, rather than generating the whole list up-front.)

    Each generated element is a list of items at that dependency level.

    >>> dependency_pairs = [(1,2), (3,4), (5,6), (1,3), (1,5), (1,6), (2,5)]
    >>> for level in iter(topsort_levels( dependency_pairs )):
    ...    print level
    [1]
    [2, 3]
    [4, 5]
    [6]

    >>> dependency_pairs = [(1,2), (1,3), (2,4), (3,4), (5,6), (4,5)]
    >>> for level in iter(topsort_levels( dependency_pairs )):
    ...    print level
    [1]
    [2, 3]
    [4]
    [5]
    [6]

    >>> dependency_pairs = [(1,2), (2,3), (3,4), (4, 3)]
    >>> try:
    ...     for level in iter(topsort_levels( dependency_pairs )):
    ...         print level
    ... except CycleError, exc:
    ...     print 'CycleError:', exc
    [1]
    [2]
    CycleError: ({3: 1, 4: 1}, {3: [4], 4: [3]})


    The cycle error should look like.
    CycleError: ({3: 1, 4: 1}, {3: [4], 4: [3]})
    # todo: Make the doctest more robust (i.e. handle arbitrary dict order).

    """
    num_parents = {}  # element -> # of predecessors 
    children = {}  # element -> list of successors 
    for parent, child in pairlist: 
        # Make sure every element is a key in num_parents.
        if not num_parents.has_key( parent ): 
            num_parents[parent] = 0 
        if not num_parents.has_key( child ): 
            num_parents[child] = 0 

        # Since child has a parent, increment child's num_parents count.
        num_parents[child] += 1

        # ... and parent gains a child.
        children.setdefault(parent, []).append(child)

    return topsort_levels_core(num_parents, children)

def topsort_levels_core(num_parents, children):
    """Topologically sort a bunch of interdependent items based on dependency.

    This returns a generator.
    Turn this into a an iterator using the iter built-in function.
    (if you iterate over the iterator, each element gets generated when
    it is asked for, rather than generating the whole list up-front.)

    Each generated element is a list of items at that dependency level.

    >>> list(topsort_levels_core(
    ...          {1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2},
    ...          {1: [2, 3, 5, 6], 2: [5], 3: [4], 4: [], 5: [6]}))
    [[1], [2, 3], [4, 5], [6]]

    >>> list(topsort_levels_core(
    ...          {1: 0, 2: 2, 3: 1},
    ...          {1: [2], 2: [3], 3: [2]}))
    Traceback (most recent call last):
    CycleError: ({2: 1, 3: 1}, {2: [3], 3: [2]})

    This function has a more complicated interface than topsort_levels,
    but is useful if the data is easier to generate in this form.

    Arguments:
    num_parents -- key: item, value: number of parents (predecessors)
    children -- key: item, value: list of children (successors)

    """
    while 1:
        # Suck up everything without a predecessor.
        level_parents = [x for x in num_parents.keys() if num_parents[x] == 0]

        if not level_parents:
            break

        # Offer the next generated item,
        # which is a list of the items at this dependency level.
        yield level_parents

        # For everything item in this level,
        # decrement the parent count,
        # since we have accounted for its parent.
        for level_parent in level_parents:

            del num_parents[level_parent]

            if children.has_key(level_parent):
                for level_parent_child in children[level_parent]:
                    num_parents[level_parent_child] -= 1
                del children[level_parent]
        
    if num_parents: 
        # Everything in num_parents has at least one child -> 
        # there's a cycle.
        raise CycleError(num_parents, children)
    else:
        # This is the end of the generator.
        raise StopIteration


def find_cycles(parent_children):
    """Yield cycles. Each result is a list of items comprising a cycle.

    Use a 'stack' based approach to find all the cycles.
    This is a generator, so yields each cycle as it finds it.

    It is implicit that the last item in each cycle list is a parent of the
    first item (thereby forming a cycle).

    Arguments:
    parent_children -- parent -> collection of children

    Simplest cycle:
    >>> cycles = list(find_cycles({'A': ['B'], 'B': ['A']}))
    >>> len(cycles)
    1
    >>> cycle = cycles[0]
    >>> cycle.sort()
    >>> print cycle
    ['A', 'B']

    Simplest cycle with extra baggage at the start and the end:
    >>> cycles = list(find_cycles(parent_children={'A': ['B'],
    ...                                            'B': ['C'],
    ...                                            'C': ['B', 'D'],
    ...                                            'D': [],
    ...                                            }))
    >>> len(cycles)
    1
    >>> cycle = cycles[0]
    >>> cycle.sort()
    >>> print cycle
    ['B', 'C']

    Double cycle:
    >>> cycles = list(find_cycles(parent_children={'A': ['B'],
    ...                                            'B': ['C1', 'C2'],
    ...                                            'C1': ['D1'],
    ...                                            'D1': ['E1'],
    ...                                            'E1': ['D1'],
    ...                                            'C2': ['D2'],
    ...                                            'D2': ['E2'],
    ...                                            'E2': ['D2'],
    ...                                            }))
    >>> len(cycles)
    2
    >>> for cycle in cycles:
    ...     cycle.sort()
    >>> cycles.sort()
    >>> cycle1 = cycles[0]
    >>> cycle1.sort()
    >>> print cycle1
    ['D1', 'E1']
    >>> cycle2 = cycles[1]
    >>> cycle2.sort()
    >>> print cycle2
    ['D2', 'E2']

    Simple cycle with children not specified for one item:
    # todo: Should this barf instead?
    >>> cycles = list(find_cycles(parent_children={'A': ['B'],
    ...                                            'B': ['A'],
    ...                                            'C': ['D']}))
    >>> len(cycles)
    1
    >>> cycle = cycles[0]
    >>> cycle.sort()
    >>> print cycle
    ['A', 'B']

    Diamond cycle
    >>> cycles = list(find_cycles(parent_children={'A': ['B1', 'B2'],
    ...                                            'B1': ['C'],
    ...                                            'B2': ['C'],
    ...                                            'C': ['A', 'B1']}))
    >>> len(cycles)
    3
    >>> sorted_cycles = []
    >>> for cycle in cycles:
    ...     cycle = list(cycle)
    ...     cycle.sort()
    ...     sorted_cycles.append(cycle)
    >>> sorted_cycles.sort()
    >>> for cycle in sorted_cycles:
    ...     print cycle
    ['A', 'B1', 'C']
    ['A', 'B2', 'C']
    ['B1', 'C']

    Hairy case (order can matter if something is wrong):
    (Note order of B and C in the list.)
    >>> cycles = list(find_cycles(parent_children={
    ...                                           'TD': ['DD'],
    ...                                           'TC': ['DC'],
    ...                                           'DC': ['DQ'],
    ...                                           'C': ['DQ'],
    ...                                           'DQ': ['IA', 'TO'],
    ...                                           'IA': ['A'],
    ...                                           'A': ['B', 'C'],
    ...                                           }))
    >>> len(cycles)
    1
    >>> cycle = cycles[0]
    >>> cycle.sort()
    >>> print cycle
    ['A', 'C', 'DQ', 'IA']

    """
    cycles = []
    visited_nodes = set()

    for parent in parent_children:
        if parent in visited_nodes:
            # This node is part of a path that has already been traversed.
            continue

        paths = [[parent]]
        while paths:
            path = paths.pop()

            parent = path[-1]
            
            try:
                children = parent_children[parent]
            except KeyError:
                continue

            for child in children:
                # Keeping a set of the path nodes, for O(1) lookups at the
                # expense of more memory and complexity, actually makes speed
                # worse. (Due to construction of sets.)
                # This is O(N).
                if child in path:
                    # This is a cycle.
                    cycle = path[path.index(child):]
                    # Check that this is not a dup cycle.
                    is_dup = False
                    for other_cycle in cycles:
                        if is_rotated(other_cycle, cycle):
                            is_dup = True
                            break
                    if not is_dup:
                        cycles.append(cycle)
                        yield cycle
                else:
                    # Push this new path onto the 'stack'.
                    # This is probably the most expensive part of the algorithm
                    # (a list copy).
                    paths.append(path + [child])
                    # Mark the node as visited.
                    visited_nodes.add(child)


if __name__ == '__main__':
    # Run the doctest tests.
    import sys
    import doctest
    doctest.testmod(sys.modules['__main__'])
