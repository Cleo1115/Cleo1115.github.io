"""
597PR - Numpy Assignment
Copyright John Weible

Goals:
 * Practicing Test-Driven-Design and writing more tests
 * Manipulating multi-dimensional arrays in Numpy

Part 1: Comparing Numpy's efficiency and convenience with standard Python

Part 2: Exploring porosity of solids.
        Given a block of material with random holes in it, light shine straight through it?

        Complete the code for functions can_light_pass_top_down() and can_light_pass_through()
        so that all the Doctests work correctly.

Part 1 Details...
Take a look at the Numpy function unique() in the documentation.
   https://numpy.org/doc/stable/reference/generated/numpy.unique.html

I want you to implement a similar function that works on lists using standard Python library only.
To simplify this work, we'll ignore several of the options and features numpy.unique() has.

To help clarify which parts to implement and which to ignore, I'm giving you a completed
wrapper function called limited_numpy_unique() with working Doctests.
Your list_unique() function should behave as closely as possible to limited_numpy_unique().
"""
import numpy as np
from time import process_time_ns
from random import randint


def list_unique(a: list, return_index=False, return_counts=False):
    """
    TODO! You fill out this Docstring to complete it
    TODO! You add several more working Doctests for this function.
    """
    # TODO: Complete the code here using ONLY standard Python libraries, so
    #  that it behaves analogously to the limited_numpy_unique() function below.

    # TIPS: You might find the collections.Counter class helpful.
    # Standard dictionary and/or the set data type may also be useful.
    # If you need help figuring out how to "flatten" a nested list into
    # a simple 1-dimensional (non-nested) list, either ask for help or
    # it's okay to look online for ideas (there are many). But you must
    # cite all sources you use!
    # Your solution does not have to support nested lists deeper than 3 dimensions, but it's okay if it does.
    index_count = {}

    for i in range(100):
        if type(a[0]) != list:
            break
        else:
            a = sum(a, [])
            continue

    for ele in a:
        if not index_count.get(ele):
            index_count[ele] = [a.index(ele), 1]
        else:
            index_count.get(ele)[1] += 1
    a = sorted(set(a))
    index = [index_count.get(index)[0] for index in index_count]
    counts = [index_count.get(count)[1] for count in index_count]
    if return_index and return_counts:
        return a, index, counts
    elif return_index:
        return a, index
    elif return_counts:
        return a, counts
    else:
        return a


def limited_numpy_unique(ar: np.ndarray, return_index=False, return_counts=False):
    """An artificial wrapper around numpy.unique() intended to limit the options it supports.
    This function is already complete! You don't need to modify anything in it.
    :param ar: a Numpy ndarray. If it has more than one dimension, it gets flattened into 1D.
    :param return_index: If True, also return the indices of ar that result in the unique array.
    :param return_counts: If True, also return the number of times each unique item appears in ar.
    :returns: unique ndarray -- The sorted unique values.
              unique_indices ndarray, optional --
                The indices of the first occurrences of the unique values in the original array. Only provided if return_index is True.
              unique_counts ndarray, optional --
                The number of times each of the unique values comes up in the original array. Only provided if return_counts is True.
    >>> limited_numpy_unique([[1, 1, 2], [2, 3, 3]])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> limited_numpy_unique(a)
    array([1, 2, 3])
    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = limited_numpy_unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'], dtype='<U1')
    >>> indices
    array([0, 1, 3])
    """
    if isinstance(ar, np.ndarray):
        ar = ar.flatten()
    return np.unique(ar, return_index=return_index, return_counts=return_counts)


def can_light_pass_top_down(block: np.ndarray) -> int:
    """This is a simpler function than can_light_pass_through() which is used iteratively by it.
    It ONLY considers looking "down" along axis 0 and doesn't rotate anything.

    TIPS: Using the concepts in my lecture examples, this can be implemented in just 2-3 lines of code.
          Once this function works, it can be used by the can_light_pass_through() function to make that easier.

    :param block: The array representing a hard opaque substance with gaps in it.
    :returns: Returns the maximum count of cells that are hollow all the way through. If zero, that also means False.
    :raises: ValueError if block is not 2D or 3D.  TODO: Implement this!
    >>> a = np.array([[2, 0, 2],
    ...               [3, 0, 4],
    ...               [5, 0, 8]])
    >>> can_light_pass_top_down(a)
    1
    >>> b = np.array([[2, 0, 2],
    ...               [3, 1, 4],
    ...               [5, 0, 8]])
    >>> can_light_pass_top_down(b)
    0
    """
    # TODO: Implement this function
    count = 0
    block[np.isnan(block)] = 0
    if len(block.shape) == 2:
        for pointer in range(0, len(block[0])):
            if block[:, pointer].any() == 0:
                count += 1
    elif len(block.shape) == 3:
        for pointer1 in range(0, len(block[0])):
            for pointer2 in range(0, len(block[0][0])):
                if block[:, pointer1, pointer2].any() == 0:
                    count += 1
    else:
        raise ValueError
    return count


def can_light_pass_through(block: np.ndarray) -> int:
    """Given an array that represents a 2D or 3D hard opaque substance (with holes represented by ZEROES or NAN),
    determines whether light shined directly on any side of it would be able to penetrate all the way through it.
    This means where at least one straight line has a hole all the way through it. Assume ONLY orthogonal tunnels
    through are considered, not other angles (e.g. straight vertical, horizontal, or back-to-front). For 2D arrays,
    we only consider looking along axis 0 & 1, not the theoretical 3rd dimension (back-to-front).

    :param block: The array representing a hard opaque substance with gaps in it.
    :returns: Returns the MAXIMUM count of cells that are hollow all the way through FROM ANY SINGLE DIRECTION. If zero, also that means False.
    >>> a = np.array([[2, 0, 1,      0, 2],
    ...               [3, 0, 0,      0, 4],
    ...               [5, 0, 1, np.nan, 8]])
    >>> can_light_pass_through(a)
    2
    >>> b = np.array([[2, 4, 1],
    ...               [0, 2, 4],
    ...               [0, 0, 0],   # <--- We can see through this from the SIDE at row 2
    ...               [5, 9, 8]])
    >>> can_light_pass_through(b)
    1
    >>> c = np.array([[1, 0, 0, 1, 1, 1],
    ...               [1, 1, 0, 0, 0, 1],
    ...               [1, 1, 1, 1, 0, 1]])  # a straight DIAGONAL path is open, but we don't consider those.
    >>> can_light_pass_through(c)
    0
    >>> d = np.array([[[1, 1, 1, 1],
    ...                [1, 0, 1, 1],
    ...                [1, 1, 1, 1],
    ...                [1, 1, 1, 1]],
    ...               [[1, 1, 1, 1],
    ...                [1, 0, 1, 1],
    ...                [1, 1, 1, 1],
    ...                [1, 1, 1, 1]]])
    >>> can_light_pass_through(d)
    1
    >>> d_on_its_side = np.rot90(d, k=1, axes=(0,1))  # Turn array d on its side, check function still works.
    >>> can_light_pass_through(d_on_its_side)
    1
    >>> e = np.array([[[1, 0, 1],  # from top to bottom, 2 cells can see through.
    ...                [1, 0, 1],
    ...                [1, 1, 1],
    ...                [1, 1, 1]],
    ...               [[1, 0, 1],
    ...                [0, 0, 0],  # from the side, only 1 cell sees through.
    ...                [1, 1, 1],
    ...                [1, 1, 1]]])
    >>> can_light_pass_through(e)
    2
    """

    # TODO: Implement this by rotating the given block each direction needed, then calling can_light_pass_top_down() for each rotation.
    def turn_block(block: np.ndarray) -> np.ndarray:
        if len(block.shape) == 2:
            block[np.isnan(block)] = 0
            row = len(block)
            column = len(block[0])

            new_row = column
            new_column = row
            new_block = np.array([[0] * new_column] * new_row)
            for r in range(0, new_row):
                for c in range(0, new_column):
                    new_block[r, c] = block[c, r]
            return new_block

        if len(block.shape) == 3:
            block[np.isnan(block)] = 0
            height = len(block)
            row = len(block[0])
            column = len(block[0][0])

            new_height = column
            new_row = row
            new_column = height

            turn_right_block = np.array([[[0] * new_column] * new_row] * new_height)
            for h in range(0, new_height):
                for r in range(0, new_row):
                    for c in range(0, new_column):
                        turn_right_block[h, r, c] = block[c, r, h]

            new_height2 = row
            new_row2 = height
            new_column2 = column

            turn_up_block = np.array([[[0] * new_column2] * new_row2] * new_height2)
            for h in range(0, new_height2):
                for r in range(0, new_row2):
                    for c in range(0, new_column2):
                        turn_up_block[h, r, c] = block[r, h, c]
            return turn_up_block, turn_right_block

    if len(block.shape) == 2:
        turned_block = turn_block(block)
        count0 = can_light_pass_top_down(block)
        count1 = can_light_pass_top_down(turned_block)
        return max(count0, count1)

    if len(block.shape) == 3:
        turn_up_block, turn_right_block = turn_block(block)
        count0 = can_light_pass_top_down(block)
        count1 = can_light_pass_top_down(turn_up_block)
        count2 = can_light_pass_top_down(turn_right_block)
        return max(count0, count1, count2)


if __name__ == '__main__':
    """Leave all the code below here as I have provided it!  
    This way, besides running the Doctests for this module, you can run it
    in 'normal' mode and it will demonstrate the functions and do some 
    performance timing. We're especially interested in how much slower the
    list_unique() function is compared to limited_numpy_unique()."""

    # create a bigish array to demonstrate the functions. This has 100 Million values.
    size = (10_000, 10_000)
    print("Generating array with size: ", size)
    a = np.random.randint(low=0, high=50, size=size, dtype='int8')
    # to make the odds high of being able to shine light through it, this MIGHT randomly
    # "drill" through a few times:
    holes = randint(0, 4)
    while holes > 0:
        holes -= 1
        a[randint(0, a.shape[0]), :] = 0

    print("\nCalling can_light_pass_through()...")
    start_time = process_time_ns()
    result = can_light_pass_through(a)
    elapsed_time = process_time_ns() - start_time
    print("Light can pass through {} cells in at least one direction.\n".format(result))
    print("  execution time: {:.8f} seconds".format(elapsed_time / 1_000_000_000))

    print("\nCalling limited_numpy_unique()...")
    start_time = process_time_ns()
    uniques, positions, counts = limited_numpy_unique(a, return_index=True, return_counts=True)
    elapsed_time = process_time_ns() - start_time
    print("Unique values:\n{}\nPositions of 1st occurrences:\n{}\nCounts:\n{}\n".format(uniques, positions, counts))
    print("  execution time: {:.8f} seconds".format(elapsed_time / 1_000_000_000))

    # Convert Numpy array into a standard list:
    a = a.tolist()
    print("\nCalling list_unique()...")
    start_time = process_time_ns()
    uniques, positions, counts = list_unique(a, return_index=True, return_counts=True)
    elapsed_time = process_time_ns() - start_time
    print("Unique values:\n{}\nPositions of 1st occurrences:\n{}\nCounts:\n{}\n".format(uniques, positions, counts))
    print("  execution time: {:.8f} seconds".format(elapsed_time / 1_000_000_000))
    # Convert Numpy array into a standard list:

