# This file works with models, which have an object in the form of a list of lists of lists:
## This consists of model.obj = [[[1, 1, 1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]], so you can call
## obj[x][y][z] for a 1 or 0 value. For simplicity's sake, models are cubical.
## The actual model is a python object.
# It also defines a function that takes in any pixel location and returns a mapping of where 
# that pixel should point, with intensities. Intensities are based on how much 'solid material'
# there is between the pixel and the endpoint of the solid in that direction.
import copy
import itertools
import numpy as np
from operator import add
def mergeor(a, b):
    return int(bool(a or b))

"""
>>> obj = [[[1, 1, 1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]
>>> m = Model(obj, 3, 'original')
>>> m.sum()
27
>>> b = boundary(m)
>>> b.obj
[[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
>>> b2 = merge_models(add, [m, b])
>>> b2scanx, b2scanposx, b2scany, b2scanposy, b2scanz = pixel_scan(b2, 0, 0, 'x', '-'), pixel_scan(b2, 0, 0, 'x', '+'), pixel_scan(b2, 0, 0, 'y', '-'), pixel_scan(b2, 0, 0, 'y', '+'), pixel_scan(b2, 0, 0, 'z', '-')
>>> [print(model.obj) for model in [b2scanx, b2scanposx, b2scany, b2scanposy, b2scanz]]
[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[2, 2, 2], [2, 2, 2], [2, 2, 2]]]
[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
[[[0, 0, 0], [0, 0, 0], [2, 2, 2]], [[0, 0, 0], [0, 0, 0], [2, 2, 2]],
[[0, 0, 0], [0, 0, 0], [2, 2, 2]]]
[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
[[[0, 0, 2], [0, 0, 2], [0, 0, 2]], [[0, 0, 2], [0, 0, 2], [0, 0, 2]],
[[0, 0, 2], [0, 0, 2], [0, 0, 2]]]
[None, None, None, None, None]
>>> totalmerged = merge_models(mergeor, [b2scanx, b2scanposx, b2scany, b2scanposy, b2scanz])
>>> totalmerged.obj
[[[0, 0, 2], [0, 0, 2], [2, 2, 2]], [[0, 0, 2], [0, 0, 2], [2, 2, 2]],
[[2, 2, 2], [2, 2, 2], [2, 2, 2]]]
>>> weightedtotalmerged = weighted_boundary(totalmerged, 0, 0)
>>> weightedtotalmerged.sum()
0.9999999999999998
"""

class Model:
    def __init__(self, obj, size, typeof = 'original'):
        self.size = size
        self.obj = obj
        self.type = typeof # 'original', 'empty', 'boundary'
    
    def sum(self):
        """The sum of all of the values in a MODEL's OBJ"""
        return sum([sum([sum(z) for z in y]) for y in self.obj])

def empty_model(size):
    """ Returns a model object of size. """
    z = [0 for _ in range(size)]
    y = [copy.deepcopy(z) for _ in range(size)]
    x = [copy.deepcopy(y) for _ in range(size)]
    return Model(x, size, 'empty')

def empty_scan(size):
    """ Returns a scanning sheet which has yet to be filled in. """
    z = [0 for _ in range(size)]
    y = [copy.deepcopy(z) for _ in range(size)]
    return y

def boundary(model):
    """ Returns a model of 1's and 0's that is just the boundary of a surface. """
    def check_box(model, x, y, z):
        o = model.obj
        box = [o[x+1][y][z], o[x+1][y+1][z], o[x+1][y-1][z], o[x+1][y][z+1], o[x+1][y][z-1], \
        o[x+1][y+1][z+1], o[x+1][y+1][z-1], o[x+1][y-1][z+1], o[x+1][y-1][z-1], \
        o[x-1][y][z], o[x-1][y+1][z], o[x-1][y-1][z], o[x-1][y][z+1], o[x-1][y][z-1], \
        o[x-1][y+1][z+1], o[x-1][y+1][z-1], o[x-1][y-1][z+1], o[x-1][y-1][z-1], \
        o[x][y+1][z], o[x][y+1][z+1], o[x][y+1][z-1], o[x][y-1][z], o[x][y-1][z+1], o[x][y-1][z-1] ]
        return any([not(i) for i in box])
    size = model.size
    boundary = empty_model(size)
    boundary.type = 'boundary'
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if x == size-1 or y == size-1 or z == size-1 or x == 0 or y == 0 or z == 0:
                    if model.obj[x][y][z]:
                        boundary.obj[x][y][z] = 1
                elif model.obj[x][y][z]:
                    if check_box(model, x, y, z):
                        boundary.obj[x][y][z] = 1
    return boundary

def merge_models(merger, listofboundaries, outtype = 'boundary'):
    """Merges two models by the MERGER. MERGER is a function, such as add, or mergeor (acts as or, making values 1 or 0)."""
    b1 = listofboundaries[0]
    listofboundaries = listofboundaries[1:]
    size = b1.size
    merged = empty_model(size)
    merged.type = outtype
    while listofboundaries:
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    merged.obj[x][y][z] = merger(b1.obj[x][y][z], listofboundaries[0].obj[x][y][z])
        b1 = merged
        listofboundaries = listofboundaries[1:]
    return merged

def pixel_scan(boundary, px, py, axis, direction):
    """ Takes a boundary, pixel tuple location in the form of (px, y), axis (eg 'x' 'y' 'z'),
    and direction (eg '+' or '-'), returning a model with only pixels on the opposite side
    of the object(s) to the pixel."""
    assert (direction == '-' or direction == '+') and (axis == 'x' or axis == 'y' or axis == 'z') and isinstance(boundary, Model) and type(px) == int and type(py) == int, "Incorrect inputs." 
    size = boundary.size #recall that size already accounts for using range on it
    scansheet = empty_scan(size)
    outmodel = empty_model(size)
    outmodel.type = 'boundary'
    if axis == 'z': #iterate from the top down
        for z in list(reversed(range(size))):
            for x in range(size):
                for y in range(size):
                    if not(scansheet[x][y]) and boundary.obj[x][y][z]:
                        scansheet[x][y] = 1
                        outmodel.obj[x][y][z] = boundary.obj[x][y][z]
    
    elif direction == '+':
        if axis == 'x': #iterate from 0 to x
            iterator = list(range(px))
        elif axis == 'y': #iterate from 0 to y
            iterator = list(range(py))
    elif direction == '-':
        if axis == 'x': #iterate from end to x
            iterator = list(reversed(range(px+1, size)))
        elif axis == 'y': #iterate from end to y
            iterator = list(reversed(range(py+1, size)))
    
    if axis == 'x':
        for x in iterator:
            for y in range(size):
                for z in range(size):
                    if not(scansheet[y][z]) and boundary.obj[x][y][z]:
                        scansheet[y][z] = 1
                        outmodel.obj[x][y][z] = boundary.obj[x][y][z]
    elif axis == 'y':
        for y in iterator:
            for x in range(size):
                for z in range(size):
                    if not(scansheet[x][z]) and boundary.obj[x][y][z]:
                        scansheet[x][z] = 1
                        outmodel.obj[x][y][z] = boundary.obj[x][y][z]

    return outmodel

def pix_to_pt(px, py, ox, oy, oz): #p stands for pixel and o stands for object, or a point on the object
    #points range from 0 to size of the object
    return ( (px - ox)**2 + (py - oy)**2 + (oz + 1)**2 )**(1/2)

def weighted_boundary(model, px, py):
    """ Takes in a model and coordinate of a pixel, returning a new model with only the boundaries
    of the surface, opposite the pixel, are ones with values. These values are not just 1's and 0's,
    but a representation of how much solid is between the pixel and that boundary. """
    # crappy approximation currently; its just weighted based on distance.
    size = model.size
    weighted = empty_model(size)
    weighted.type = 'boundary'
    for x in range(size):
        for y in range(size):
            for z in range(size):
                a = model.obj[x][y][z]
                if a:
                    weighted.obj[x][y][z] = a * pix_to_pt(px, py, x, y, z)
    # step two: rescale everything so that all of model's object adds to 1
    total = weighted.sum()
    for x in range(size):
        for y in range(size):
            for z in range(size):
                weighted.obj[x][y][z] = weighted.obj[x][y][z]/total
    return weighted

def visualize(model, count=0, new = True):
    print("Showing yz plane, moving through x")
    size = model.size
    userin = 'first'
    while userin != 'q':
        sheet = empty_scan(size)
        for y in range(size):
            for z in range(size):
                sheet[y][z] = model.obj[count][y][z]
        grid = np.rot90(sheet)
        print(grid)
        userin = input("Forward (f), Backward (b), or x value (#), or quit (q): ")
        try:
            userin = int(userin)
            count = userin
        except:
            if userin == 'f':
                count += 1
            elif userin == 'b':
                count -= 1
        if count >= size or count < 0:
            print("Bad input, exiting visualize.")
            userin = 'q'

obj = [[[1, 1, 1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]
m = Model(obj, 3, 'original')
b = boundary(m)
b2 = merge_models(add, [m, b])
b2scanx, b2scanposx, b2scany, b2scanposy, b2scanz = pixel_scan(b2, 0, 0, 'x', '-'), pixel_scan(b2, 0, 0, 'x', '+'), pixel_scan(b2, 0, 0, 'y', '-'), pixel_scan(b2, 0, 0, 'y', '+'), pixel_scan(b2, 0, 0, 'z', '-')
totalmerged = merge_models(mergeor, [b2scanx, b2scanposx, b2scany, b2scanposy, b2scanz])