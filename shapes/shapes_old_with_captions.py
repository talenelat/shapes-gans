'''
Module for image dataset creation. Uses Tkinter.
'''

from tkinter import Tk, Canvas
import numpy as np

def create_background(background_color, c, h, w, quad):
    '''
    Color the background of quadrant (by creating a colored rectangle
    the size of the quadrant).
    '''
    if quad == 0:
        c.create_rectangle(0, 0, h/2, w/2, fill=background_color, width=0)
    if quad == 1:
        c.create_rectangle(300, 0, h/2, w/2, fill=background_color, width=0)
    if quad == 2:
        c.create_rectangle(300, 300, h/2, w/2, fill=background_color, width=0)
    if quad == 3:
        c.create_rectangle(0, 300, h/2, w/2, fill=background_color, width=0)


def create_shape(quad, c=None, shape_type=None, shape_color=None, 
                 points=None, offset=20, height=300, width=300, ):
    '''
    Create shapes (type and color) in quadrant == quad
    '''
    delta_0_x = np.random.uniform(-offset, offset)
    delta_0_y = np.random.uniform(-offset, offset)
    delta_1_x = np.random.uniform(-offset, offset)
    delta_1_y = np.random.uniform(-offset, offset)
    delta_2_x = np.random.uniform(-offset, offset)
    delta_2_y = np.random.uniform(-offset, offset)
    delta_3_x = np.random.uniform(-offset, offset)
    delta_3_y = np.random.uniform(-offset, offset)

    deltas_x = [delta_0_x, delta_1_x, delta_2_x, delta_3_x]
    deltas_y = [delta_0_y, delta_1_y, delta_2_y, delta_3_y]
    
    if shape_type == 'rectangle': # Rectangle
        c.create_rectangle(points[0] * height + deltas_x[quad], 
                           points[1] * width + deltas_y[quad], 
                           points[2] * height + deltas_x[quad], 
                           points[3] * width + deltas_y[quad], 
                           fill=shape_color, width=0)

    if shape_type == 'circle': # Circle
        c.create_oval(points[0] * height + deltas_x[quad], 
                      points[1] * width + deltas_y[quad], 
                      points[2] * height + deltas_x[quad], 
                      points[3] * width + deltas_y[quad], 
                      fill=shape_color, width=0)

    if shape_type == 'triangle': # Triangle
        if quad == 0:
            shape = c.create_polygon(0.15 * height + deltas_x[quad], 
                                     0.35 * width + deltas_y[quad], 
                                     0.25 * height + deltas_x[quad], 
                                     0.15 * width + deltas_y[quad], 
                                     0.35 * height + deltas_x[quad], 
                                     0.35 * width + deltas_y[quad], 
                                     fill=shape_color)
        if quad == 1:
            shape = c.create_polygon(0.65 * height + deltas_x[quad], 
                                     0.35 * width + deltas_y[quad], 
                                     0.75 * height + deltas_x[quad], 
                                     0.15 * width + deltas_y[quad], 
                                     0.85 * height + deltas_x[quad], 
                                     0.35 * width + deltas_y[quad], 
                                     fill=shape_color)
        if quad == 2:
            shape = c.create_polygon(0.65 * height + deltas_x[quad], 
                                     0.85 * width + deltas_y[quad], 
                                     0.75 * height + deltas_x[quad], 
                                     0.65 * width + deltas_y[quad], 
                                     0.85 * height + deltas_x[quad], 
                                     0.85 * width + deltas_y[quad], 
                                     fill=shape_color)
        if quad == 3:
            shape = c.create_polygon(0.15 * height + deltas_x[quad], 
                                     0.85 * width + deltas_y[quad], 
                                     0.25 * height + deltas_x[quad], 
                                     0.65 * width + deltas_y[quad],  
                                     0.35 * height + deltas_x[quad], 
                                     0.85 * width + deltas_y[quad], 
                                     fill=shape_color)