'''
TODO:
# Random Colours
# More shapes

'''


from tkinter import Tk, Canvas
import numpy as np 


def shaper(canvas, 
           width=300, height=300, 
           sharding=1,
           offset=0,
           colour_list=['red', 'green', 'blue', 'yellow'], 
           shape_list=['rectangle', 'circle', 'triangle'],
           adjacency=True,
           background=False):
    
    # Iterate over the quadrants (based on the sharding level)
    for i in range(1, sharding+1):
        for j in range(1, sharding+1):
            # Create a temporary copy of the colour list
            temp_colour_list = list(colour_list)

            # Initialize Quadrant
            q = Quadrant(canvas=canvas, height=height, width=width, 
                         pos=(i, j), sharding=sharding)

            if adjacency == True and background == True:            
                # Remove colours of neighbours from the temporary colour list
                for colour in set(q.neighbours_colours):
                    temp_colour_list.remove(colour)
            
            # Background colour of the quadrant
            if background == False:
                background_colour = 'white'
            else:
                background_colour = temp_colour_list[np.random.randint(0, len(temp_colour_list))]

                # Remove the background colour from the list
                temp_colour_list.remove(background_colour)

            # Add back the colour of the neighbours
            if adjacency == True and background == True:
                temp_colour_list.extend(q.neighbours_colours)

            # Type and colour of the shape 
            shape_colour = temp_colour_list[np.random.randint(0, len(temp_colour_list))]
            shape_type = shape_list[np.random.randint(0, len(shape_list))]

            # Create the background
            q.create_background(background_colour)

            # Random offset for the shape (from its center point)
            offset_x = np.random.uniform(-offset, offset)
            offset_y = np.random.uniform(-offset, offset)

            # Create shape
            q.create_shape(shape_type, shape_colour, offset=(offset_x, offset_y))


class Quadrant():
    _quadrants_dict = {} # Store information about neighbours

    def __init__(self, canvas, width, height, 
                 pos: tuple, sharding: int):
        self.sharding = sharding
        self.position = pos # tuple: e.g. (2, 3) for 2nd row, 3rd column
        self.neighbours_colours = self._neighbours_colours() # list of colours
        self.height = height
        self.width = width
        self.canvas = canvas
        self.background_colour = None
        self.shape_colour = None
        self.shape_type = None
        self._quadrants_dict[self.position] = [self.background_colour, 
                                               self.shape_colour,
                                               self.shape_type]
        
    def create_background(self, background_colour):
        self.background_colour = background_colour
        self._quadrants_dict[self.position][0] = self.background_colour

        A = ((self.position[1] - 1)/self.sharding * self.width, 
             (self.position[0] - 1)/self.sharding * self.height)
        B = ((self.position[1])/self.sharding * self.width, 
             (self.position[0])/self.sharding * self.height)
        self.canvas.create_rectangle(A[0], A[1], B[0], B[1],
                                fill=background_colour, width=0, outline='')
        

    def create_shape(self, shape_type: str, shape_colour: str, offset=(0, 0)):
        self.shape_colour = shape_colour
        self.offset = offset
        self._quadrants_dict[self.position][2] = shape_type
        self._quadrants_dict[self.position][1] = shape_colour

        if self.shape_colour == self.background_colour:
            print(self.position, 'Background and Shape colours are the same!')
        
        if shape_type == 'rectangle':
            self._create_rectangle()
        elif shape_type == 'circle':
            self._create_circle()
        elif shape_type == 'triangle':
            self._create_triangle()
        else:
            raise Exception('shape_type must be in [\'rectangle\', \'circle\', or \'triangle\']')

    def _neighbours_colours(self, verbose=False):
        neighbours = []
        try:
            neighbours.append(self._quadrants_dict[(self.position[0], self.position[1] - 1)][0])
        except KeyError:
            if verbose == True:
                print("No left neighbour")
        try:
            neighbours.append(self._quadrants_dict[(self.position[0], self.position[1] + 1)][0])
        except KeyError:
            if verbose == True:
                print("No right neighbour")
        try:
            neighbours.append(self._quadrants_dict[(self.position[0] - 1, self.position[1])][0])
        except KeyError:
            if verbose == True:
                print("No bottom neighbour")
        try:
            neighbours.append(self._quadrants_dict[(self.position[0] + 1, self.position[1])][0])
        except KeyError:
            if verbose == True:
                print("No top neighbour")
        return neighbours

    def _create_rectangle(self):
        A = ((self.position[1] - 1) / self.sharding * self.width + 0.30 * self.width/self.sharding,
             (self.position[0] - 1) / self.sharding * self.height + 0.30 * self.height/self.sharding)
        B = ((self.position[1] - 1) / self.sharding * self.width + 0.70 * self.width/self.sharding,
             (self.position[0] - 1) / self.sharding * self.height + 0.70 * self.height/self.sharding)
        self.canvas.create_rectangle(A[0] + self.offset[0], A[1] + self.offset[1], 
                                     B[0] + self.offset[0], B[1] + self.offset[1],
                                     fill=self.shape_colour, width=0, outline='')

    def _create_circle(self):
        A = ((self.position[1] - 1) / self.sharding * self.width + 0.30 * self.width/self.sharding,
             (self.position[0] - 1) / self.sharding * self.height + 0.30 * self.height/self.sharding)
        B = ((self.position[1] - 1) / self.sharding * self.width + 0.70 * self.width/self.sharding,
             (self.position[0] - 1) / self.sharding * self.height + 0.70 * self.height/self.sharding)
        self.canvas.create_oval(A[0] + self.offset[0], A[1] + self.offset[1], 
                                B[0] + self.offset[0], B[1] + self.offset[1],
                                fill=self.shape_colour, width=0, outline='')

    def _create_triangle(self):
        A = ((self.position[1] - 1) / self.sharding * self.width + 0.30 * self.width/self.sharding,
             (self.position[0] - 1) / self.sharding * self.height + 0.70 * self.height/self.sharding)
        B = ((self.position[1] - 1) / self.sharding * self.width + 0.50 * self.width/self.sharding,
             (self.position[0] - 1) / self.sharding * self.height + 0.30 * self.height/self.sharding)
        C = ((self.position[1] - 1) / self.sharding * self.width + 0.70 * self.width/self.sharding,
             (self.position[0] - 1) / self.sharding * self.height + 0.70 * self.height/self.sharding)
        self.canvas.create_polygon(A[0] + self.offset[0], A[1] + self.offset[1],
                                   B[0] + self.offset[0], B[1] + self.offset[1],
                                   C[0] + self.offset[0], C[1] + self.offset[1],
                                   fill=self.shape_colour, outline='')
            
