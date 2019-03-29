from tkinter import Tk, Canvas
from ust.shapes import Quadrant, shaper
import os
import json
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fileConfig', action='store_true', default=True)
    parser.add_argument('--width', type=int, default=300, help='width')
    parser.add_argument('--height', type=int, default=300, help='height')
    parser.add_argument('--sharding', type=int, default=1, help='no. quadrants = sharding**2')
    parser.add_argument('--noImages', type=int, default=1, help='no. of images to create')
    parser.add_argument('--offset', type=int, default=0, help='shapes offset from center')
    parser.add_argument('--background', action='store_true', default=True)
    args = parser.parse_args()

    root = Tk()

    height = args.height
    width = args.width
    sharding = args.sharding
    no_of_images = args.noImages
    offset = args.offset
    background_true = args.background
    colour_list = ["red", "green", "blue", "yellow", "black", "magenta", "purple"]

    if args.fileConfig == True:
        with open('config.txt') as json_file:  
            config = json.load(json_file)

        height = config['height']
        width = config['width']
        sharding = config['sharding']
        no_of_images = config['no_of_images']
        colour_list = config['colour_list'] 
        offset = config['offset']
        background_true = config['background']

    root.geometry(f'{height}x{width}')
    
    for i in range(no_of_images):
        c = Canvas(root, height=height, width=width, bg='white')

        # shaper create the quadrants, their backgrounds and their shapes
        shaper(canvas=c, height=height, width=width, sharding=sharding, 
               colour_list=colour_list, offset=offset, adjacency=True, background=background_true)

        # Canvas stuff
        c.pack()
        c.update()

        # Save the image
        if os.path.isdir('./dataset/') == True:
            c.postscript(file=f'./dataset/{str(i).zfill(5)}.ps', colormode='color')
        else:
            try:
                os.makedirs('./dataset/')
            except OSError:
                raise OSError
            c.postscript(file=f'./dataset/{str(i).zfill(5)}.ps', colormode='color')
        
        # Destroy current canvas
        c.destroy()
