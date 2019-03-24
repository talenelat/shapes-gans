'''
Create GIF animations from the generative networks resulting images.
'''

import imageio 
import os

def gen_animated(output_folder):
    root_jpg = f'.\\{output_folder}'
    images = []

    for file_name in os.listdir(root_jpg):
        if file_name.endswith('.png'):
            images.append(imageio.imread(os.path.join(root_jpg, file_name)))
    imageio.mimsave(f'.\\{output_folder}\\test_1_evol.gif', images)