import os, sys
sys.path.append('../')
import numpy as np
import cv2 as cv
import pygame
import math
from util import readPFM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rgb', default= None, help="path to RGB image")
parser.add_argument('--depth', default=None, help="path to depth file")
args = parser.parse_args()

def get_rotation_matrix(x , y , z):
    '''
    parameters x, y, z represent the rotation angle
    along x-axis, y-axis, z-axis respectively 
    '''
    matrix_x = np.array([
        [1,           0,           0],
        [0, math.cos(x),-math.sin(x)],
        [0, math.sin(x), math.cos(x)]
    ])
    matrix_y = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0,           1,           0],
        [-math.sin(y),0, math.cos(y)]
    ])
    matrix_z = np.array([
        [math.cos(z),-math.sin(z), 0],
        [math.sin(z), math.cos(z), 0],
        [0,        0,              1]
    ])

    matrix = np.dot(matrix_x, matrix_y)
    matrix = np.dot(matrix, matrix_z)
    return matrix

class PointCloud:
    def __init__(self, rgb_image_path = None, depth_image_path = None):
        self.rgb_image = cv.imread(rgb_image_path)
        self.shape = self.rgb_image.shape
        if depth_image_path.endswith('pfm'):
            self.depth = readPFM(depth_image_path)
        elif depth_image_path.endswith('npy'):
            self.depth = np.load(depth_image_path)
        self.vertices = []
        self.colors = []
        self.process_data()

    def process_data(self):
        height, width, _ = self.shape
        for w in range(width):
            for h in range(height):
                vertex = (w-width//2, h-height//2, self.depth[h, w]*(-4))
                color = self.rgb_image[h, w]
                self.vertices.append(vertex)
                self.colors.append(color)

        self.vertices = np.array(self.vertices)

    def rotate(self, x, y ,z):
        matrix = get_rotation_matrix(x, y, z)
        self.vertices = self.vertices.dot(matrix)

    def get_vertices(self):
        return self.vertices

    def get_colors(self):
        return self.colors

def main():
    points = PointCloud(args.rgb, args.depth)
    pygame.init()
    win_width, win_height = 1500, 500
    center_pos = (win_width//2, win_height//2)
    screen = pygame.display.set_mode((win_width, win_height))
    clock = pygame.time.Clock()

    while True:
        #handlle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

        screen.fill((255, 255, 255))
        points.rotate(0, 0.3, 0)
        vertices = points.get_vertices()
        colors   = points.get_colors()
        for i in range(len(vertices)):
            x, y, z = vertices[i]
            #z += 50
            #f = 10 / z
            f =  1
            x, y = int(center_pos[0] + x*f), int(center_pos[1] + y*f)
            pygame.draw.circle(screen, colors[i], (x, y), 1)

        pygame.display.flip()
        
if __name__ == '__main__':
    print('PyGame testing ...')
    main()
