import sys
sys.path.append('../')
import json
import cv2 as cv
import numpy as np
from util import readPFM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rgb', default= None, help='Path to RGB image')
parser.add_argument('--depth', default= None, help='Path to depth file')
parser.add_argument('--save', default= 'model.json', help= 'Path to saved json')
parser.add_argument('--load', default=None, help='file to be loaded for checking')
parser.add_argument('--mode', default= 'create', help='create/test')
args = parser.parse_args()

def normalize_vertices(vertices, width, height):
    vertices = np.array(vertices).reshape(-1, 3)
    
    depth = vertices[:, 2]
    scale = 5
    vertices[:, 0] = (vertices[:, 0] - width//2) * scale/ width
    vertices[:, 1] = (vertices[:, 1] -height//2) * scale/ height
    vertices[:, 2] = (depth - depth.mean()) * scale/(depth.max()- depth.min())
    vertices = vertices.reshape(-1, )
    return list(vertices)

def normalize_colors(colors):
    colors = np.array(colors)
    colors = (colors / 200).round(5)
    return list(colors)

def create():
    rgb_img = cv.imread(args.rgb)
    height, width, _ = rgb_img.shape
    if args.depth.endswith('pfm'):
        depth = readPFM(args.depth)
    elif args.depth.endswith('npy'):
        depth = np.load(args.depth)
    
    ## create model 
    model = {}
    vertices = []
    colors = []
    for w in range(width):
        for h in range(height):
            x, y, z = -w, -h, (depth[h, w]*(-1))
            r, g, b = rgb_img[h, w]
            vertices.extend([x, y, z])
            colors.extend([r, g, b])
            
    vertices = [float(i) for i in vertices]
    vertices = normalize_vertices(vertices, width, height)
    colors   = [int(i) for i in colors]
    colors   = normalize_colors(colors)

    model['vertexPositions'] = vertices
    model['vertexFrontcolors'] = colors
    model['vertexBackcolors']  = colors

    ## dump to file
    file = open(args.save, 'w')
    content = json.dumps(model)
    file.write(content)
    file.close()
    print('Saved to file:', args.save)

def test():
    file = open(args.load)
    model = json.loads(file.read())
    print(model['vertexPositions'][:10])

if __name__ == '__main__':
    print('craeting Json file ...')
    if args.mode == 'create':
        create()
    else:
        test()
