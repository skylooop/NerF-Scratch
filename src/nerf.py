from absl import app, flags

import wget
import torch
import torch.nn as nn
import typing as tp
import numpy as np
import os
import tqdm as tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

FLAGS = flags.FLAGS
flags.DEFINE_string(name="output_dir", default="data", help="Provide path to saving data.")

def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file = wget.download(url="http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz", out=FLAGS.output_dir)

    data = np.load(os.path.join(FLAGS.output_dir, 'tiny_nerf_data.npz'))
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    
    print(f'Images shape: {images.shape}')
    print(f'Poses shape: {poses.shape}')
    print(f'Focal length: {focal}')

    height, width = images.shape[1:3]
    near, far = 2., 6.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    n_training = 100
    testimg_idx = np.random.randint(images.shape[0])
    testimg, testpose = images[testimg_idx], poses[testimg_idx]

    ax.imshow(testimg)
    plt.savefig("assets/sample_image.jpg")
    print(poses[0])
if __name__ == "__main__":
    app.run(main)


