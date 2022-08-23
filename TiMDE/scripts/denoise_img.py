import os
import time
import sys

import cv2

from modules import Denoiser
from config import SCALE_BAR_HEIGHT, img_path, denoised_img

class DenoiseImages():
    def __init__(self, SCALE_BAR_HEIGHT, img_path, denoised_img):
        self.model_architecture_path = os.path.join('models', 'pre_trained_unet', 'model_sem_bse.json')

        self.model_weights_path = os.path.join('models', 'pre_trained_unet', 'model_sem_bse.hdf5')

        self.denoiser = Denoiser(self.model_architecture_path, self.model_weights_path, scale_bar_height=SCALE_BAR_HEIGHT)

        self.path = img_path
        self.output_path = denoised_img

    def execute(self):
        for root, dirs, files in os.walk(self.path):
            for f in files:
                if '.tif' in f:
                    print(f'Denoising image: {f}')
                    
                    self.denoiser.denoise_image(
                        os.path.join(root, f),
                        tile_dim_x=2**8, tile_dim_y=2**8,
                    )
                    
                    print(self.denoiser.result.shape)
                    output_path_modified = os.path.join(self.output_path, f+'.png')
                    print('saving...')
                    #plt.imsave(output_path_modified, self.denoiser.result, cmap='gray')
                    cv2.imwrite(output_path_modified, self.denoiser.result)
                    time.sleep(5)
                    print(f'Finished denoising image and writing to: {output_path_modified}.\n\n')
                
                else:
                    print(f'Skipping file: {f}')

def denoise():
    denoiser = DenoiseImages(SCALE_BAR_HEIGHT, img_path, denoised_img)
    denoiser.execute()