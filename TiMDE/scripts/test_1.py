import os

def denoise():
    hit = 0
    for root, dirs, files in os.walk('./img_data/raw'):
        for f in files:
            hit += 1
    return hit