import os

def tabulate():
    hit = 0
    for root, dirs, files in os.walk('./img_data/denoised'):
        for f in files:
            hit += 1
    return hit