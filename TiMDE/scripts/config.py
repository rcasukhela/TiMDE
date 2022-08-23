###############################################################################
###############################################################################
# If you need to make changes to this config file, I highly recommend modifying
# a copy!
###############################################################################
###############################################################################

import os
from pathlib import Path

# Input path to the raw images. Put your images in here and run the processing
# script!!!
img_path = os.path.abspath(os.path.join('./img_data', 'raw'))


#######################################################
# Denoiser model and related processing steps run here.
#######################################################

# Higher values will chop off more pixels from the bottom of the micrograph.
# This is meant to get rid of the scale bar at the bottom of the image for
# further analysis.
SCALE_BAR_HEIGHT = 139

# length and width of the ROI to take from the image.
SAMPLE_LENGTH = 256

# This is where you'll find your processed raw images. There should be no scale
# bar at this step.
denoised_img = os.path.abspath(os.path.join('./img_data', 'denoised'))


################################
# img_to_tabular runs from here.
################################

# This parameter is used to determine the number of images to add to the csv
# file.
imgs = os.listdir(img_path)
# Lowering the batch size may help with memory issues.
BATCH_SIZE = len(imgs)


write_feature_path = os.path.abspath(os.path.join('./img_data', 'proc_features'))
# After the img_to_tabular script runs, you should find your csv files here.
features_path = Path(os.path.abspath(os.path.join('./img_data', 'proc_features')))

# Just making a list of the csvs.
features_path = list(features_path.glob(f'ftr_{BATCH_SIZE}_*'))


######################################################
# Encoder model and related processing steps run here.
######################################################
INFERENCE_BATCH_SIZE = 128

encoder_model = os.path.abspath(os.path.join('models', 'encoder_model.pt'))

# I have added these here for reference in case other transformations need to be
# done.
lbl_min = -42.19742787080911
lbl_max = -4.278194215968117
bxcx_tr = -4.533317367784158

# Predictions output path.
pred_output = os.path.abspath(os.path.join('output_pred', 'pred.npy'))