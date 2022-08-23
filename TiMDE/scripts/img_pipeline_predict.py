from config import encoder_model, features_path, lbl_min, lbl_max, bxcx_tr, pred_output, INFERENCE_BATCH_SIZE

from scripts.encoder_model_architecture import AE

import os
import time

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader
    
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def predict():
    #Load model.
    print(f'Loading Encoder model.')
    model = AE()
    model.load_state_dict(torch.load(encoder_model))
    model.eval()
    print(f'Loading Encoder model finish.')

    # Load features.
    print(f'Loading tabulated instances.')
    ftr = pd.concat([pd.read_csv(f) for f in features_path])
    ftr = ftr.iloc[:,1:]
    ftr = np.array(ftr, dtype=np.float32)
    ftr = torch.from_numpy(ftr)
    m = torch.nn.ConstantPad1d((1,0), 128)
    ftr = m(ftr)
    print(f'{ftr}, {ftr.shape}')
    #ftr = DataLoader(ftr, batch_size=INFERENCE_BATCH_SIZE, shuffle=True)
    print(f'Loading tabulated instances finish.')

    # Predict labels.
    print(f'Predicting alpha lath thicknesses.')
    with torch.no_grad():
        y_pred = torch.squeeze(model(ftr))

    # Transform and save labels.
    y_pred = y_pred * (lbl_max - lbl_min) + lbl_min
    y_pred = inv_boxcox(y_pred, bxcx_tr)
    print(f'Predicting alpha lath thicknesses finish.')

    print(f'Saving predictions.')
    with open(pred_output, 'wb') as f:
        np.save(f, y_pred)
    print(f'Saving predictions finish.')