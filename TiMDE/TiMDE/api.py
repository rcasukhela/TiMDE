from operator import is_
from django.http import JsonResponse
from django.http import HttpResponse
from django.shortcuts import render

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# Modules that are used to run the pipeline scripts.
import os
import subprocess
# These will have to be replaced with the necessary processing functions for the
# pipeline.
from scripts.denoise_img import denoise
from scripts.img_to_tabular import tabulate
from scripts.img_pipeline_predict import predict

'''
from scripts.denoise_img import DenoiseImages
denoiser = DenoiseImages(SCALE_BAR_HEIGHT, img_path, denoised_img)
from scripts.img_to_tabular import tabulate <-- logic is there but this 
    function needs to be made.
from scripts.img_pipeline_predict import predict <-- logic is there but this 
    function needs to be made.

The reaosn that the functions don't exist yet for some parts of the pipeline is that,
for whatever reason, the pipeline on the supercomputing cluster, doesn't like functions
and classes too much, so I had to put the logic into script form as much as possible
for it to work.

The REST compliant application just needs to take that logic and wrap it in functions.
As the functions don't return anything to the client and just save files on the system,
this should be an easy task. The only thing you should have to do is to make sure that
the config.py file has the paths pointing in the right direction when you set
the application up.
'''

@api_view(['GET'])
def execute_pipeline(request):
    if request.method == 'GET':
        try:
            denoise()
            tabulate()
            predict()
            return Response(status=status.HTTP_200_OK)
        except:
            return Response(status=status.HTTP_400_BAD_REQUEST)