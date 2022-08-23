<<<<<<< HEAD
################################################################################
################################################################################
                                    TiMDE:
An ML Tool for Microstructural Descriptor Extraction of Ti-6Al-4V BSE-SEM images
################################################################################
################################################################################

Copyright 2022, Rohan Casukhela

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



################################################################################

This model consists of two neural networks.

######################################

The first model (the Denoiser model) is a CNN that was trained by Thermo-Fisher,
and can be found at this URL:

https://xtras.amira-avizo.com/xtras/bse-sem-denoiser

Further details can be found there.

I implemented the Denoiser model in such a way that it is capable of accepting
large images--if large enough, the image will be tiled and sent to the model in
these smaller tiles. The image is reconstructed and saved as one file.

######################################

The second model (the Encoder model) is an Encoder that I trained myself.
I used the Adam optimizer with no learning rate schedule. I used no regularization
or batch normalization either--just bogstandard densely connected layers, ReLUs,
and an output neuron that had linear activation.

I trained the model by putting the labels through a Box-Cox transformation,
and 0-1 normalization (so that all values were between 0 and 1). This was done
to improve model performance.

You should feed your denoised images to this model, and inverse transform your
predicted labels using the following parameters:

* minimum of unnormalized Box-Cox transformed labels: -42.19742787080911
* maximum of unnormalized Box-Cox transformed labels: -4.278194215968117
* Box-Cox transformation parameter: -4.533317367784158

This is already implemented within the training script I provided. However,
knowledge of the procedure taken to transform the labels will become important
if you decide to employ transfer learning for this pipeline.



################################################################################
################################################################################
                                Transfer Learning
################################################################################
################################################################################

I have decided to not provide any scripts for transfer learning.
It is impossible to say what modifications to the architecture,
training scheme, dataset, and so on will need to be implemented in totality.

A general word of advice, though: the Encoder model is trained for regression
tasks, meaning that it is meant purely for quantitative label extraction from
images. I would advise not using the Encoder model for classification tasks.



################################################################################
################################################################################
                                Using the API
################################################################################
################################################################################

TiMDE will have to be hosted on your hardware. Discussing the finer
details of hosting applications is out of scope for this readme, but I recommend
looking at the Python Django documentation and the Python Django REST framework
documentation up for starters. At the very least, that should allow you to run
TiMDE on a localhost server and process images on your own machine.

Place the images you want processed into the `img_data/raw` directory, and go to
the `<BASE_URL>/execute/` URL of the application. The pipeline will process images
=======
################################################################################
################################################################################
                                    TiMDE:
An ML Tool for Microstructural Descriptor Extraction of Ti-6Al-4V BSE-SEM images
################################################################################
################################################################################

Copyright 2022, Rohan Casukhela

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



################################################################################

This model consists of two neural networks.

######################################

The first model (the Denoiser model) is a CNN that was trained by Thermo-Fisher,
and can be found at this URL:

https://xtras.amira-avizo.com/xtras/bse-sem-denoiser

Further details can be found there.

I implemented the Denoiser model in such a way that it is capable of accepting
large images--if large enough, the image will be tiled and sent to the model in
these smaller tiles. The image is reconstructed and saved as one file.

######################################

The second model (the Encoder model) is an Encoder that I trained myself.
I used the Adam optimizer with no learning rate schedule. I used no regularization
or batch normalization either--just bogstandard densely connected layers, ReLUs,
and an output neuron that had linear activation.

I trained the model by putting the labels through a Box-Cox transformation,
and 0-1 normalization (so that all values were between 0 and 1). This was done
to improve model performance.

You should feed your denoised images to this model, and inverse transform your
predicted labels using the following parameters:

* minimum of unnormalized Box-Cox transformed labels: -42.19742787080911
* maximum of unnormalized Box-Cox transformed labels: -4.278194215968117
* Box-Cox transformation parameter: -4.533317367784158

This is already implemented within the training script I provided. However,
knowledge of the procedure taken to transform the labels will become important
if you decide to employ transfer learning for this pipeline.



################################################################################
################################################################################
                                Transfer Learning
################################################################################
################################################################################

I have decided to not provide any scripts for transfer learning.
It is impossible to say what modifications to the architecture,
training scheme, dataset, and so on will need to be implemented in totality.

A general word of advice, though: the Encoder model is trained for regression
tasks, meaning that it is meant purely for quantitative label extraction from
images. I would advise not using the Encoder model for classification tasks.



################################################################################
################################################################################
                                Using the API
################################################################################
################################################################################

TiMDE will have to be hosted on your hardware. Discussing the finer
details of hosting applications is out of scope for this readme, but I recommend
looking at the Python Django documentation and the Python Django REST framework
documentation up for starters. At the very least, that should allow you to run
TiMDE on a localhost server and process images on your own machine.

Place the images you want processed into the `img_data/raw` directory, and go to
the `<BASE_URL>/execute/` URL of the application. The pipeline will process images
>>>>>>> e1c233b3bf09bc9fbb6c06db3728031217eed955
automatically from there.