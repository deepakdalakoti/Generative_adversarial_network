The PIESRGAN is adapted based on ESRGAN to reconstruct high resolution reactive turbulence field from low resolution(LES for example) data.

To run the existing PIESRGAN.py, make sure that 
1. GPU is available on your cluster. 
2. tensorflow-gpu is installed. 
3. enable environment: cuda/90, cudnn/7.0.4, python/3.6.0
   you could change the environment settings from the run.sub batch file 

Currently has two branches

1) cleaner : High resolution data on the same grid
2) upsampling : Upsamle low resolution to high resolution

References:

[1] Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[2] Wang, Xintao, et al. "Esrgan: Enhanced super-resolution generative adversarial networks." Proceedings of the European conference on computer vision (ECCV) workshops. 2018.

[3] Shi, Wenzhe, et al. "Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[4] Bode, Mathis, et al. "Using physics-informed super-resolution generative adversarial networks for subgrid modeling in turbulent reactive flows." arXiv preprint arXiv:1911.11380 (2019).




