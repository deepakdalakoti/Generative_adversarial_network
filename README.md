The PIESRGAN is adapted based on ESRGAN to reconstruct high resolution reactive turbulence field from low resolution(LES for example) data.

To run the existing PIESRGAN.py, make sure that 
1. GPU is available on your cluster. 
2. tensorflow-gpu is installed. 
3. enable environment: cuda/90, cudnn/7.0.4, python/3.6.0
   you could change the environment settings from the run.sub batch file 

Currently has two branches

1) cleaner : High resolution data on the same grid
2) upsampling : Upsamle to 8x grid


To Do : clean code


