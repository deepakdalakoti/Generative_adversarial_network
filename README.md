The PIESRGAN is adapted based on ESRGAN to reconstruct high resolution reactive turbulence field from low resolution(LES for example) data.

To run the existing PIESRGAN.py, make sure that 
1. GPU is available on your cluster. 
2. tensorflow-gpu is installed. 
3. enable environment: cuda/90, cudnn/7.0.4, python/3.6.0
   you could change the environment settings from the run.sub batch file 


>>> change datapath_train, datapath_validation, datapath_test in the __main__ to the path that your test data are located.  

>>> To train the PIESRGAN, uncomment the train_generator and train_piesrgan from __main__
    comment the test()
>>> If networks from previous trainings are available, load the saved generator weights (already given in __main__)
    and uncomment the test()
>>> The weights are to be found in /data/weights
>>> Change the #batch_size, epochs, print_frequency etc.. in the parameter list
>>> After the steps above are done, run the script with 
    >$ python3 PIESRGAN.py
    oder directly submit the run.sub file to the slurm batch system

