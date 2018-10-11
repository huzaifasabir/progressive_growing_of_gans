#!/bin/bash
# If using Anaconda, add it to the path (comment otherwise)
export PATH=/netscratch/huzaifa/anaconda3/bin/:$PATH
# Go to the directory from where the code should be executed
cd /netscratch/huzaifa/progressive_growing_of_gans-master/
# Activate the required virtual env if using Anaconda (comment otherwise)
source activate my_env_py3
#source activate py3_env
#% Call the python scripts
echo "Training the model"
python train.py
echo "Training completed!"
#echo "Performing inference"
#python test.py densenet
#echo "Inference completed!"
#NV_GPU=7 sudo userdocker run -it -v /netscratch:/netscratch dlcc/tensorflow_opencv:17.11 /netscratch/huzaifa/progressive_growing_of_gans-master/script.sh