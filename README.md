###################################### vistra_SRResNet ############################################
###################################################################################################
#### This code was developed by Mariana Afonso, Phd student @ University of Bristol, UK, 2018 #####
################################## All rights reserved © ##########################################

Instructions:

- Install Pycharm: there is a free version. Good for debugging.
- Install python 3.5 (I have 3.5.2) or 3.6 + libraries + Tensorflow gpu (I have 1.8.0)
- Python dependencies: Tensorlayer, numpy, scipy, matplotlib, sklearn and imageio 

Quick uses (examples):


- Run YUV model evaluation

--mode=evaluate
--evalHR=
--evalLR=
--testModel=0
--ratio=1
--nlayers=16
--GAN=0
--nframes=1
--eval_inputType=YUV
--readBatch_flag=1
--inputFormat=RGB
