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

Please cite our papers if you use this code for your research

@article{zhang2021vistra2,
  title={ViSTRA2: Video coding using spatial resolution and effective bit depth adaptation},
  author={Zhang, Fan and Afonso, Mariana and Bull, David R},
  journal={Signal Processing: Image Communication},
  pages={116355},
  year={2021},
  publisher={Elsevier}
}

@article{afonso2018video,
  title={Video compression based on spatio-temporal resolution adaptation},
  author={Afonso, Mariana and Zhang, Fan and Bull, David R},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={29},
  number={1},
  pages={275--280},
  year={2018},
  publisher={IEEE}
}
