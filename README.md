
### Instructions:

- Install Pycharm: there is a free version. Good for debugging.
- Install python 3.5 (I have 3.5.2) or 3.6 + libraries + Tensorflow gpu (I have 1.8.0)
- Python dependencies: Tensorlayer, numpy, scipy, matplotlib, sklearn and imageio 

### Quick uses (examples):


Run YUV model evaluation

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

### Reference

Please cite our papers if you use this code for your research

[1] Zhang, Fan and Afonso, Mariana and Bull, David R, ViSTRA2: Video coding using spatial resolution and effective bit depth adaptation, Elsevier, Signal Processing: Image Communication, pp 116355, 2021.

[2] Afonso, Mariana and Zhang, Fan and Bull, David R, Video compression based on spatio-temporal resolution adaptation, IEEE Transactions on Circuits and Systems for Video Technology, vol 29, no 1, pages 275-280, 2018.
