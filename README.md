
### Instructions:

- Install Pycharm: there is a free version. Good for debugging.
- Install python 3.5 (I have 3.5.2) or 3.6 + libraries + Tensorflow gpu (I have 1.8.0)
- Python dependencies: Tensorlayer, numpy, scipy, matplotlib, sklearn and imageio 

### Usage


Run YUV model evaluation

--mode=evaluate
--evalHR=$output yuv path$
--evalLR=$input yuv path$
--testModel=0
--ratio=1
--nlayers=16
--GAN=0
--nframes=1
--eval_inputType=YUV
--readBatch_flag=1
--inputFormat=RGB

All input files should use the standard file name, e.g. Campfire_3840x2160_30fps_10bit_qp22.yuv (filename_HxW_xxxfps_xbit_qpxx.yuv)
### Reference

Please cite our papers if you use this code for your research

[1] Zhang, Fan and Afonso, Mariana and Bull, David R, ViSTRA2: Video coding using spatial resolution and effective bit depth adaptation, Elsevier, Signal Processing: Image Communication, pp 116355, 2021.

[2] Afonso, Mariana and Zhang, Fan and Bull, David R, Video compression based on spatio-temporal resolution adaptation, IEEE Transactions on Circuits and Systems for Video Technology, vol 29, no 1, pages 275-280, 2018.

[3] Zhang, Fan, Di Ma, Chen Feng, and David R. Bull. "Video Compression With CNN-Based Postprocessing." IEEE MultiMedia, vol 28, no. 4, pages 74-83, 2021.
