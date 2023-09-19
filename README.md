# Demo for pose estimation
This is a demo for pose estimation.
## Requirements
```
mmcv==2.0.1
mmcv_full==1.3.18
mmdet==2.25.0
mmpose==0.28.0
moviepy==1.0.3
numpy==1.21.2
opencv_python_headless==4.2.0.34
torch==1.8.0
```
Please make sure you have installed `mmcv-full`, `mmpose` and `mmdet`. Please consult the official installation tutorial if you experience any difficulties.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
## Run
For example:
```
python demo/my_demo_skeleton.py demo/eval_6_level2_mess_mercury_0001_01_visual.mp4 demo/agent_output_1.mp4
```