# APPNet（code for An Adaptive Post-Processing Network With Global-Local Aggregation For semantic segmentation）
The code is being sorted and will be uploaded later.
## Introduction
  In this paper, we propose an adaptive post-processing network (APPNet) for semantic segmentation based on the predictions of current methods in the global image and local image patches. The key point of APPNet is the global-local aggregation module, which models the context between global
predictions and local predictions to generate accurate pixel-wise representation. Furthermore, we develop an adaptive points
replacement module to compensate for the lack of fine detail in global prediction and the overconfidence in local predictions.
<img src="https://github.com/zhu-gl-ux/APPNet/blob/main/image/pipeline.png" />
## Results
Our method can be readily integrated into existing segmentation methods (i.e., ConvNeXt, HRNet, ViT-Adapter) with little memory and without extra modification in current models. We empirically demonstrate our method brings performance improvements across diverse datasets (i.e., Cityscapes, ADE20K, PASCAL-Context, COCO-Stuff).
<img src="https://github.com/zhu-gl-ux/APPNet/blob/main/image/results.png" />
## Visualization
<img src="https://github.com/zhu-gl-ux/APPNet/blob/main/image/cityscapes.png" />
## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)
    
[2] Object-Contextual Representations for Semantic Segmentation. Yuhui Yuan, Xilin Chen, Jingdong Wang. [download](https://arxiv.org/pdf/1909.11065.pdf)

