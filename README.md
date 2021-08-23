# Domain Adaptive Faster R-CNN in PyTorch 
## Updates
* Our new paper [Scale-Aware Domain Adaptive Faster R-CNN](https://link.springer.com/article/10.1007/s11263-021-01447-x) has been accepted by IJCV. The corresponding code is maintained under [sa-da-faster](https://github.com/yuhuayc/sa-da-faster).

## Introduction
This is a PyTorch implementation of 'Domain Adaptive Faster R-CNN for Object Detection in the Wild', implemented by Haoran Wang(whrzxzero@gmail.com). The original paper can be found [here](https://arxiv.org/pdf/1803.03243.pdf). This implementation is built on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) @ [e60f4ec](https://github.com/facebookresearch/maskrcnn-benchmark/tree/e60f4ec8dc50531debcfd5ae671ea167b5b7a1d9).

If you find this repository useful, please cite the oringinal paper:

```
@inproceedings{chen2018domain,
  title={Domain Adaptive Faster R-CNN for Object Detection in the Wild},
      author =     {Chen, Yuhua and Li, Wen and Sakaridis, Christos and Dai, Dengxin and Van Gool, Luc},
      booktitle =  {Computer Vision and Pattern Recognition (CVPR)},
      year =       {2018}
  }
```

and maskrnn-benchmark:

```
@misc{massa2018mrcnn,
author = {Massa, Francisco and Girshick, Ross},
title = {{maskrnn-benchmark: Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch}},
year = {2018},
howpublished = {\url{https://github.com/facebookresearch/maskrcnn-benchmark}},
note = {Accessed: [Insert date here]}
}
```
## Installation

Please follow the instruction in [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) to install and use Domain-Adaptive-Faster-RCNN-PyTorch.
## Example Usage
An example of Domain Adaptive Faster R-CNN with FPN adapting from **Cityscapes** dataset to **Foggy Cityscapes** dataset is provided:
1. Follow the example in [Detectron-DA-Faster-RCNN](https://github.com/krumo/Detectron-DA-Faster-RCNN) to download dataset and generate coco style annoation files
2. Symlink the path to the Cityscapes and Foggy Cityscapes dataset to `datasets/` as follows:
    ```bash
    # symlink the dataset
    cd ~/github/Domain-Adaptive-Faster-RCNN-PyTorch
    ln -s /<path_to_cityscapes_dataset>/ datasets/cityscapes
    ln -s /<path_to_foggy_cityscapes_dataset>/ datasets/foggy_cityscapes
    ```
3. Train the Domain Adaptive Faster R-CNN:
    ```
    python tools/train_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes.yaml"
    ```
4. Test the trained model:
    ```
    python tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes.yaml" MODEL.WEIGHT <path_to_store_weight>/model_final.pth
    ```
### Pretrained Model & Results
[Pretrained model](https://polybox.ethz.ch/index.php/s/OgkNFJHVkEscTO0) with image+instance+consistency domain adaptation on Resnet-50 bakcbone for Cityscapes->Foggy Cityscapes task is provided. For those who might be interested, the corresponding training log could be checked at [here](logs/city2foggy_r50_consistency_log.txt). The following results are all tested with Resnet-50 backbone.

|                  | image                | instsnace            | consistency          | AP@50 | 
|------------------|:--------------------:|:--------------------:|:--------------------:|:-----:|
| Faster R-CNN     |                      |                      |                      | 24.9  |
| DA Faster R-CNN  |          ✓           |                      |                      | 38.3  | 
| DA Faster R-CNN  |                      |          ✓           |                      | 38.8  |
| DA Faster R-CNN  |          ✓           |          ✓           |                      | 40.8  | 
| DA Faster R-CNN  |          ✓           |          ✓           |          ✓           | 41.0  |

## Other Implementation
[da-faster-rcnn](https://github.com/yuhuayc/da-faster-rcnn) based on Caffe. (original code by paper authors)

[Detectron-DA-Faster-RCNN](https://github.com/krumo/Detectron-DA-Faster-RCNN) based on Caffe2 and Detectron.

[sa-da-faster](https://github.com/yuhuayc/sa-da-faster) based on PyTorch and maskrcnn-benchmark.
