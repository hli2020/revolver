# Few-shot Segmentation Propagation with Guided Networks

on arxiv: https://arxiv.org/abs/1806.07373

by Kate Rakelly\*, Evan Shelhamer\*, Trevor Darrell, Alexei A. Efros, and Sergey Levine
UC Berkeley

> Learning-based methods for visual segmentation have made progress on particular
types of segmentation tasks, but are limited by the necessary supervision, the
narrow definitions of fixed tasks, and the lack of control during inference for
correcting errors. To remedy the rigidity and annotation burden of standard
approaches, we address the problem of few-shot segmentation: given few image
and few pixel supervision, segment any images accordingly. We propose guided
networks, which extract a latent task representation from any amount of
supervision, and optimize our architecture end-to-end for fast, accurate
few-shot segmentation. Our method can switch tasks without further optimization
and quickly update when given more guidance. We report the first results for
segmentation from one pixel per concept and show real-time interactive video
segmentation. Our unified approach propagates pixel annotations across space
for interactive segmentation, across time for video segmentation, and across
scenes for semantic segmentation. Our guided segmentor is state-of-the-art in
accuracy for the amount of annotation and time.

This is a **work-in-progress**, and not yet a reference implementation of the paper, and could change at any time.


### Below are Hongyang's notes
- VOCSeg dataset download from VOC official website.
- SBDD dataset download 
[here](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/README.md#pascal-voc-and-sbd).



Create symlink to datasets:

    ln -s /absolute/path/to/voc_seg data/voc2012
    ln -s /absolute/path/to/sbdd data/sbdd
    
Install dependencies:
    
    conda install -c conda-forge setproctitle
    conda install -c anaconda click scipy 
    
How to run:

    # train
    python train.py your_exp_name --model fcn32s
    # test 
    python evaluate.py your_exp_name --model fcn32s
    
Repo environment:

- On work macbook, pycharm, OSX, `no-cuda`, compiler `Python 3.6 (revolver)`
- On s42, remote debug, ``cuda``