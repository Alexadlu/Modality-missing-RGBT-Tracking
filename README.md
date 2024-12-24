#  [Modality-missing RGBT Tracking: Invertible Prompt Learning and High-quality Benchmarks](https://rdcu.be/d2Nx7)

Current RGBT tracking research relies on the complete multi-modality input, but modal information might miss due to some factors such as thermal sensor self-calibration and data transmission error, called modality-missing challenge in this work. To address this challenge, we propose a novel invertible prompt learning approach, which integrates the content-preserving prompts into a well-trained tracking model to adapt to various modality-missing scenarios, for robust RGBT tracking. Given one modality-missing scenario, we propose to utilize the available modality to generate the prompt of the missing modality to adapt to RGBT tracking model. However, the cross-modality gap between available and missing modalities usually causes semantic distortion and information loss in prompt generation. To handle this issue, we design the invertible prompter by incorporating the full reconstruction of the input available modality from the generated prompt. To provide a comprehensive evaluation platform, we construct several high-quality benchmark datasets, in which various modality-missing scenarios are considered to simulate real-world challenges. Extensive experiments on three modality-missing benchmark datasets show that our method achieves significant performance improvements compared with state-of-the-art methods.


## Invertible Prompt Learning Model
### News :sparkles:  IPL code is released

<img src="./imgs/image.png" alt="Miss patterns" width="80%" height="auto">

[Checkpoint Link](https://drive.google.com/file/d/1JW2uj7l5wmfIgAbANaJMYZYxqXAh443d/view?usp=drive_link)

## Modality-missing Dataset 
### News :sparkles:  Modality-missing dataset is released

<img src="./imgs/modality_miss_pattern_fig1.drawio_00.png" alt="Miss patterns" width="80%" height="auto">

<img src="./imgs/miss_pattersn.drawio.pdf.drawio_00.png" alt="Miss patterns" width="80%" height="auto">


### Citation

Please kindly cite this paper in your publications if it helps your research:
```
@article{Lu2024IPL, 
title = {Modality-missing RGBT Tracking: Invertible Prompt Learning and High-quality Benchmarks}, 
author = {Lu, Andong and Li, Chenglong and Zhao, Jiacong and Tang, Jin and Luo, Bin}, 
journal = {International Journal of Computer Vision}, 
year = {2024}, 
issn = {0920-5691}, 
doi = {10.1007/s11263-024-02311-4},
pages = {1--21}
}
```
Contact: adlu_ah@foxmail.com

## Acknowledgment
This repo is based on [OSTrack](https://github.com/botaoye/OSTrack), helps us to quickly implement our ideas.

