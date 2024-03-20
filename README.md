SwIPE: Efficient and Robust Medical Image Segmentation with Implicit Patch Embeddings
===========
<details>
<summary>
  <b>SwIPE: Efficient and Robust Medical Image Segmentation with Implicit Patch Embeddings</b>, MICCAI 2023.
  <a href="https://conferences.miccai.org/2023/papers/635-Paper1380.html" target="blank">[MICCAI]</a>
  <a href="https://arxiv.org/abs/2307.12429" target="blank">[arXiv]</a>
  <a href="https://arxiv.org/pdf/2307.12429.pdf" target="blank">[PDF]</a>
	<br><em>
    <a href="https://charzharr.github.io/">Charley Y. Zhang</a>, 
    <a href="https://pgu-nd.github.io/">Pengfei Gu</a>, 
    <a href="https://nsapkota417.github.io/">Nishchal Sapkota</a>, 
    <a href="https://engineering.nd.edu/faculty/danny-chen/">Danny Z. Chen</a></em></br>
</summary>

```bash
@inproceedings{zhang2023swipe,
  title        = {SwIPE: Efficient and Robust Medical Image Segmentation with Implicit Patch Embeddings},
  author       = {Zhang, Charley Y. and Gu, Pengfei and Sapkota, Nishchal and Chen, Danny Z},
  booktitle    = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages        = {315--326},
  year         = {2023},
  organization = {Springer}
}
```
</details>


<details>
  <summary>
	  <b>Key Ideas & Main Findings</b>
  </summary>

  SwIPE (Segmentation with Implicit Patch Embeddings) is a medical image segmentation approach that utilizes implicit neural representations (INRs) to learn continuous representations rather than discrete ones which are commonly adopted by modern methods (e.g., CNNs, transformers, or combinations of both). 

1. **Patch-based Implicit Neural Representations (INRs)**: SwIPE is the first approach to leverage patch-based INRs for medical image segmentation. This novel methodology allows for both accurate local boundary delineation and global shape coherence while moving away from discrete raster representations.
2. **Efficieny and Robustness**: Through extensive evaluations, SwIPE outperforms state-of-the-art methods in both 2D polyp segmentation and 3D abdominal organ segmentation tasks. Notably, SwIPE achieves these results with over 10x fewer parameters, showcasing exceptional model efficiency. Additionally, SwIPE exhibits superior robustness to data shifts across image resolutions and datasets.
3. **Augmented Contextual Understanding with Multi-stage Embedding Attention (MEA) and Stochastic Patch Overreach (SPO)**: The introduction of MEA for dynamic feature extraction and SPO for enhanced boundary improve contextual understanding during the encoding step and address boundary continuities during occupancy decoding, leading to more accurate and coherent segmentation results.
</details>



## Issues
- Please open new threads or report issues to Charley's email: yzhang46@nd.edu.

## Acknowledgements, License & Usage 
- Code for [OSSNet](https://github.com/ChristophReich1996/OSS-Net)
- Code for [IOSNet](https://github.com/osamakhaan/iosnet)
- Code for [OccNet](https://github.com/autonomousvision/occupancy_networks)
- If you found our work useful in your research, please consider citing our works(s):
```bash
@inproceedings{zhang2023swipe,
  title        = {SwIPE: Efficient and Robust Medical Image Segmentation with Implicit Patch Embeddings},
  author       = {Zhang, Charley Y. and Gu, Pengfei and Sapkota, Nishchal and Chen, Danny Z},
  booktitle    = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages        = {315--326},
  year         = {2023},
  organization = {Springer}
}
```

© This code is made available under the Commons Clause License and is available for non-commercial academic purposes.


