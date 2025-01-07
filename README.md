# [**[2025 IEEE International Conference on Automatic Face and Gesture Recognition (FG 2025)] 	Exploiting Aggregation and Segregation of Representations for Domain Adaptive Human Pose Estimation**](https://arxiv.org/abs/2412.20538)

### Run Training Codes

```

# Assume you have put the datasets under the path 

CUDA_VISIBLE_DEVICES=0 python epic.py data/RHD data/H3D_crop -s RenderedHandPose -t Hand3DStudio --seed 0 --debug --log logs

```

### Citation

If you find this code useful for your research, please cite our paper

```
@article{peng2024exploiting,
  title={Exploiting Aggregation and Segregation of Representations for Domain Adaptive Human Pose Estimation},
  author={Peng, Qucheng and Zheng, Ce and Ding, Zhengming and Wang, Pu and Chen, Chen},
  journal={arXiv preprint arXiv:2412.20538},
  year={2024}
}
```
