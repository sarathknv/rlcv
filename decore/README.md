# Implementation of DECORE: Deep Compression with Reinforcement Learning

[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Alwani_DECORE_Deep_Compression_With_Reinforcement_Learning_CVPR_2022_paper.pdf) (CVPR 2022)  

## Dependencies  
* Python3  
* PyTorch
* [NNI (Neural Network Intelligence)](https://nni.readthedocs.io/en/stable/)
* tqdm
* Matplotlib


## Contents

```prune_model.ipynb```: for pruning a model using DECORE.  
```investigate_policies.ipynb```: implementation of section 5 of the paper (Analysis: Does DECORE find important channels?).  
```agents.py```: contains the `Agent` class that handles interaction of agents with the network, and a function for attaching agents.  
```models.py```: just VGG16 for now. This is a modified VGG16, please refer to the paper for details.  
```training_utils.py```: training and validation functions.  
```best.pth.tar```: pretained weights of VGG16 trained on CIFAR-10. 


## Experiments

VGG16 trained on CIFAR-10  
Note: Percentage inside brackets is pruned rate.  
Note: DECORE-λ is DECORE with penalty λ.  
Note: For FLOPs and Params calculation, I've included all kinds of operations, not just convolution and linear. 

| Model           | Accuracy mine     | FLOPs mine | Params mine| Accuracy paper    | FLOPs paper   |  Params paper |
|:----------------|:-----------------:|:--------------:|:--------------:|:-----------------:|:-----------------:|:-----------------:|        
| VGG16           | 91.94             | 314.29M(0.0%)  | 14.99M(0.0%)   | 313.73M(0.0%)     | 14.98M(0.0%)      |  
| DECORE-50       |                   |                |                |                   |                   |    
| DECORE-4        | 89.41             | 43.41M(86.18%) | 2.01M(86.55%)  | -                 | -                 |    

&nbsp;    
Analysis: Does DECORE find important channels?



## Citations

```bibtex
@inproceedings{alwani2022decore,
  title={DECORE: Deep Compression with Reinforcement Learning},
  author={Alwani, Manoj and Wang, Yang and Madhavan, Vashisht},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12349--12359},
  year={2022}
}
```
