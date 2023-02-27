# Online Update of Safety Assurances Using Confidence-Based Predictions
### [Lab Website](https://smlbansal.github.io/sia-lab/index.html) | [Paper](https://arxiv.org/abs/2210.01199)<br>

Kensuke Nakamura,
Somil Bansal<br>
University of Southern California

This is the open-source repository for the paper "Online Update of Safety Assurances Using Confidence-Based Predictions".

## Get started
Setup DeepReach. The instructions can be found the README [here](https://github.com/smlbansal/deepreach)

Setup Trajectron++. The instructions can be found [here](https://github.com/StanfordASL/Trajectron-plus-plus).

You will need to add Trajectron++ to this directory. We use pretrained directory found in the eccv2020 branch `Trajectron-plus-plus/experiments/nuScenes/models/int_ee_me/model_registrar-12.pt`

Furthermore, we provide two scenarios based on the original NuScenes dataset. These can be found in the two pkl files, and should be added into a directory like so: `Trajectron-plus-plus/experiments/nuScenes/processed/FILENAME.pkl`
You will need to create the `processed` directory. This takes the original NuScenes scenarios and processes the data into something Trajectron++ can parse.

## Reproducing the parameterized FRT
The directory `experiment_scripts` contains one script per experiment in the paper.

To monitor progress, the training code writes tensorboard summaries into a "summaries"" subdirectory in the logging_root.

To start training DeepReach for the extended unicycle model run:
```
CUDA_VISIBLE_DEVICES=0 python experiment_scripts/train_dubins4dForwardSetParam2set_scaled.py --experiment_name experiment_1 --minWith target --num_src_samples 12000 --pretrain --pretrain_iters 40000 --num_epochs 150000 --counter_end 110000 --periodic_boundary --adjust_relative_grads --diffModel --diffModel_mode 'mode2' --collisionR 0.17
```
This will regularly save checkpoints in the directory specified by the rootpath in the script, in a subdirectory "experiment_1". 

We also provide a pretrained FRT, which is in `logs/dubins4dParamFRS_pre40_src12_epo150_rad0017_2set_adjgrad_scaled_time`   


To run our examples, run 
```
cd post_processing
python FRT_Query_final.py 
```

This will spit out the numerical data given in our paper.

## Citation
If you find our work useful in your research, please cite:
```
@misc{nakamura2022onlineupdate,
  doi = {10.48550/ARXIV.2210.01199},
  url = {https://arxiv.org/abs/2210.01199},
  
  author = {Nakamura, Kensuke and Bansal, Somil},
  
  keywords = {Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Online Update of Safety Assurances Using Confidence-Based Predictions},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Contact
If you have any questions, please feel free to email at k.nakamura@princeton.edu

