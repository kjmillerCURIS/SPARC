# ✨SPARC✨

**✨SPARC✨: Score Prompting and Adaptive Fusion for Zero-Shot Multi-Label Recognition in Vision-Language Models**

[Published in CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Miller_SPARC_Score_Prompting_and_Adaptive_Fusion_for_Zero-Shot_Multi-Label_Recognition_CVPR_2025_paper.pdf)

## Install

After cloning our repo, please install numpy and sklearn, e.g.

```pip install scipy ```

```pip install scikit-learn ```

## Data

For ease of use, we have provided all necessary [CLIP cosine similarities](https://drive.google.com/drive/folders/1jluZ7tJq5LUceptu4mBuAjppzs8VFVFz?usp=sharing) for reproducing our main results (Tab. 2 in our paper). We computed these similarities using [these](https://github.com/kjmillerCURIS/dualcoopstarstar/blob/main/cooccurrence_correction_experiments/compute_cossims_test_noaug.py) [codes](https://github.com/kjmillerCURIS/dualcoopstarstar/blob/main/cooccurrence_correction_experiments/compute_cossims_test_noaug_arbitrary_prompts.py), although the process involved is fairly standard (and described in our paper). The data files also include image filenames, ground-truth labels, classnames, and compound prompts.

## Run

Our pipeline requires just a CPU! 😻

 ``` python run_SPARC.py --input_dir=<path to data folder> --dataset_name=<COCO2014, VOC2007, or NUSWIDE> --model_type=<ViT-L14336px, ViT-L14, ViT-B16, ViT-B32, RN50x64, RN50x16, RN50x4, RN101, or RN50> --output_prefix=<prefix for output filenames> ```

The script will produce a .csv file with mAPs, and a .pkl file with both mAPs and individual class APs.

## Coming soon...

Link to CVPR camera-ready version of our paper

Code for noise model analysis

Code for ablations and other results
