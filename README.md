# LiteReconfig: Cost and Content Aware Reconfiguration of Video Object Detection Systems for Mobile GPUs
Authors: Ran Xu (https://starsthu2016.github.io/), [Jayoung Lee](https://www.schaterji.io/team/jayoung-lee.html),  [Pengcheng Wang](https://www.schaterji.io/team/pengcheng-wang.html), [Saurabh Bagchi](https://saurabhbagchi.us), [Yin Li](https://www.biostat.wisc.edu/~yli/), and [Somali Chaterji](https://schaterji.io/)

An adaptive video object detection system selects different execution paths at runtime, based on a user specified latency requirement, video content characteristics, and available resources on a platform, so as to maximize its accuracy under the target latency service level agreement (SLA). Such a system is well suited for mobile devices with limited computing resources, often times running multiple contending applications. In spite of several recent efforts, we show that existing solutions suffer from two major drawbacks when facing a tight latency requirement (e.g., 30 fps). First, it can be very expensive to collect some feature values for a scheduler to decide on the best execution branch to run. Second, the system suffers from the switching overhead of transitioning from one branch to another, which is variable depending on the transition pair. This paper addresses these challenges and presents LiteReconfig --- an efficient and adaptive video object detection framework for mobiles. Underlying LiteReconfig is a cost-benefit analyzer for the scheduler that decides which features to use and then which execution branch to run at inference time. LiteReconfig is further equipped with a content-aware accuracy prediction model to select an execution branch tailored for frames in a streaming video. With a large-scale real-world video dataset and multiple current generation embedded devices, we demonstrate that LiteReconfig achieves significantly better accuracy under a set of varying latency requirements when compared to existing adaptive object detection systems, while running at speeds up to 50 fps on an NVIDIA AGX Xavier board.

# Contact
If you have any questions or suggestions, feel free to email Ran Xu (xu943@purdue.edu).

# Setup the Evaluation
## Hardware Prerequisite and Get Access to Devices
An NVIDIA Jetson TX2 and an NVIDIA Jetson AGX Xavier boards. We provide two boards for you to evaluate on. Please find the ip address and password in the Artifact Apeendix.
```
ssh ae@$TX2_IP # TX2
ssh ae@$AGX_IP # AGX Xavier
```

## Installation
If you use the two boards that we provide, you may use the virtual environment that we provide,
```
conda activate ae
```
Otherwise, check our [Installation Guide](INSTALL.md).

## Code, Models, and Datasets
If you use the two boards that we provide, the code, models, and datasets are already placed in the following file tree,
```
~/LiteReconfig_AE  # source code directory
~/LiteReconfig_AE/models  # all the trained models
$PATH_TO_DATASETS  # the path to ILSVRC 2015 dataset
```
Otherwise, use the following code to download code and models. You must manully download the ILSVRC 2015 dataset [here](https://image-net.org/challenges/LSVRC/2015/) and place it according to the file tree above.
```
git clone https://github.com/StarsThu2016/LiteReconfig_AE
cd LiteReconfig_AE
gdown https://drive.google.com/uc?id=1rexa1JsWmREy_nOGzO-opeTDtVki2iP_
tar -xvf models.tar && rm models.tar
```

# Major claims
* [C1] LiteReconfig achieves 45.4% mAP accuracy at 30 fps on the NVIDIA TX2 board under no resource contention for a video object detection task. The accuracy is 46.4% mAP accuracy at 50 fps on the NVIDIA Xavier board. This is proven by experiment (E1) described in Section 5.2 whose results are reported in Table 2.
* [C2] LiteReconfig improves the accuracy 1.8% to 3.5% mean average precision (mAP) over the state-of-the-art (SOTA) adaptive object detection systems. This is proven by experiment (E2) described in Section 5.2 whose results are reported in Table 2.
* [C3] LiteReconfig is 74.9X, 30.5X, and 20.0X faster than SELSA, MEGA, and REPP on the jetson TX2 board. This is proven by experiment (E3) described in Section 5.2 whose results are reported in Table 3.
* [C4] LiteReconfig is 1.0% and 2.2% mAP better than LiteReconfig-MaxContent-ResNet given (0% contention, 33.3 ms latency SLA) and (50% contention, 50.0 ms latency SLA) cases. This is proven by experiment (E4) described in Section 5.2 whose results are reported in Table 2.

# Experiments
## Experiment (E1)
[Key accuracy and latency performance of LiteReconfig] [10 human-minutes + 4 compute-hours]: we will run LiteReconfig on two types of embedded devices and examine the key accuracy and latency performance of it. Expected accuracy and latency on TX2 are 45.4% mAP and < 33.3 ms (95 percentile), and those on Xavier are 46.4% mAP and < 20.0 ms (95 percentile) (claim C1).

On TX2, run the following commands,
```
$ conda activate ae
(ae) $ cd ~/LiteReconfig_AE
(ae) $ python LiteReconfig.py --gl 0 \
  --lat_req 33.3 --mobile_device=tx2 \
  --output=test/executor_LiteReconfig.txt
```

On AGX Xavier, run the following commands (this can be done in parallel with the ones on TX2).
```
$ conda activate ae
(ae) $ cd ~/LiteReconfig_AE
(ae) $ python LiteReconfig.py --gl 0 \
  --lat_req 20 --mobile_device=xv \
  --output=test/executor_LiteReconfig.txt
```
The results will be written to ```test/executor_LiteReconfig_g0_{lat33_tx2,lat20_xv}_{det,lat}.txt```. We have saved a copy of these files in ```offline_logs_AE/```, and use ```python offline_eval_exp1.py``` to compute the accuracy and latency from these results files. One may replace the filenames by those in the online execution.

## Experiment (E2)
[Accuracy improvement at the same latency over the state-of-the-art (SOTA) work, ApproxDet] [20 human-minutes + 12 compute-hours]: we will run LiteReconfig on TX2 given 100 ms latency SLA and examine the true accuracy and latency metrics of it. Then we will compare it with ApproxDet. Expected accuracy and latency of LiteReconfig under no contention is 50.3% mAP and less than 100 ms 95 percentile latency, which is 3.5% higher than that of ApproxDet in the same condition (46.8%). Under 50% GPU contention and 100 ms SLA, the accuracy of LiteReconfig is 47.0%, which is 1.8% mAP higher than that of SmartAdapt (45.2%) (claim C2).  
On TX2, run the following command,
```
$ conda activate ae
(ae) $ cd ~/LiteReconfig_AE
(ae) $ python LiteReconfig.py --gl 0 \
  --lat_req 100 --mobile_device=tx2 \
  --output=test/executor_LiteReconfig.txt
(ae) $ python LiteReconfig_CG.py --GPU 50 
(ae) $ python LiteReconfig.py --gl 50 \
  --lat_req 100 --mobile_device=tx2 \
  --output=test/executor_LiteReconfig.txt
(ae) $ python LiteReconfig_CG.py --GPU 0 
```
The results will be written to ```test/executor_LiteReconfig_g{0,50}_lat100_tx2_{det,lat}.txt```. We have saved a copy of these files in ```offline_logs_AE/```, and use ```python offline_eval_exp2.py``` to compute the accuracy and latency from these results files. One may replace the filenames by those in the online execution.

## Experiment (E3)
[Latency improvement of LiteReconfig over accuracy-optimized baselines, i.e. SELSA, MEGA, and REPP] [20 human-minutes + 1 compute-hours]: we will run LiteReconfig on the TX2 and examine the latency performance of it. Expected mean latency of LiteReconfig is 28.2 ms. Those of SELSA, MEGA, and REPP are 2112 ms, 861 ms, and 565 ms. So LiteReconfig achieves 74.9X, 30.5X, and 20.0X speed up over these three baselines (claim C3).
On TX2, run the following commands,
```
$ conda activate ae
(ae) $ cd ~/LiteReconfig_AE
(ae) $ python LiteReconfig.py --gl 0 \
  --lat_req 33.3 --mobile_device=tx2 \
  --output=test/executor_LiteReconfig.txt
```
The results will be written to ```test/executor_LiteReconfig_g0_lat33_tx2_{det,lat}.txt```. We have saved a copy of these files in ```offline_logs_AE/```, and use ```python offline_eval_exp3.py``` to compute the accuracy and latency from these results files. One may replace the filenames by those in the online execution.

## Experiment (E4)
[Accuracy improvement at the same latency over a variant of LiteReconfig, i.e. LiteReconfig-MaxContent-ResNet] [20 human-minutes + 10 compute-hours]: we will run LiteReconfig and LiteReconfig-MaxContent-ResNet on the TX2 and examine the accuracy and latency performance of them. Expected accuracy given no contention and 33.3 ms latency SLA is 45.4% for LiteReconfig and 44.4% for LiteReconfig-MaxContent-ResNet. Expected accuracy given 50\% contention and 50 ms latency SLA is 43.6% for LiteReconfig and 41.4% for LiteReconfig-MaxContent-ResNet. Thus, LiteReconfig is 1.0% and 2.2% mAP better than LiteReconfig-MaxContent-ResNet in these two cases  (claim C4).
On TX2, run the following commands,
```
$ conda activate ae
(ae) $ cd ~/LiteReconfig_AE
(ae) $ python LiteReconfig.py --gl 0 \
  --lat_req 33.3 --mobile_device=tx2 \
  --output=test/executor_LiteReconfig.txt
(ae) $ python LiteReconfig_MaxContent.py \
  --protocol SmartAdapt_RPN --gl 0 \
  --lat_req 33.3 --mobile_device=tx2 \
  --output=test/executor_LR_MC_ResNet.txt
(ae) $ python LiteReconfig_CG.py --GPU 50 
(ae) $ python LiteReconfig.py --gl 50 \
  --lat_req 50 --mobile_device=tx2 \
  --output=test/executor_LiteReconfig.txt
(ae) $ python LiteReconfig_MaxContent.py \
  --protocol SmartAdapt_RPN --gl 50 \
  --lat_req 50 --mobile_device=tx2 \
  --output=test/executor_LR_MC_ResNet.txt
(ae) $ python LiteReconfig_CG.py --GPU 0 
```
The results will be written to ```test/executor_{LiteReconfig,LR_MC_ResNet}_{g0_lat33,g50_lat50}_tx2_{det,lat}.txt```. We have saved a copy of these files in ```offline_logs_AE/```, and use ```python offline_eval_exp4.py``` to compute the accuracy and latency from these results files. One may replace the filenames by those in the online execution.
