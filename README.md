# Tofu

Code and data for [Learning Stable Classifiers by Transferring Unstable Features](https://proceedings.mlr.press/v162/bao22a.html) ICML 2022

![](summary.png)

If you like this work and use it on your own research, please cite our paper.

```
@inproceedings{bao2022learning,
  title={Learning Stable Classifiers by Transferring Unstable Features},
  author={Bao, Yujia and Chang, Shiyu and Barzilay, Regina},
  booktitle={International Conference on Machine Learning},
  pages={1483--1507},
  year={2022},
  organization={PMLR}
}
```




## Get started
**Setup [conda](https://docs.conda.io/en/latest/)**
1. Create a new conda environment with the required dependencies
`conda env create --file environment.yml`
2. Activate the environment `conda activate tofu`

**Download the datasets**
+ You can download the datasets by running the script `bin/download_data.sh`. All datasets will be downloaded under `datasets/`

**Run some experiments**
+ `bin/beer.sh` transfer from aspect look to aspect aroma
+ `bin/ask2me.sh` transfer from task penetrance to task incidence
+ `bin/bird.sh` transfer from seabird to waterbird
+ `bin/mnist.sh` transfer from even to odd
+ `bin/celeba.sh` transfer from eyeglasses to blond hair

## Acknowledgements

Research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

🌈 This paper is dedicated to the memory of our beloved family member [Tofu](https://people.csail.mit.edu/yujia/samoyed/), who filled our lives with so many wuffs and wuvs.

```                                                                     
TTTTTTTTTTTTTTTTTTTTTTT                 ffffffffffffffff                    
T:::::::::::::::::::::T                f::::::::::::::::f                   
T:::::::::::::::::::::T               f::::::::::::::::::f                  
T:::::TT:::::::TT:::::T               f::::::fffffff:::::f                  
TTTTTT  T:::::T  TTTTTTooooooooooo    f:::::f       ffffffuuuuuu    uuuuuu  
        T:::::T      oo:::::::::::oo  f:::::f             u::::u    u::::u  
        T:::::T     o:::::::::::::::of:::::::ffffff       u::::u    u::::u  
        T:::::T     o:::::ooooo:::::of::::::::::::f       u::::u    u::::u  
        T:::::T     o::::o     o::::of::::::::::::f       u::::u    u::::u  
        T:::::T     o::::o     o::::of:::::::ffffff       u::::u    u::::u  
        T:::::T     o::::o     o::::o f:::::f             u::::u    u::::u  
        T:::::T     o::::o     o::::o f:::::f             u:::::uuuu:::::u  
      TT:::::::TT   o:::::ooooo:::::of:::::::f            u:::::::::::::::uu
      T:::::::::T   o:::::::::::::::of:::::::f             u:::::::::::::::u
      T:::::::::T    oo:::::::::::oo f:::::::f              uu::::::::uu:::u
      TTTTTTTTTTT      ooooooooooo   fffffffff                uuuuuuuu  uuuu
```
在这项工作中，我们明确地告知目标分类器关于源任务中不稳定特征。具体来说，我们通过对比源任务中的不同数据环境来导出对不稳定特征进行编码的表示。我们通过根据这种表示对目标任务的数据进行聚类并最小化这些集群的最坏情况风险来实现鲁棒性。

在资源稀缺的目标任务中，我们只能访问输入标签对。然而，在训练数据足够的源任务中，识别偏差可能更容易。例如，我们可能会从多个环境中收集示例，其中偏差特征与标签之间的相关性不同（Arjovsky 等人，2019 年）。这些源环境帮助我们定义我们想要调节的确切偏差特征。

在本文中，我们建议明确告知目标分类器关于源数据不稳定特征。具体来说，我们推导出一种使用源环境对这些不稳定特征进行编码的表示。然后，我们通过基于这种表示对示例进行聚类并应用组 DRO (Sagawa et al., 2019) 来识别不同的子种群，以最小化这些子种群的最坏情况风险。因此，我们强制目标分类器对不稳定特征的不同值具有鲁棒性。在上述示例中，动物将根据背景进行聚类，无论集群（背景）如何，分类器都应该表现良好。

剩下的问题是如何使用源数据环境计算不稳定的特征表示。继Bao等人(2021年)之后，我们假设不稳定的特征反映在跨环境的分类器传输期间观察到的错误中。例如，如果分类器使用背景来区分骆驼和牛，正确预测的骆驼图像会有沙漠背景，而那些预测错误的图像很可能有草地背景。更一般地说，我们证明了在标签值相同的示例中，具有相同预测结果的示例将具有比具有不同预测的示例更相似的不稳定特征。通过强制具有相同预测结果的示例在特征空间中保持更接近，我们获得了编码这些潜在不稳定特征的表示。

问题表述我们考虑从源任务到目标任务的转移问题。对于源任务，我们假设标准设置 (Arjovsky et al., 2019)，其中训练数据包含 n 个环境 E1。, En。在每个环境Ei中，示例来自联合分布Pi(x, y)。继 Woodward (2005) 之后，我们将不稳定特征 Z(x) 定义为与环境的标签差异相关的特征。我们注意到 Z(x) 对模型是未知的。对于目标任务，我们只能访问输入标签对 (x, y)（即没有环境）。我们假设目标标签与上述不稳定特征 Z 没有因果关联。然而，由于收集偏差，目标数据可能包含标签和 Z 之间的虚假相关性。我们的目标是转移 Z 在源任务中不稳定的知识，以便目标分类器不依赖于这些虚假特征。

虚假特征就是原来根据手写体的分类特征，fz等于预训练好的分类模型