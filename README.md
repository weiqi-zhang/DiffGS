<p align="center">
<h1 align="center">DiffGS: Functional Gaussian Splatting Diffusion <br>
(NeurIPS 2024)</h1>
<p align="center">
    <a href="https://junshengzhou.github.io/"><strong>Junsheng Zhou*</strong></a>
    ·
    <a href="https://weiqi-zhang.github.io/"><strong>Weiqi Zhang*</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu</strong></a>
</p>
<p align="center"><strong>(* Equal Contribution)</strong></p>
<h3 align="center"><a href="https://arxiv.org/abs/2410.19657">Paper</a> | <a href="https://junshengzhou.github.io/DiffGS">Project Page</a></h3>
<div align="center"></div>
</p>
<p align="center">
    <img src="figs/mainfig.png" width="780" />
</p>


We release the code of the paper <a href="https://arxiv.org/abs/2410.19657">DiffGS: Functional Gaussian Splatting Diffusion</a> in this repository.

## Abstract

<p>
In this work, we propose DiffGS, a general Gaussian generator based on latent diffusion models. DiffGS is a powerful and efficient 3D generative model which is capable of generating Gaussian primitives at arbitrary numbers for high-fidelity rendering with rasterization. We explore DiffGS for various tasks, including unconditional generation, conditional generation from text, image, and partial 3DGS, as well as Point-to-Gaussian generation. We believe that DiffGS provides a new direction for flexibly modeling and generating Gaussian Splatting. 
          </p>

## Method

<p align="center">
  <img src="figs/method.png" width="780" />
</p>
<p style="margin-top: 30px">
<b>Overview of DiffGS.</b> <b>(a)</b> We disentangle the fitted 3DGS into three Gaussian Splatting Functions to model the Gaussian probability, colors and transforms. We then train a Gaussian VAE with a conditional latent diffusion model for generating these functions. <b>(b)</b> During generation, we first extract Gaussian geometry from the generated <b>GauPF</b>, followed by the <b>GauCF</b> and <b>GauTF</b> to obtain the Gaussian attributes.
</p>


## Generation Results

### Unconditional Generations

<img src="./figs/unconditional.gif" class="center">

### Visual Comparisons

<img src="./figs/shapenetchair.png" class="center">

## Visualization Results

### Text/Image-to-3D Generation

<img src="./figs/conditional.gif" class="center">

### Gaussian Completion

<img src="./figs/completion.gif" class="center">

### Point-to-Gaussian Generation

<img src="./figs/point2gaussian.gif" class="center">

## Install
We recommend creating an [anaconda](https://www.anaconda.com/) environment using our provided `environment.yml`:
```
conda env create -f environment.yml
conda activate diffgs
```
**Notice**：Since the code uses the original repository of Gaussian Splatting, please follow the environment setup instructions provided in the [official repository](https://github.com/graphdeco-inria/gaussian-splatting) to install the required dependencies.

## Pretrained model
We first provide the pretrained models: `Gaussian VAE` and `Gaussian LDM` of the chair unconditional model. Please download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/13JyZtXV6ep26HnVIiFza0jn9F8VL5I1_?usp=sharing).


## Inference

To inference pretrained model of ShapeNet Chair, save the downloaded model checkpoint  to `config/stage1` and `config/stage2`. Additionally, you also need to adjust the checkpoint path in `config/genetate/specs.json`, then run the following command:

```
python test.py -e config/generate/
```
## Data preparation
1. We would like to thank [Stanford ShapeNet Renderer repository](https://github.com/panmari/stanford-shapenet-renderer) for their contribution,  we have made modifications to the code based on their open-source work. Please install `Blender` and run the following command: 

```bash
cd proecess_data
blender --background --python render_blender.py -- --output_folder {images_path} {mesh_path}
```

2. Next, perform point sampling on the mesh and modify the `shapene_folder` path in `sample_points.py`. The sampled points will be used as the initial positions for the Gaussians.
```
python sample_points.py
```
3. Run the Gaussian fitting script provided by us.

```
python train_gaussian.py -s <path to COLMAP or NeRF Synthetic dataset>
```
4. Run the conversion script `convert.py` provided by us to transform the Gaussians into data suitable for training, and perform sampling of the Gaussian probability field.

```
python convert_data.py
```

## Training

### 1. Train Gaussian modulations
```
python train.py -e config/stage1/ -b 4 -w 8    # -b for batch size, -w for workers, -r to resume training
```

### 2. Train the diffusion model using the modulations extracted from the first stage
```
# extract the modulations / latent vectors, which will be saved in a "modulations" folder in the config directory
# the folder needs to correspond to "Data_path" in the diffusion config files

python test.py -e config/stage1/ -r {num epoch}

python train.py -e config/stage2 -b 32 -w 8 
```

### Application: 

### 1. Trian Point to Gaussian

If you want to train point2gaussian, simply add `--point2gs` after the "Train Gaussian modulations" command.

```
python train.py -e config/stage1/ -b 4 -w 8 --point2gs
```

### 2. Train Conditional Generation

If you want to train a conditional generative model, please first prepare the condition for each Gaussian, set the `context_path` in `specs.json` to the correct path, and then run the following command.

```
python train.py -e config/stage2_conditional -b 32 -w 8 
```



## Citation

If you find our code or paper useful, please consider citing

    @inproceedings{diffgs,
        title={DiffGS: Functional Gaussian Splatting Diffusion},
        author={Zhou, Junsheng and Zhang, Weiqi and Liu, Yu-Shen},
        booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
        year={2024}
    }


---

- 安装依赖时，出现`ERROR: Could not find a version that satisfies the requirement clip==1.0 (from versions: 0.0.1, 0.1.0, 0.2.0)`，需要自行从git安装，并注释了对应`environment.yml`文件：

```shell
python -m pip install git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
```

另外，手动安装子模块：
```shell
python -m pip install process_data/submodules/diff-gaussian-rasterization
python -m pip install process_data/submodules/simple-knn
```

否则，会出现下载迟滞问题。

---

- 通过hugging face申请获得了下载ShapeNetCore数据集的权限，通过token直接把数据从hugging face下载到服务器上，避免了网盘传输或本地传输速度过慢的问题。

---

- 在数据准备工作中，需要安装2.9版本的blender：

```shell
cd /opt
wget https://download.blender.org/release/Blender2.93/blender-2.93.2-linux-x64.tar.xz
tar -xvf blender-2.93.2-linux-x64.tar.xz
echo 'export PATH=$PATH:/opt/blender-2.93.2-linux-x64/' >> ~/.bashrc
rm blender-2.93.2-linux-x64.tar.xz
```

---

- 在headless的server中，运行render_blender，需要安装`xvfb`：

```shell
apt update && apt install xvfb -y
```

然后如下运行脚本：
```shell
xvfb-run -a blender --background --python render_blender.py -- --output_folder {images_path} {mesh_path}
```

---

- 在编制数据集目录结构时，由于ShapeNetCore体积太大，编写了一个自动化解压脚本，筛选并解压了其中数据量较小的类别。

- 在针对obj文件进行渲染时，由于每个类别中有非常多物体，例如，chair类别中有9000多个文件夹，不可能逐个执行`render_blender.py`文件进行渲染。因此，添加了针对obj文件的批处理脚本，可以一键渲染多个obj文件并保持对应路径。
```shell
python render_blender_batch.py -s {shapenet_folder}
```

---

- 采集点云时，出现`AttributeError: 'Scene' object has no attribute 'area'.`，修改了`sample_points.py`中对应代码，手动将scene转换为单一mesh，然后再调用`mesh.sample`函数进行点云采样。同时，修改了点云文件保存的路径，保存在物品id对应的文件夹下，这样`dataset_reader.py`才能正确读取。

---

- 运行`train_gaussian.py`，准备GS数据时，出现`ValueError: no field of name nx'`，通过断点调试和单步执行，定位错误到文件`dataset_readers.py`的`fetchPly`函数，意思是读取的点云文件中没有法线数据，判断是`sample_points.py`中`point_cloud.export`函数没有保存法线数据。此外，还发现`fetchPly`函数读取到的点云文件中颜色值都是0。于是，不再使用`point_cloud.export`方法保存数据，转而手动构建plyfile文件，确保点云文件中包含xyz、rgb、normal等数据。

---

- 新建`train_gaussian_batch.py`，批量准备多个场景的数据，默认保存在`process_data/output`下，文件路径参考ShapeNetCore路径，遵循先类别后物体的双重目录结构。

---

- 修正了准备stage2阶段特征向量数据时，运行`python test.py -e config/stage1/ -r {num epoch}`报错的bug，具体原因为`test.py`中未加入`-r`parser参数。现在运行参数为`python test.py -e config/stage1/ -r {ckpt_fileName}`。

---

- 修正了在stage2阶段unconditional diffusion training报错`TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>`的bug，具体原因为在无条件输入的情况下，数据加载器`ModulationLoader`的`__getitem__`函数会返回包含None值的字典，而torch在按批次加载数据时，会把字典中同一个键的值堆叠起来，遇到None就出错了。解决方法为在`train.py`中略微修改`torch.utils.data.DataLoader`创建对象的参数，在参数`collate_fn`中传入一个经过略微修改的`default_collate`，规避了遇到None值的堆叠。

---

- 利用保存的ckpt，进行无条件输入的diffusion生成场景：修改`config/generate/specs.json`中的ckpt路径，然后运行：

```shell
python test.py -e config/generate/ --epoches 1 -n 5
```

`epoches`控制生成的轮数，`-n`控制每轮生成的数量。

目前，源代码仓库没有提供有条件生成场景的代码。但是，在`diffusion.py`中，另有一个成员函数`generate_from_pc`，而`test.py`中调用的是`generate_unconditional`，因此，函数`generate_from_pc`应该就是原文中用来完成point2GS任务的方法。

---

- 可视化：

1. SIBR_viewer只能可视化数据准备阶段的GS场景，无法可视化diffusion生成的GS场景，因为SIBR_viewer对目录结构有要求。
2. `render.py`文件用来批量生成数据准备阶段的GS场景的渲染图片，并保证相机视角与训练数据一致，应该是用来准备计算PSNR重建指标的。但是同样有一个问题，`render.py`对目录结构也有要求，无法直接渲染diffusion生成的GS场景。
3. 实时可视化diffusion生成的GS场景的方法：导入生成的ply场景，`https://superspl.at/editor`

---

- 条件输入的形式：

根据代码逻辑，需要在`config1/modulations`中准备好该场景对应的一个或多个条件输入转化出的特征向量，文件格式形同场景的潜在向量`latent.txt`，且以`text`开头。在训练时会随机在这些条件输入的特征向量中选择一个，来训练条件生成模型。

关键问题：如何准备条件输入的特征向量？代码仓库中没有提供相关脚本，也没有任何说明。

推测： 条件输入数据需要另外手动生成，例如使用CLIP提取特征（environment.yaml中要求了这个库），并保存为以`text`开头的文件名，放置在同级目录下。

1. 根据代码逻辑，条件输入的维度是[B, N, 3]，后续经过`conv_pointnet`，即，直接用点云作为条件输入。但是，按照论文中的叙述，`partial 3DGS`作为条件输入，那么也应该是[B, N, 59]。
2. 如果是文本或图像的特征向量作为条件输入，那么应该是[B, D]，后续也不可能经过`conv_pointnet`，就需要另外的处理方式。
3. 原文消融实验里涉及的重建指标是关于GS VAE模块的，与diffusion生成无关。因此，不是`partial 3DGS`作为条件输入的结果，应该是point2GS任务或者GS2GS的重建结果。