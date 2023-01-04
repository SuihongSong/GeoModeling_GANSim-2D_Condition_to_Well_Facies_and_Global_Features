## GANSim: Conditional Facies Simulation Using an Improved Progressive Growing of GANs
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)


![Teaser image](./Generated_facies_models_Conditioned_to_mud_proportion_and_well_data.png) 
**Picture:** *These facies models are produced by pre-trained generator conditioned to input mud proportion and well facies data.*

![Teaser image](./Generated_facies_models_Conditioned_to_sinuosity_and_well_data.png) 
**Picture:** *These facies models are produced by pre-trained generator conditioned to input channel sinuosity and well facies data.*

This repository contains the official TensorFlow implementation of the following paper:

> **GANSim: Conditional Facies Simulation Using an Improved Progressive Growing of Generative Adversarial Networks (GANs)**<br>
> Suihong Song (CUPB & Stanford), Tapan Mukerji (Stanford), and Jiagen Hou (CUPB)<br>
> CUPB: China University of Petroleum - Beijing

> https://link.springer.com/article/10.1007/s11004-021-09934-0
>
> **Abstract:** Conditional facies modeling combines geological spatial patterns with different types of observed data, to build earth models for predictions of subsurface resources. Recently, researchers have used generative adversarial networks (GANs) for conditional facies modeling, where an unconditional GAN is first trained to learn the geological patterns using the original GANs loss function, then appropriate latent vectors are searched to generate facies models that are consistent with the observed conditioning data. A problem with this approach is that the time-consuming search process needs to be conducted for every new conditioning data. As an alternative, we improve GANs for conditional facies simulation (GANSim) by introducing an extra condition-based loss function and adjusting the architecture of the generator to take the conditioning data as inputs, based on progressive growing of GANs. The condition-based loss function is defined as the inconsistency between the input conditioning value and the corresponding characteristics exhibited by the output facies model, and forces the generator to learn the ability of being consistent with the input conditioning data, together with the learning of geological patterns. Our input conditioning factors include global features (e.g. the mud facies proportion) alone, local features such as sparse well facies data alone, and joint combination of global features and well facies data. After training, we evaluate both the quality of generated facies models and the conditioning ability of the generators, by manual inspection and quantitative assessment. The trained generators are quite robust in generating high-quality facies models conditioned to various types of input conditioning information.

This study is based on our previous study (
GeoModeling_Unconditional_ProGAN, see Github https://github.com/SuihongSong/GeoModeling_Unconditional_ProGAN). Our next study (GeoModeling_Conditional-to-Probability-maps-plus, see Github https://github.com/SuihongSong/GeoModeling_Conditional_to_Probability_maps_plus) is further based on this study.

For any question, please contact [songsuihong@126.com]<br>


## Resources

Material related to our paper is available via the following links:

- Paper: https://link.springer.com/article/10.1007/s11004-021-09934-0 or (my research gate) https://www.researchgate.net/profile/Suihong_Song.
- Code: (Github) https://github.com/SuihongSong/GeoModeling_Conditional_ProGAN 
- Training and test datasets: (Zenodo) https://zenodo.org/record/3993791#.X1FQuMhKhaR
- Pre-trained GANs: (Zenodo) https://zenodo.org/record/3993791#.X1FQuMhKhaR, or (Google Drive) https://drive.google.com/drive/folders/1A8oGyni8YBnJ4to2Uu7a03oMfewjbgIx

## Licenses

All material, including our training dataset, is made available under MIT license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

## System requirements

* Both Linux and Windows are supported, but Linux is suggested.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.10.0 or newer with GPU support.
* (NOTE: the codes can also be run with TensorFlow2.x environment after adjusting several lines of codes, see 'Codes adjustments for TensorFlow 2')
* One or more high-end NVIDIA GPUs. 
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.4.1 or newer.

## Using pre-trained networks

We trained four GANs, conditioned to global features (i.e., channel sinuosity, channel width, non-channel mud proportion) only, well facies data only, both channel sinuosity and well facies data, and both mud proportion and well facies data, respectively. In [Code](./Code/), there are two subfolders: [0_only_conditioning_to_global_features](./Code/0_only_conditioning_to_global_features/) and [1_conditioning_to_well_facies_alone_or_with_global_features](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/); the former is used for only conditioning to global features, and the latter is used for conditioning to well facies data alone or together with global features. All GANs are trained with progressive training method illustrated in our previous paper (see Github repository https://github.com/SuihongSong/GeoModeling_Unconditional_ProGAN).

The pre-trained GANs (including generators) are evaluated in `Analyses_of_Trained_Generator-xxxx.ipynb` (refered to as `*.ipynb` in following) files using Test dataset (Zenodo, https://zenodo.org/record/3993791#.X1FQuMhKhaR, Google Drive, https://drive.google.com/drive/folders/1A8oGyni8YBnJ4to2Uu7a03oMfewjbgIx) regarding the evaluation metrics in paper. Corresponding to the four pre-trained GANs, there are four evaluation \*.ipynb files: 

(1) only conditioning to global features [Analyses_of_Trained_Generator.ipynb](./Code/0_only_conditioning_to_global_features/Analyses_of_Trained_Generator.ipynb); 

(2) only conditioning to well facies data [Analyses_of_Trained_Generator-WellCond-AfterEnlarg.ipynb](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/Analyses_of_Trained_Generator-WellCond-AfterEnlarg.ipynb); 

(3) conditioning to channel sinuosity and well facies data [Analyses_of_Trained_Generator-Sinuosity-WellEnlarg.ipynb](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/Analyses_of_Trained_Generator-Sinuosity-WellEnlarg.ipynb);

(4) conditioning to mud proportion and well facies data [Analyses_of_Trained_Generator-MudProp-WellEnlarg.ipynb](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/Analyses_of_Trained_Generator-MudProp-WellEnlarg.ipynb).


Before running  \*.ipynb files, please download related code files, corresponding pre-trained networks, and Test dataset, and modify corresponding paths in \*.ipynb files. Readers are welcome to play with other randomly defined input conditioning data in \*.ipynb files. 

Our training and test datasets both include synthesized facies models, corresponding global features, well facies data, and probability maps. Each one facies model corresponds to one pair of global features (mud proportion, channel sinuosity, channel width, and channel orientation), 8 random well facies data, and 8 probability maps with various blurriness. Probability maps are not used in this study, but used in our another paper (see Github https://github.com/SuihongSong/GeoModeling_Conditional_to_Probability_maps_plus). These four categories of data can be extracted as numpy arrays using following code:
```
# Initialize TensorFlow session.
tf.InteractiveSession()

import dataset
# tfrecord_dir='TestData' to fetch test dataset, if tfrecord_dir='TrainingData' to fetch training dataset
# labeltypes: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity'
# well_enlarge: if True, well points occupy 4x4 area, otherwise occupy 1x1 area
test_set = dataset.load_dataset(data_dir=data_dir_test, verbose=True, tfrecord_dir='TestData', labeltypes = [1,2,3], well_enlarge = False, shuffle_mb = 0, prefetch_mb = 0)

# labels are from -1 to 1
image_test, label_test = test_set.get_minibatch_imageandlabel_np(3000)  
probimg_test, wellfacies_test = test_set.get_minibatch_probandwell_np(3000*8)
```

`*.ipynb` files can also be run on Colab: 


(1) open `*.ipynb` files in Colab by using Github link: `File` -> `Open notebook` -> `Github` -> enter the corresponding `*.ipynb` Github link. 

(2) in Colab, check tensorflow version by running `%tensorflow_version`, and run `%tensorflow_version 1.x` if the tensorflow has default version of 2.x; also, make sure `GPU` is used in Colab by `Change runtime type` (`Runtime` -> `Change runtime type`).

(3) download all related codes into Colab by running in Colab:
```
!git clone https://github.com/SuihongSong/GeoModeling_Conditional_ProGAN.git
```
(4) download test dataset from my google drive (https://drive.google.com/drive/folders/1A8oGyni8YBnJ4to2Uu7a03oMfewjbgIx) as \*.zip file, by running in Colab:
```
!gdown --id 1PYOduluZ3M7JcN5acSO3rnwJ9VFZc0SE  #1PYO… is the Google Drive id for the file
```
then unzip the downloaded `TestData.zip`, by running in Colab:
```
!unzip /content/TestData.zip
```
(5) download corresponding pre-trained GAN networks, by running in Colab:
```
!gdown --id 1h61drFXGR-WZOpEJ192xHOx9ANLpwRUL  #here 1h61d… refers Google Drive ID for pre-trained GAN network only conditioning to global features as an example; obtain Google Drive ID for other pre-trained GAN networks and replace it.
```

(6) run following codes in `*.ipynb` files, and play with the pre-trained generators. 


The pre-trained GAN networks are stored as standard pickle files:
```
# pre-trained generator directory path; please replace it with your own path.
network_dir = '/scratch/users/suihong/…/…/'

# replace with downloaded pre-trained generator name.
with open(network_dir + 'network-snapshot-011520.pkl', 'rb') as file:
G, D, Gs = pickle.load(file)
    # G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
```

The above code unpickles pre-trained GAN networks to yield 3 instances of networks. To generate facies models, you will want to use `Gs` or `G`. The exact details of the generator and discriminator are defined in [networks.py](./Code/0_only_conditioning_to_global_features/networks.py) or [networks.py](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/networks.py) (see ` G_paper `, and ` D_paper `). 

The input of generator contains latent vectors of 128 dimensions, global features, well facies data, or both global features and well facies data, depending on which data the generator is conditioned to:

(1) for generator only conditioned to global features, the inputs have dimensions of [[N, 128], [N, 3]], where N refers to number of input vectors, 128 is dimension of latent vector, and 3 means three types of global features are inputted as conditions.

(2) for generator only conditioned to well facies, the inputs have dimensions of [[N, 128], [N, 0], [N, 2, 64, 64]], where [2, 64, 64] refers to the dimension of each input well facies data, and 0 is because no global feature is conditioned to.

(3) for generator conditioned to channel sinuosity and well facies data, the inputs have dimensions of [[N, 128], [N, 1], [N, 2, 64, 64]].

(4) for generator conditioned to mud proportion and well facies data, the inputs have dimensions of [[N, 128], [N, 1], [N, 2, 64, 64]].

## Training dataset

The training dataset (Zenodo, https://zenodo.org/record/3993791#.X1FQuMhKhaR) includes synthesized facies models, corresponding global features, and sparsely distributed well facies data. Training facies models are stored as multi-resolution TFRecords. Each original facies model (64x64) is downsampled into multiple resolutions (32x32, …, 4x4) and stored in `1r*.tfrecords` files for efficient streaming during training. There is a separate `1r*.tfrecords` file for each resolution. Training global features are stored as `*.labels`, training probability maps are stored in `2probimages.tfrecordsand` although they are not used as conditioning data, and training well facies data is stored as `*3wellfacies.tfrecords`. 


### How to make training data as TFRecords?

(1) In our study, we synthesize training facies models using object-based method in Petrel software, and export them into one file as model properties with `"Gslib"` format. An Gslib format example of the exported file is [Format_example_of_simulated_facies_models_from_Petrel.txt](./Code/Format_example_of_simulated_facies_models_from_Petrel.txt).

First lines of the exported file are like:

>PETREL: Properties
>
>17820 % Number of synthesized facies models
>
>Facies unit1 scale1
>
>Facies unit1 scale1
>
>...
>
>Facies unit1 scale1
>
>% Totally, there are 64x64 lines, corresponding to 64x64 pixels in each facies model; each line has 17820 numbers splitted by space, corresponding to 17820 facies code values of 17820 generated facies realizations at each pixel. 0-background mud faceis, 1-channel sand facies, 2-channel bank facies.
>
>0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ... 0.000000 1.000000 2.000000
>
>0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 ... 0.000000 0.000000 0.000000
>
>...
>
>0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 ... 0.000000 0.000000 0.000000


(2) This exported file containing synthesized facies models is read in [Preparing_training_and_test_datasets.ipynb](./Code/Preparing_training_and_test_datasets.ipynb). The data in the file is rearranged into `(FaciesModelNumber, 1, 64, 64)`. 

In our study, when synthesizing facies models in Petrel, we only consider orientation of channels varying from 0 to 90 degrees, thus in [Preparing_training_and_test_datasets.ipynb](./Code/Preparing_training_and_test_datasets.ipynb), we further enlarge the facies model dataset by reversing the synthesized facies mdoels vertically whose orientation become from -90 to 0 degrees:
```
allimgs = np.concatenate((partimgs, partimgs[::-1,:,:]),2)
```
Other software, like SGeMS, can also be used to simulate the training facies models, as long as the final generated facies models are arranged into `(FaciesModelNumber, 1, 64, 64)`.

Global features (also called labels) are arranged into `(FaciesModelNumber, GlobalFeaturesNumber)`.

(3) The facies models are then used to simulate probability maps in `3 Generate probability maps` of [Preparing_training_and_test_datasets.ipynb](./Code/Preparing_training_and_test_datasets.ipynb). The probability maps are then used to produce well facies data in `4 Generate well facies` of [Preparing_training_and_test_datasets.ipynb](./Code/Preparing_training_and_test_datasets.ipynb). Although probability maps are not used in this case, they should be contained in a TFrecord training data `2probimages.tfrecords`, because `dataset.py` will need to take `2probimages.tfrecords` as inputs.

(4) When downsampling training facies models, two methods were proposed currently: averaging facies codes, or remaining the most frequent facies code. In this paper, here we use the averaging facies codes. In the near future, we will propose to use a third downsampling method: averaging indicator of each facies. 

```
# used to produce low-D with most frequent facies code
#real_img_t = np.expand_dims(real_img, axis = 3)
#real_img_t_c = np.concatenate((real_img_t[:, 0::2, 0::2], real_img_t[:, 0::2, 1::2], real_img_t[:, 1::2, 0::2], real_img_t[:, 1::2, 1::2]), axis = 3)                
#mode, _ = stats.mode(real_img_t_c, axis = 3)
#real_img = np.squeeze(mode, axis = 3)
                
# used to produce low-D with averaging method
real_img = (real_img[:, 0::2, 0::2] + real_img[:, 0::2, 1::2] + real_img[:, 1::2, 0::2] + real_img[:, 1::2, 1::2]) * 0.25  
```

## Training networks

Once the training dataset and related codes are downloaded, you can train your own facies model generators as follows:

1. Edit [config.py](./Code/0_only_conditioning_to_global_features/config.py) or [config.py](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/config.py) to set path `data_dir` (this path points to the folder containing `TrainingData` and `TestData` folders) for the downloaded training data and path for expected results `result_dir`, gpu number. Global feature types are set with following code:
```
labeltypes = [1]  # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity'; but the loss for channel orientation has not been designed in loss.py.
# [] for no label conditioning.
```
If using conventional GAN training process (non-progressive training), uncomment the line of code: 
```
#desc += '-nogrowing'; sched.lod_initial_resolution = 64; train.total_kimg = 10000
```
Set if the input well facies data is enlarged (each well facies data occupies 4x4 pixels) or unenlarged (each well facies data only occupies 1x1 pixel), by uncommenting or commenting following line of code:
```
dataset.well_enlarge = True; desc += '-Enlarg';  # uncomment this line to let the dataset output enlarged well facies data; comment to make it unenlarged.
```

2. Edit [train.py](./Code/0_only_conditioning_to_global_features/train.py) or [train.py](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/train.py) to set detailed parameters of training, such as parameters in `class TrainingSchedule` and `def train_progressive_gan`.

3. Set default path as the directory path of downloaded code files, and run the training script with `python train.py`. Or, edit path in [RunCode.py](./Code/0_only_conditioning_to_global_features/RunCode.ipynb) or [RunCode.py](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/RunCode.ipynb), and run `% run train.py` in `RunCode.py` files with Jupyter notebook.

## Assessment of the trained generator

Each of the four pre-trained generators are evaluated using Test dataset (Zenodo, https://zenodo.org/record/3993791#.X1FQuMhKhaR) in `Analyses_of_Trained_Generator-xxxx.ipynb ` files:

(1) for generator only conditioned to global features [Analyses_of_Trained_Generator.ipynb](./Code/0_only_conditioning_to_global_features/Analyses_of_Trained_Generator.ipynb); 

(2) for generator only conditioned to well facies data [Analyses_of_Trained_Generator-WellCond-AfterEnlarg.ipynb](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/Analyses_of_Trained_Generator-WellCond-AfterEnlarg.ipynb); 

(3) for generator conditioned to channel sinuosity and well facies data [Analyses_of_Trained_Generator-Sinuosity-WellEnlarg.ipynb](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/Analyses_of_Trained_Generator-Sinuosity-WellEnlarg.ipynb);

(4) for generator conditioned to mud proportion and well facies data [Analyses_of_Trained_Generator-MudProp-WellEnlarg.ipynb](./Code/1_conditioning_to_well_facies_alone_or_with_global_features/Analyses_of_Trained_Generator-MudProp-WellEnlarg.ipynb).

Detailed steps are illustrated inside these `*.ipynb` files. How to run them is also explained in previous section ` Using pre-trained networks `.

Please note that the exact results may vary from run to run due to the non-deterministic nature of TensorFlow.

## License
Most code files of this study are derived from the original Progressive GANs work (https://github.com/tkarras/progressive_growing_of_gans), but we have largely amended the original codes, especially networks.py, loss.py, dataset.py, and train.py. The original Progressive GANs codes are under license of Attribution-NonCommercial 4.0 International (https://creativecommons.org/licenses/by-nc/4.0/). Other materials produced by us (e.g., the trained models and the parts of codes we have added or adjusted) are under MIT license.

## Acknowledgements

Code for this project is improved from the original code of Progressive GANs (https://github.com/tkarras/progressive_growing_of_gans). We thank the authors for their great job.
