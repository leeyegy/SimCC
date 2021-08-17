# Is 2D Heatmap Even Necessary for Human Pose Estimation?
PyTorch training code and pretrained models for **SimDR** (**Sim**ple yet effective **D**isentangled **R**epresentation for keypoint coordinate) 

The 2D heatmap representation has dominated human pose estimation for years due to its high performance. However, heatmap-based approaches suffer from several shortcomings: 

- 1) The performance drops dramatically in the low-resolution images, which are frequently encountered in real-world scenarios. 

- 2) To improve the localization precision, multiple upsample layers may be needed to recover the feature map resolution from low to high, which are computationally expensive. 

- 3) Extra coordinate refinement is usually necessary to reduce the quantization error of downscaled heatmaps. 

**_Intro:_** Given the shortcomings revealed above, we don't think 2D heatmap is the final solution for keypoint coordinate representation to this field. By contrast, SimDR is a simple yet effective scheme which gets rid of extra post-processing and reduces the quantisation error by the coordinate representation design.  **For the first time**, SimDR brings **heatmap-free** methods to the competitive performance level of **heatmap-based** methods, outperforming the latter by a large margin in low input resolution cases. Additionally, SimDR allows one to directly remove the time-consuming upsampling module of some methods, which may inspire new researches on lightweight models for Human Pose Estimation


We hope proposed SimDR will motivate the community to rethink the design of coordinate representation for 2D human pose estimation.

For details see [Is 2D Heatmap Even Necessary for Human Pose Estimation](http://arxiv.org/abs/2107.03332) by Yanjie Li, Sen Yang, Shoukui Zhang, Zhicheng Wang, Wankou Yang, Shu-Tao Xia, Erjin Zhou.

![image](https://user-images.githubusercontent.com/35657511/123757223-55502b00-d8f0-11eb-8872-0072d7b61e91.png)


## News!
- [2021.08.17] The pretrained models are released in [Google Drive](https://drive.google.com/drive/folders/1HtIkWDpHasULk_MArlGLtyf-XRAyAsuP?usp=sharing)!
- [2021.07.09] The codes for SimDR and SimDR* (space-aware SimDR) are released!

## Experiments
### Results on COCO test-dev set 
|Method|Representation|Input size|GFLOPs|AP|AR|
|-|-|-|-|-|-|
|[SimBa-Res50](https://arxiv.org/abs/1804.06208)|heatmap|384x288|20.0|71.5|76.9|
|[SimBa-Res50](https://arxiv.org/abs/1804.06208)|**SimDR\***|384x288|20.2|**72.7**|**78.0**|
|[HRNet-W48](https://arxiv.org/abs/1902.09212)|heatmap|256x192|14.6|74.2|79.5|
|[HRNet-W48](https://arxiv.org/abs/1902.09212)|**SimDR\***|256x192|14.6|**75.4**|**80.5**|
|[HRNet-W48](https://arxiv.org/abs/1902.09212)|heatmap|384x288|32.9|75.5|80.5|
|[HRNet-W48](https://arxiv.org/abs/1902.09212)|**SimDR\***|384x288|32.9|**76.0**|**81.1**|


### Note:
* Flip test is used.
* Person detector has person AP of 60.9 on COCO test-dev2017 dataset.
* GFLOPs is for convolution and linear layers only.


### Results on COCO validation set
<details>
<table>
	<tr>
	    <th>Method</th>
	    <th>Representation</th>
	    <th>Input size</th>  
	    <th>#Params</th>
	    <th>GFLOPs</th>
	    <th>Extra post.</th>
	    <th>AP</th>
	    <th>AR</th>	
	</tr >
	<tr >
	    <td rowspan="9"><a href ="https://arxiv.org/abs/1804.06208">SimBa-Res50</a></td>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>34.0M</td>
	    <td>0.7</td>
	    <td>Y</td>
	    <td>34.4</td>
	    <td>43.7</td>		
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>34.0M</td>
	    <td>0.7</td>
	    <td>N</td>
	    <td>25.8</td>
	    <td>36.0</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>64x64</td>
	    <td>34.1M</td>
	    <td>0.7</td>
	    <td><b>N</b></td>
	    <td><b>40.8</b></td>
	    <td><b>49.6</b></td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>128x128</td>
	    <td>34.0M</td>
	    <td>3.0</td>
	    <td>Y</td>
	    <td>60.3</td>
	    <td>67.6</td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>128x128</td>
	    <td>34.0M</td>
	    <td>3.0</td>
	    <td>N</td>
	    <td>55.4</td>
	    <td>63.6</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>128x128</td>
	    <td>34.8M</td>
	    <td>3.0</td>
	    <td><b>N</b></td>
	    <td><b>62.6</b></td>
	    <td><b>69.5</b></td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>34.0M</td>
	    <td>8.9</td>
	    <td>Y</td>
	    <td>70.4</td>
	    <td>76.3</td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>34.0M</td>
	    <td>8.9</td>
	    <td>N</td>
	    <td>68.5</td>
	    <td>74.8</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>256x192</td>
	    <td>36.8M</td>
	    <td>9.0</td>
	    <td><b>N</b></td>
	    <td><b>71.4</b></td>
	    <td><b>77.4</b></td>
	</tr>
	<tr >
	    <td rowspan="9"><a href ="https://github.com/leeyegy/TokenPose">TokenPose-S</a></td>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>4.9M</td>
	    <td>1.4</td>
	    <td>Y</td>
	    <td>57.1</td>
	    <td>64.8</td>		
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>4.9M</td>
	    <td>1.4</td>
	    <td>N</td>
	    <td>35.9</td>
	    <td>47.0</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>64x64</td>
	    <td>4.9M</td>
	    <td>1.4</td>
	    <td><b>N</b></td>
	    <td><b>62.8</b></td>
	    <td><b>70.1</b></td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>128x128</td>
	    <td>5.2M</td>
	    <td>1.6</td>
	    <td>Y</td>
	    <td>65.4</td>
	    <td>71.6</td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>128x128</td>
	    <td>5.2M</td>
	    <td>1.6</td>
	    <td>N</td>
	    <td>57.6</td>
	    <td>64.9</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>128x128</td>
	    <td>5.1M</td>
	    <td>1.6</td>
	    <td><b>N</b></td>
	    <td><b>71.4</b></td>
	    <td><b>76.4</b></td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>6.6M</td>
	    <td>2.2</td>
	    <td>Y</td>
	    <td>72.5</td>
	    <td>78.0</td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>6.6M</td>
	    <td>2.2</td>
	    <td>N</td>
	    <td>69.9</td>
	    <td>75.8</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>256x192</td>
	    <td>5.5M</td>
	    <td>2.2</td>
	    <td><b>N</b></td>
	    <td><b>73.6</b></td>
	    <td><b>78.9</b></td>
	</tr>
	<tr>
	    <td rowspan="9"><a href ="https://arxiv.org/abs/1804.06208">SimBa-Res101</a></td>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>53.0M</td>
	    <td>1.0</td>
	    <td>Y</td>
	    <td>34.1</td>
	    <td>43.5</td>		
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>53.0M</td>
	    <td>1.0</td>
	    <td>N</td>
	    <td>25.7</td>
	    <td>36.1</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>64x64</td>
	    <td>53.1M</td>
	    <td>1.0</td>
	    <td><b>N</b></td>
	    <td><b>39.6</b></td>
	    <td><b>48.9</b></td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>128x128</td>
	    <td>53.0M</td>
	    <td>4.1</td>
	    <td>Y</td>
	    <td>59.2</td>
	    <td>66.7</td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>128x128</td>
	    <td>53.0M</td>
	    <td>4.1</td>
	    <td>N</td>
	    <td>54.4</td>
	    <td>62.5</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>128x128</td>
	    <td>53.5M</td>
	    <td>4.1</td>
	    <td><b>N</b></td>
	    <td><b>63.1</b></td>
	    <td><b>70.1</b></td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>53.0M</td>
	    <td>12.4</td>
	    <td>Y</td>
	    <td>71.4</td>
	    <td>77.1</td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>53.0M</td>
	    <td>12.4</td>
	    <td>N</td>
	    <td>69.5</td>
	    <td>75.6</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>256x192</td>
	    <td>53.7M</td>
	    <td>12.4</td>
	    <td><b>N</b></td>
	    <td><b>72.3</b></td>
	    <td><b>78.0</b></td>
	</tr>	
	<tr >
	    <td rowspan="9"><a href ="https://arxiv.org/abs/1902.09212">HRNet-W32</a></td>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>28.5M</td>
	    <td>0.6</td>
	    <td>Y</td>
	    <td>45.8</td>
	    <td>55.3</td>		
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>28.5M</td>
	    <td>0.6</td>
	    <td>N</td>
	    <td>34.6</td>
	    <td>45.6</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>64x64</td>
	    <td>28.6M</td>
	    <td>0.6</td>
	    <td><b>N</b></td>
	    <td><b>56.4</b></td>
	    <td><b>64.9</b></td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>128x128</td>
	    <td>28.5M</td>
	    <td>2.4</td>
	    <td>Y</td>
	    <td>67.2</td>
	    <td>74.1</td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>128x128</td>
	    <td>28.5M</td>
	    <td>2.4</td>
	    <td>N</td>
	    <td>61.9</td>
	    <td>69.4</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>128x128</td>
	    <td>29.1M</td>
	    <td>2.4</td>
	    <td><b>N</b></td>
	    <td><b>70.7</b></td>
	    <td><b>76.7</b></td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>28.5M</td>
	    <td>7.1</td>
	    <td>Y</td>
	    <td>74.4</td>
	    <td>79.8</td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>28.5M</td>
	    <td>7.1</td>
	    <td>N</td>
	    <td>72.3</td>
	    <td>78.2</td>
	</tr>
	<tr>
	    <td><b>SimDR</b></td>
	    <td>256x192</td>
	    <td>31.3M</td>
	    <td>7.1</td>
	    <td><b>N</b></td>
	    <td><b>75.3</b></td>
	    <td><b>80.8</b></td>
	</tr>
	<tr >
	    <td rowspan="9"><a href ="https://arxiv.org/abs/1902.09212">HRNet-W48</a></td>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>63.6M</td>
	    <td>1.2</td>
	    <td>Y</td>
	    <td>48.5</td>
	    <td>57.8</td>		
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>63.6M</td>
	    <td>1.2</td>
	    <td>N</td>
	    <td>36.9</td>
	    <td>47.8</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>64x64</td>
	    <td>63.7M</td>
	    <td>1.2</td>
	    <td><b>N</b></td>
	    <td><b>59.7</b></td>
	    <td><b>67.5</b></td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>128x128</td>
	    <td>63.6M</td>
	    <td>4.9</td>
	    <td>Y</td>
	    <td>68.9</td>
	    <td>75.3</td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>128x128</td>
	    <td>63.6M</td>
	    <td>4.9</td>
	    <td>N</td>
	    <td>63.3</td>
	    <td>70.5</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>128x128</td>
	    <td>64.1M</td>
	    <td>4.9</td>
	    <td><b>N</b></td>
	    <td><b>72.0</b></td>
	    <td><b>77.9</b></td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>63.6M</td>
	    <td>14.6</td>
	    <td>Y</td>
	    <td>75.1</td>
	    <td>80.4</td>
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>63.6M</td>
	    <td>14.6</td>
	    <td>N</td>
	    <td>73.1</td>
	    <td>78.7</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>256x192</td>
	    <td>66.3M</td>
	    <td>14.6</td>
	    <td><b>N</b></td>
	    <td><b>75.9</b></td>
	    <td><b>81.2</b></td>
	</tr>
</table>
</details>

### Note:
* Flip test is used.
* Person detector has person AP of 56.4 on COCO val2017 dataset.
* GFLOPs is for convolution and linear layers only.
* Extra post. = extra post-processing towards refining the predicted keypoint coordinate.


#### Results on higher input resolution
Results on the COCO validation set with the input size of 384×288.
<table>
	<tr>
	    <th>Method</th>
	    <th>Representation</th>
	    <th>AP</th>  
	    <th>AP_50</th>
	    <th>AP_75</th>
	    <th>AP_M</th> 		
	    <th>AP_L</th>
	    <th>AR</th> 		
	</tr >
	<tr >
	    <td rowspan="3"><a href ="https://arxiv.org/abs/1804.06208">SimBa-Res50</a></td>
	    <td>heatmap</td>
	    <td>72.2</td>
	    <td>89.3</td>
	    <td>78.9</td>
	    <td>68.1</td>
	    <td>79.7</td>
	    <td>77.6</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>73.0</td>
	    <td><b>89.3</b></td>
	    <td>79.7</td>
	    <td>69.5</td>
	    <td>79.9</td>
	    <td>78.6</td>
	</tr>
	<tr>
	    <td><b>SimDR* (ours)</b></td>
	    <td><b>73.4</b></td>
	    <td>89.2</td>
	    <td><b>80.0</b></td>
	    <td><b>69.7</b></td>
	    <td><b>80.6</b></td>
	    <td><b>78.8</b></td>
	</tr>	
	<tr >
	    <td rowspan="2"><a href ="https://arxiv.org/abs/1804.06208">SimBa-Res101</a></td>
	    <td>heatmap</td>
	    <td>73.6</td>
	    <td>89.6</td>
	    <td>80.3</td>
	    <td>69.9</td>
	    <td><b>81.1</b></td>
	    <td>79.1</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td><b>74.2</b></td>
	    <td><b>89.6</b></td>
	    <td><b>80.9</b></td>
	    <td><b>70.7</b></td>
	    <td>80.9</td>
	    <td><b>79.8</b></td>
	</tr>
	<tr >
	    <td rowspan="2"><a href ="https://arxiv.org/abs/1804.06208">SimBa-Res152</a></td>
	    <td>heatmap</td>
	    <td>74.3</td>
	    <td>89.6</td>
	    <td>81.1</td>
	    <td>70.5</td>
	    <td>81.6</td>
	    <td>79.7</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td><b>74.9</b></td>
	    <td><b>89.9</b></td>
	    <td><b>81.5</b></td>
	    <td><b>71.4</b></td>
	    <td><b>81.7</b></td>
	    <td><b>80.4</b></td>
	</tr>
	<tr >
	    <td rowspan="2"><a href ="https://arxiv.org/abs/1902.09212">HRNet-W48</a></td>
	    <td>heatmap</td>
	    <td>76.3</td>
	    <td>90.8</td>
	    <td>82.9</td>
	    <td>72.3</td>
	    <td>83.4</td>
	    <td>81.2</td>		
	</tr>
	<tr>
	    <td><b>SimDR* (ours)</b></td>
	    <td><b>76.9</b></td>
	    <td><b>90.9</b></td>
	    <td><b>83.2</b></td>
	    <td><b>73.2</b></td>
	    <td><b>83.8</b></td>
	    <td><b>82.0</b></td>
	</tr>	
</table>

### Note:
* Flip test is used.
* Person detector has person AP of 56.4 on COCO val2017 dataset.


### Results on MPII val set
<table>
	<tr>
	    <th>Method</th>
	    <th>Representation</th>
	    <th>Input size</th>  
	    <th>Hea</th>
	    <th>Sho</th>
	    <th>Elb</th>  
	    <th>Wri</th>
	    <th>Hip</th>
	    <th>Kne</th>  
	    <th>Ank</th>
	    <th>Mean</th>
	</tr >
	<tr >
	    <td colspan="11" align="center"><b>PCKh@0.5</b></td>
	</tr>	
	<tr >
	    <td rowspan="5"><a href ="https://arxiv.org/abs/1902.09212">HRNet-W32</a></td>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>89.7</td>
	    <td>86.6</td>
	    <td>75.1</td>
	    <td>65.7</td>
	    <td>77.2</td>
	    <td>69.2</td>
	    <td>63.6</td>
	    <td>76.4</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>64x64</td>
	    <td><b>96.5</b></td>
	    <td><b>89.5</b></td>
	    <td><b>77.5</b></td>
	    <td><b>67.6</b></td>
	    <td><b>79.8</b></td>
	    <td><b>71.5</b></td>
	    <td><b>65.0</b></td>
	    <td><b>78.7</b></td>		
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x256</td>
	    <td>97.1</td>
	    <td>95.9</td>
	    <td>90.3</td>
	    <td><b>86.4</b></td>
	    <td>89.1</td>
	    <td><b>87.1</b></td>
	    <td><b>83.3</b></td>
	    <td><b>90.3</b></td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>256x256</td>
	    <td>96.8</td>
	    <td>95.9</td>
	    <td>90.0</td>
	    <td>85.0</td>
	    <td>89.1</td>
	    <td>85.4</td>
	    <td>81.3</td>
	    <td>89.6</td>	
	</tr>
	<tr>
	    <td><b>SimDR* (ours)</b></td>
	    <td>256x256</td>
	    <td><b>97.2</b></td>
	    <td><b>96.0</b></td>
	    <td><b>90.4</b></td>
	    <td>85.6</td>
	    <td><b>89.5</b></td>
	    <td>85.8</td>
	    <td>81.8</td>
	    <td>90.0</td>	
	</tr>	
	<tr >
	    <td colspan="11" align="center"><b>PCKh@0.1</b></td>
	</tr>	
	<tr >
	    <td rowspan="4"><a href ="https://arxiv.org/abs/1902.09212">HRNet-W32</a></td>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>12.9</td>
	    <td>11.7</td>
	    <td>9.7</td>
	    <td>7.1</td>
	    <td>7.2</td>
	    <td>7.2</td>
	    <td>6.6</td>
	    <td>9.2</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>64x64</td>
	    <td><b>30.9</b></td>
	    <td><b>23.3</b></td>
	    <td><b>18.1</b></td>
	    <td><b>15.0</b></td>
	    <td><b>10.5</b></td>
	    <td><b>13.1</b></td>
	    <td><b>12.8</b></td>
	    <td><b>18.5</b></td>		
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x256</td>
	    <td>44.5</td>
	    <td>37.3</td>
	    <td>37.5</td>
	    <td>36.9</td>
	    <td>15.1</td>
	    <td>25.9</td>
	    <td>27.2</td>
	    <td>33.1</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>256x256</td>
	    <td><b>50.1</b></td>
	    <td><b>41.0</b></td>
	    <td><b>45.3</b></td>
	    <td><b>42.4</b></td>
	    <td><b>16.6</b></td>
	    <td><b>29.7</b></td>
	    <td><b>30.3</b></td>
	    <td><b>37.8</b></td>	
	</tr>	
</table>

### Note:
* Flip test is used.
* It seems that there is a bug while computing PCKh@0.1 in the original code, we have it fixed in this repo.



### Results on CrowdPose
<table>
	<tr>
	    <th>Method</th>
	    <th>Representation</th>
	    <th>Input size</th> 
	    <th>AP</th>
	    <th>AP_50</th>
	    <th>AP_75</th> 
	    <th>AP_E</th>
	    <th>AP_M</th>
	    <th>AP_H</th> 		
	</tr >
	<tr >
	    <td rowspan="4"><a href ="https://arxiv.org/abs/1902.09212">HRNet-W32</a></td>
	    <td>heatmap</td>
	    <td>64x64</td>
	    <td>42.4</td>
	    <td>69.6</td>
	    <td>45.5</td>
	    <td>51.2</td>
	    <td>43.1</td>
	    <td>31.8</td>		
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>64x64</td>
	    <td><b>46.5</b></td>
	    <td><b>70.9</b></td>
	    <td><b>50.0</b></td>
	    <td><b>56.0</b></td>
	    <td><b>47.5</b></td>
	    <td><b>34.7</b></td>		
	</tr>
	<tr>
	    <td>heatmap</td>
	    <td>256x192</td>
	    <td>66.4</td>
	    <td>81.1</td>
	    <td>71.5</td>
	    <td>74.0</td>
	    <td>67.4</td>
	    <td>55.6</td>
	</tr>
	<tr>
	    <td><b>SimDR (ours)</b></td>
	    <td>256x192</td>
	    <td><b>66.7</b></td>
	    <td><b>82.1</b></td>
	    <td><b>72.0</b></td>
	    <td><b>74.1</b></td>
	    <td><b>67.8</b></td>
	    <td><b>56.2</b></td>
	</tr>
</table>

## Start to use
### 1. Dependencies installation & data preparation
Please refer to [THIS](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) to prepare the environment step by step.

### 2. Model Zoo
Pretrained models are provided in our [model zoo](https://drive.google.com/drive/folders/1HtIkWDpHasULk_MArlGLtyf-XRAyAsuP?usp=sharing).

### 3. Trainging
#### Training on COCO train2017 dataset 
To train with **_SimDR_** as keypoint coordinate representation :
```
python tools/train.py \
    --cfg experiments/coco/hrnet/simdr/nmt_w48_256x192_adam_lr1e-3.yaml\
```
To train with **_SimDR\*_** as keypoint coordinate representation :
```
python tools/train.py \
    --cfg experiments/coco/hrnet/sa_simdr/w48_256x192_adam_lr1e-3_split2_sigma4.yaml\
```

**_*Note:_**
After using  **_SimDR_**, the decovonlution layers of SimpleBaseline can be reserved or removed.

#### Training on MPII dataset 
To train with **_SimDR_** as keypoint coordinate representation :
```
python tools/train.py \
    --cfg experiments/mpii/hrnet/simdr/norm_w32_256x256_adam_lr1e-3_ls2e1.yaml
```
To train with **_SimDR\*_** as keypoint coordinate representation :
```
python tools/train.py \
    --cfg experiments/mpii/hrnet/sa_simdr/w32_256x256_adam_lr1e-3_split2_sigma6.yaml
```
### 4. Testing
#### Testing on COCO val2017 dataset using model zoo's models
```
python tools/test.py \
    --cfg experiments/coco/hrnet/simdr/nmt_w48_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE _PATH_TO_CHECKPOINT_ \
    TEST.USE_GT_BBOX False
```
```
python tools/test.py \
    --cfg experiments/coco/hrnet/sa_simdr/w48_256x192_adam_lr1e-3_split2_sigma4.yaml \
    TEST.MODEL_FILE _PATH_TO_CHECKPOINT_ \
    TEST.USE_GT_BBOX False
```

#### Testing on MPII dataset using model zoo's models
```
python tools/test.py \
    --cfg experiments/mpii/hrnet/simdr/norm_w32_256x256_adam_lr1e-3_ls2e1.yaml \
    TEST.MODEL_FILE _PATH_TO_CHECKPOINT_ TEST.PCKH_THRE 0.5
```

## Citations
If you use our code or models in your research, please cite with:
```
@misc{li20212d,
      title={Is 2D Heatmap Representation Even Necessary for Human Pose Estimation?}, 
      author={Yanjie Li and Sen Yang and Shoukui Zhang and Zhicheng Wang and Wankou Yang and Shu-Tao Xia and Erjin Zhou},
      year={2021},
      eprint={2107.03332},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Acknowledgement
Thanks for the open-source HRNet.
* [Deep High-Resolution Representation Learning for Human Pose Estimation, Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/)

