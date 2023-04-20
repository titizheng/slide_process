# Extracting tissue areas on slides without precise annotation
##  

## Directory Set-Up

```bash
Slideprocess/
   ├── Slideprodata.py
   ├── slide
         ├─── Normal
         ├─── Tumor
   ├──output
         ├─── pth
         ├─── stitch_images
         ├─── vis_images
```

## Reference
reference [CLAM](https://github.com/mahmoodlab/CLAM)



## openslide
在OpenSlide库中，常用的属性包括：

OpenSlide中的properties属性指的是图像的元数据，包括图像的大小、物理分辨率、扫描仪的制造商、扫描日期等信息。这些元数据通常是从扫描仪设备中提取出来的，可以用于描述图像的属性和特征，帮助人们更好地理解和处理图像数据.

使用OpenSlide.properties['name']

openslide.level-count：图像的层级数量。

openslide.level[0].width：第一层图像的宽度（像素数）。

openslide.level[0].height：第一层图像的高度（像素数）。

openslide.level[0].downsample：第一层图像的下采样因子。

openslide.level[x].width：第x层图像的宽度（像素数）。

openslide.level[x].height：第x层图像的高度（像素数）。

openslide.level[x].downsample：第x层图像的下采样因子。

openslide.vendor：扫描仪的制造商。

openslide.quickhash-1：第一种快速哈希值。

openslide.mpp-x：图像的水平方向物理分辨率（每像素毫米数）。

openslide.mpp-y：图像的垂直方向物理分辨率（每像素毫米数）。

openslide.objective-power：扫描镜头的物理放大倍数。

其中，level表示图像的层级，从0开始，依次递增；downsample表示下采样因子，即当前层级相对于原始图像的缩放比例；mpp-x和mpp-y表示图像的水平和垂直方向物理分辨率，单位为毫米每像素；objective-power表示扫描镜头的物理放大倍数。除了这些常用属性外，OpenSlide还支持其他一些元数据属性，可以使用openslide_get_property_names()函数获取所有属性名称。

OpenSlide.name

width, height = slide.dimensions # 访问图像宽度和高度
level_count = slide.level_count # 访问图像的层级数量
downsample = slide.level_downsamples[0] # 访问第0层图像的下采样因子

'aperio.AppMag'是OpenSlide库中一个元数据属性，它表示扫描镜头的物理放大倍数。'aperio.AppMag'代表着APERIO扫描仪的物理放大倍数，是一个实际值而非计算值，通常以整数表示，例如 40、 20 等。在OpenSlide库中，可以使用openslide_get_property_value()函数来获取该属性的值。

## OpenSlide库需

安装OpenSlide库
在Linux系统下，可以使用以下命令安装OpenSlide库：


sudo apt-get install openslide-tools
sudo apt-get install python-openslide
在Windows系统下，可以从OpenSlide官网（https://openslide.org/download/）下载相应版本的OpenSlide二进制文件并安装。
