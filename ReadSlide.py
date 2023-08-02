import openslide as ol
slide = ol.OpenSlide('/home/omnisky/hdd_15T_sdd/TCGA-06-0213-01Z-00-DX2.5968b29e-a689-4d8a-80b0-e0ffff8bcec6.svs')
print(slide.dimensions) #(33576, 24479)
print(slide.level_downsamples) #(1.0, 4.00024513809446, 16.006811744468713)
print(slide.level_dimensions) #((33576, 24479), (8394, 6119), (2098, 1529))
print(slide.level_count)
print(slide.properties)
print(slide.associated_images)
print(slide.properties['aperio.AppMag'])

# (33576, 24479)
# (1.0, 4.00024513809446, 16.006811744468713)
# ((33576, 24479), (8394, 6119), (2098, 1529))
# 3
# <_PropertyMap {'aperio.AppMag': '20', 'aperio.DSR ID': 'AP1258-DSR', 'aperio.Date': '12/05/08', 'aperio.Filename': '6930', 
#                'aperio.Filtered': '5', 'aperio.ICC Profile': 'ScanScope v1', 'aperio.ImageID': '6930', 'aperio.Left': '31.918922', 
#                'aperio.LineAreaXOffset': '0.000000', 'aperio.LineAreaYOffset': '0.000000', 'aperio.LineCameraSkew': '-0.000389', 
#                'aperio.MPP': '0.5040', 'aperio.OriginalWidth': '35000', 'aperio.Originalheight': '24579', 'aperio.ScanScope ID': 'SS1302',
#                'aperio.StripeWidth': '1000', 'aperio.Time': '11:58:59', 'aperio.Title': 'none', 'aperio.Top': '21.809845', 
#                'aperio.User': '8e32aa25-a625-4f07-94b5-6228c29a3733', 'openslide.comment': 'Aperio Image Library v9.0.22\r\n35000x24579 [0,100 33576x24479] (240x240) JPEG/RGB Q=70|AppMag = 20|StripeWidth = 1000|ScanScope ID = SS1302|Filename = 6930|Title = none|Date = 12/05/08|Time = 11:58:59|User = 8e32aa25-a625-4f07-94b5-6228c29a3733|MPP = 0.5040|Left = 31.918922|Top = 21.809845|LineCameraSkew = -0.000389|LineAreaXOffset = 0.000000|LineAreaYOffset = 0.000000|DSR ID = AP1258-DSR|ImageID = 6930|OriginalWidth = 35000|Originalheight = 24579|Filtered = 5|ICC Profile = ScanScope v1', 'openslide.level-count': '3', 'openslide.level[0].downsample': '1', 'openslide.level[0].height': '24479', 'openslide.level[0].tile-height': '240', 'openslide.level[0].tile-width': '240', 'openslide.level[0].width': '33576', 'openslide.level[1].downsample': '4.0002451380944599', 'openslide.level[1].height': '6119', 'openslide.level[1].tile-height': '240', 'openslide.level[1].tile-width': '240', 'openslide.level[1].width': '8394', 'openslide.level[2].downsample': '16.006811744468713', 'openslide.level[2].height': '1529', 'openslide.level[2].tile-height': '240', 'openslide.level[2].tile-width': '240', 'openslide.level[2].width': '2098', 'openslide.mpp-x': '0.504', 'openslide.mpp-y': '0.504', 'openslide.objective-power': '20', 'openslide.quickhash-1': 'd468982d6c5b723f7870113ed9da8b8f4eafe52d3dcb456cf018644c7f45c329', 'openslide.vendor': 'aperio', 'tiff.ImageDescription': 'Aperio Image Library v9.0.22\r\n35000x24579 [0,100 33576x24479] (240x240) JPEG/RGB Q=70|AppMag = 20|StripeWidth = 1000|ScanScope ID = SS1302|Filename = 6930|Title = none|Date = 12/05/08|Time = 11:58:59|User = 8e32aa25-a625-4f07-94b5-6228c29a3733|MPP = 0.5040|Left = 31.918922|Top = 21.809845|LineCameraSkew = -0.000389|LineAreaXOffset = 0.000000|LineAreaYOffset = 0.000000|DSR ID = AP1258-DSR|ImageID = 6930|OriginalWidth = 35000|Originalheight = 24579|Filtered = 5|ICC Profile = ScanScope v1', 'tiff.ResolutionUnit': 'inch'}>
# <_AssociatedImageMap {'thumbnail': <PIL.Image.Image image mode=RGBA size=1024x746 at 0x7F1C1F93EEF0>}>
# 20



from openslide import OpenSlide

slide = OpenSlide('path/to/slide/file')
# 访问图像宽度和高度
width, height = slide.dimensions

# 访问图像的层级数量
level_count = slide.level_count

# 访问第0层图像的下采样因子
downsample = slide.level_downsamples[0]

# 访问图像的制造商和快速哈希值
vendor = slide.properties['openslide.vendor']
quickhash1 = slide.properties['openslide.quickhash-1']

# 访问图像的所有属性
properties = slide.properties

# 访问第0层图像的全部像素
image = slide.read_region((0, 0), 0, slide.level_dimensions[0])

# 访问第1层图像(下采样因子为4)的指定区域像素
region = slide.read_region((x, y), 1, (w, h))


import openslide

# 打开SVS文件
slide = openslide.OpenSlide('example.svs')

# 获取level 3的倍率和大小
level = 3
downsample = slide.level_downsamples[level]
level_size = slide.level_dimensions[level]

# # 计算在level 0中的坐标和区域大小
# x, y = int(120 * downsample), int(120 * downsample)
# w, h = int(224 * downsample), int(224 * downsample)

# 读取区域
region = slide.read_region((x, y), level, (w, h))

# 显示图像
region.show()

