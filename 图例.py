import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.image as mpimg

# 加载PNG图像
png_1 = mpimg.imread(r"E:\jiax论文\code\Semantic_de_cls\cls_semantic_de\classification_maps\IP.png")
png_gt = mpimg.imread(r"E:\jiax论文\code\Semantic_de_cls\cls_semantic_de\classification_maps\IP_gt.png")

# 定义类别和对应的颜色
honghu_categories = ['Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 'Rape',
              'Chinese cabbage', 'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis',
              'Brassica chinensis', 'Small brassica chinensis', 'Lactuca sativa', 'Celtuce',
              'Film covered lettuce', 'Romaine lettuce', 'Carrot', 'White radish',
              'Garlic sprout', 'Broad bean', 'Tree']

ip_categories = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn','Grass-pasture',
                     'Grass-trees', 'Grass-pasture-mowed','Hay-windrowed','Oats',
                     'Soybean-notill', 'Soybean-mintill','Soybean-clean', 'Wheat', 'Woods',
                     'Buildings-Grass-Trees-Drives','Stone-Steel-Towers']
sv_categories = ["Brocoli_green_weeds_1","Brocoli_green_weeds_2","Fallow","Fallow_rough_plow",
                    "Fallow_smooth","Stubble","Celery","Grapes_untrained","Soil_vinyard_develop",
                     "Corn_senesced_green_weeds","Lettuce_romaine_4wk","Lettuce_romaine_5wk","Lettuce_romaine_6wk",
                     "Lettuce_romaine_7wk","Vinyard untrained","Vinyard _vertical_trellis"]
coast_categories = ['Suaeda glauca', 'Cement road', 'Asphalt road','Water', 'Stone',
                     'Hay','lron', 'Tamarix', 'Withered reed','Reed','Spartina alterniflora'
                     "Moss","Dry soil","Moist soil","Mudflat",'Standard refectance cloth']
# 定义颜色（RGB格式并归一化）
colors_rgb = [
    [147, 67, 46], [0, 0, 255], [255, 100, 0], [0, 255, 123],
    [164, 75, 155], [101, 174, 255], [118, 254, 172], [60, 91, 112],
    [255, 255, 0], [255, 255, 125], [255, 0, 255], [100, 0, 255], [0, 172, 254],
    [0, 255, 0], [171, 175, 80], [101, 193, 60], [255, 105, 180],  [0, 128, 128],
    [128, 0, 128], [210, 105, 30], [32, 178, 170], [0, 255, 255]]
colors_rgb_16 = [
    [147, 67, 46], [0, 0, 255], [255, 100, 0], [0, 255, 123],
    [164, 75, 155], [101, 174, 255], [118, 254, 172], [60, 91, 112],
    [255, 255, 0], [255, 255, 125], [255, 0, 255], [100, 0, 255], [0, 172, 254],
    [0, 255, 0], [171, 175, 80], [101, 193, 60]]

# 将 RGB 颜色转换为 matplotlib 可以识别的格式
colors_normalized = [np.array(color) / 255. for color in colors_rgb]

colors_normalized_16 = [np.array(color) / 255. for color in colors_rgb_16]

# 使用香港湖类目或其他类目（示例）
categories = sv_categories

# 创建图例
patches = [mpatches.Patch(color=colors_normalized_16[i], label=categories[i]) for i in range(len(categories))]
# patches = [mpatches.Patch(color=colors_normalized[i], label=categories[i]) for i in range(len(categories))]

# 获取图像的尺寸
fig, ax = plt.subplots()
ax.imshow(png_1)
fig.canvas.draw()
width, height = fig.get_size_inches() * fig.get_dpi()
plt.close(fig)

# 创建图例图像，设置大小与子图相同
legend_fig = plt.figure(figsize=(width/100, height/100), facecolor='white')
legend_ax = legend_fig.add_subplot(111)
# legend_ax.legend(handles=patches, loc='center', ncol=1, prop={'weight': "bold", "size": 18}, frameon=False)
legend_ax.legend(handles=patches, loc='center', ncol=1, prop={"size": 18}, frameon=False)
legend_ax.axis('off')

# 保存图例图像
# legend_fig.savefig("output/ip_legend_image.png", bbox_inches='tight', pad_inches=0, facecolor='white')
# legend_fig.savefig("output/sv_legend_image.png", bbox_inches='tight', pad_inches=0, facecolor='white')
legend_fig.savefig("output/sv_legend_image.eps", bbox_inches='tight', pad_inches=0, facecolor='white')
# legend_fig.savefig("output/honghu_legend_image.png", bbox_inches='tight', pad_inches=0, facecolor='white')

plt.close(legend_fig)
