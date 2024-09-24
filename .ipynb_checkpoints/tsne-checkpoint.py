import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def loadData():
    # 读入数据
    # ip 与 salinas数据集同一传感器AVIRIS
    # pu与pua属于一个传感器ROSIS
    # 珠海一号OHS CMOS传感器
    if args.dataset_name=="IP":
        data = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']
    elif args.dataset_name=="PU":
        data = sio.loadmat('../data/PaviaU.mat')['paviaU']
        labels = sio.loadmat('../data/PaviaU_gt.mat')['paviaU_gt']
    elif args.dataset_name == "Salinas":
        data = sio.loadmat('../data/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('../data/Salinas_gt.mat')['salinas_gt']
    elif args.dataset_name == "PC":
        data = sio.loadmat('../data/Pavia.mat')['pavia']
        labels = sio.loadmat('../data/Pavia_gt.mat')['pavia_gt']
    elif args.dataset_name == "Coast":
        data = rearrange(tf.imread(r'/wenjiaxiang/ywz_sub/HSI_data/newtest.tif'), "d h w -> h w d")
        labels = tf.imread(r'/wenjiaxiang/ywz_sub/HSI_data/newtest_label.tif')

    return data, labels

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):


    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = args.batch

def create_data_loader():
    # 地物类别
    # class_num = 16
    # 读入数据
    X, y = loadData()
    # 用于测试样本的比例
    test_ratio = args.test_ratio
    # 每个像素周围提取 patch 的尺寸
    patch_size = args.patch
    # 使用 PCA 降维，得到主成分的数量
    pca_components = args.pca

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # Xtrain = rearrange(Xtrain,"b h w c -> b c h w")
    # Xtest = rearrange(Xtest,"b h w c -> b c h w")

    # Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)
    #
    # # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest = Xtest.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                              )

    return train_loader, test_loader, all_data_loader, y
class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len
def visualize_with_tsne(memory_output, category_index, save_path='tsne_visualization.png'):
    # 将特征转换为2D
    tsne = TSNE(n_components=2, random_state=42)
    memory_output_2d = tsne.fit_transform(memory_output.detach().cpu().numpy())

    # 可视化
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(memory_output_2d[:, 0], memory_output_2d[:, 1], c=category_index.detach().cpu().numpy(),
                          cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=list(set(category_index.detach().cpu().numpy())))
    plt.colorbar()
    plt.title('t-SNE visualization of memory output')

    # 保存图像
    plt.savefig(save_path)
    plt.show()

from SemanticViT import MyNet  # 确保导入路径正确

# 假设MyNetWithTSNE已经定义
model = MyNet(mode="train")

# 加载训练好的权重
weight_path = "/wenjiaxiang/Jiax/Semantic_de_cls/cls_semantic_de/cls_params/checkpoint.pth"
model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

# 确保模型处于评估模式
model.eval()

# 示例使用
if __name__ == "__main__":
    input = torch.randn(64, 32, 13, 13)
    target = torch.randint(0, 16, (64,))  # 修正target的生成方式
    y, memory_output, category_index = model(input, target)  # 注意这里不再需要将target转换为浮点类型

    # 可视化并保存图像
    visualize_with_tsne(memory_output, category_index)
