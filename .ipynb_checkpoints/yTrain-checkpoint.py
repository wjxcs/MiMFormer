import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import get_cls_map
import time
from SemanticViT import MyNet
# from patchbasedvit import MyNet_patch
from args_parse import args
import random
from einops import rearrange
from MI import MINE

from sklearn.manifold import TSNE

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# setup_seed(620)
# setup_seed(620)
# setup_seed(4403)
import h5py as hp

def loadData():
    # 读入数据
    # ip 与 salinas数据集同一传感器AVIRIS
    # pu与pua属于一个传感器ROSIS
    # 珠海一号OHS CMOS传感器
    if args.dataset_name=="IP":
        data = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']
    elif args.dataset_name=="Honghu":
        data = sio.loadmat('../data/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
        labels = sio.loadmat('../data/WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
    elif args.dataset_name == "Salinas":
        data = sio.loadmat('../data/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('../data/Salinas_gt.mat')['salinas_gt']
    elif args.dataset_name == "PC":
        data = sio.loadmat('../data/Pavia.mat')['pavia']
        labels = sio.loadmat('../data/Pavia_gt.mat')['pavia_gt']
    elif args.dataset_name == "KSC":
        data = sio.loadmat('../data/KSC.mat')['KSC']
        labels = sio.loadmat('../data/KSC_gt.mat')['KSC_gt']
    elif args.dataset_name == "Coast":
        data = np.array(hp.File('../data/NC16.mat')['HSI'])
        data = rearrange(data, "c h w -> h w c")
        labels = np.array(hp.File('../data/NC16.mat')['GT'])


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
    X_all = TestDS(X, y_all)
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
    all_data_loader = torch.utils.data.DataLoader(dataset=X_all,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                              )
    print("starting Training test_ratio is {}.".format(args.test_ratio))
    print("X shape is{}".format(X.shape))

    return train_loader, test_loader, all_data_loader, y,X

""" Training dataset"""

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
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Mine = MINE(in_channels_x=32, in_channels_y=32, inter_channels=4).to(device)
optimizer_mine = optim.Adam(Mine.parameters(), lr=1e-4)
ma_rate = 0.001

def mi_estimator(x, y, y_):
    joint, marginal = Mine(x, y), Mine(x, y_)
    return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal))), joint, marginal

def update_MI(inputs_a, inputs_b,  optim_1, optim_2):
    z_a = inputs_a.detach()
    z_d = inputs_b.detach()
    z_d_shuffle = torch.index_select(z_d, 0, torch.randperm(z_d.shape[0]).to(device)).clone()

    mutual_loss, _, _ = mi_estimator(z_a, z_d, z_d_shuffle)
    mutual_loss = F.elu(mutual_loss).clone()   # 避免就地操作
    mutual_loss.backward()
    optim_1.step()
    optim_2.step()

    optim_1.zero_grad()
    optim_2.zero_grad()
    optimizer_mine.zero_grad()

    return mutual_loss


def learn_mine(output_1, output_2, optim_1, optim_2, ma_rate=0.001):
    with torch.no_grad():
        z_a = output_1.detach()
        z_d = output_2.detach()

        z_d_shuffle = torch.index_select(z_d, 0, torch.randperm(z_d.shape[0]).to(device)).clone()


    et = torch.mean(torch.exp(Mine(z_a, z_d_shuffle)))
    if Mine.ma_et is None:
        Mine.ma_et = et.detach().item()
    Mine.ma_et += ma_rate * (et.detach().item() - Mine.ma_et)
    mutual_information = torch.mean(Mine(z_a, z_d)) - torch.log(et) * et.detach() / Mine.ma_et

    mi_loss = -mutual_information

    mi_loss.backward()
    optimizer_mine.step()

    optim_1.zero_grad()
    optim_2.zero_grad()
    optimizer_mine.zero_grad()

    return mutual_information

def train(train_loader, epochs):
    # torch.autograd.set_detect_anomaly(True)  # 启用异常检测
    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = MyNet(mode="train").to(device)
    # net = MyNet_patch(mode="train").to(device)

    gen_feature_optimizer = optim.Adam(net.Gen_FeatureEXM.parameters(), lr=0.001)
    clsAtt_optimizer = optim.Adam(net.clsAtt.parameters(), lr=0.001)

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # 开始训练
    total_loss = 0
    epoch_losses = []
    mi_loss = 0
    # 初始化 ParetoOptimizer
    for epoch in range(epochs):
        epoch_loss = 0
        net.train()
        for i, (data, target) in enumerate(tqdm(train_loader, desc='Epoch {}'.format(epoch+1), unit='batch', leave=False, dynamic_ncols=True)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            gen_feature_optimizer.zero_grad()
            clsAtt_optimizer.zero_grad()
            optimizer_mine.zero_grad()

            outputs, output_a, output_b = net(data, target)
            # logits, output_1, output_2 = model(inputs, target=labels)
            semantic_tokens = net.conceptT(output_b)
            memory_output, category_index, attention_scores = net.MemoryC(semantic_tokens)
            # 计算损失函数
            loss = criterion(outputs, target)
            # memory_loss = net.calculate_memory_loss(semantic_tokens, memory_output, category_index)
            # loss_sum = loss + 0.0001*memory_loss
            loss_sum = loss
            loss_sum.backward(retain_graph=True)
            optimizer.step()
            # print("hello")
            # 0.35 0.15 0.5 95.38  0.4 0.1 0.5

            # total_losses = loss + 0.0001 * memory_loss
            # 反向传播
            total_loss += loss_sum.item()
            epoch_loss += loss_sum.item()
            optimizer.zero_grad()
            gen_feature_optimizer.zero_grad()
            clsAtt_optimizer.zero_grad()
            optimizer_mine.zero_grad()
            # print(output_a.shape)
            # print(output_b.shape)

            mi_loss = update_MI(output_a, output_b, clsAtt_optimizer, gen_feature_optimizer)


            for _ in range(5):
                learn_mine(output_a, output_b,clsAtt_optimizer,gen_feature_optimizer)
                # print(learn_mi_loss)

        epoch_loss /= len(train_loader)  # 计算epoch的平均损失
        epoch_losses.append(epoch_loss)  # 将平均损失添加到列表中
        print('\n[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]   [mi loss: %.4f]' % (epoch+1,
                                                                         total_loss / (epoch+1),
                                                                         loss.item(),
                                                                         mi_loss))
    # 训练结束后，生成损失图像
    plt.figure(figsize=(10, 5))  # 创建一个新的figure
    plt.plot(epoch_losses, 'r', label='Training loss')  # 绘制损失曲线
    plt.title('Training Loss vs. Epochs')  # 添加图表标题
    plt.xlabel('Epochs')  # 添加x轴标签
    plt.ylabel('Loss')  # 添加y轴标签
    plt.legend()  # 显示图例
    plt.savefig("training_loss"+args.dataset_name+".png")  # 保存图像到文件
    plt.close()  # 关闭图表以释放内存

    print('Finished Training')

    return net, device

def ytest(device, net, test_loader,mode="01"):
    count = 0
    # 模型测试
    net.eval()
#     memorydata = net.MemoryC.memory.data
#     np.save('memory_output.npy', memorydata.detach().cpu().numpy())
#     xs = []
    targets = []
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)

        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
#             xs.append(xl.detach().cpu().numpy())
            targets.append(labels)
        else:
#             xs.append(xl.detach().cpu().numpy())
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
#     if mode=="01":
#         np.save('xs.npy', np.concatenate(xs))
#         np.save('targets.npy', np.concatenate(targets))
    return y_pred_test, y_test

from matplotlib.colors import ListedColormap, BoundaryNorm

# 自定义颜色映射函数
def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 1:
            y[index] = np.array([147, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 123]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 7:
            y[index] = np.array([118, 254, 172]) / 255.
        if item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 255, 125]) / 255.
        if item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 13:
            y[index] = np.array([0, 172, 254]) / 255.
        if item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 16:
            y[index] = np.array([101, 193, 60]) / 255.
        if item == 17:
            y[index] = np.array([255, 105, 180]) / 255.  # 粉色
        if item == 18:
            y[index] = np.array([0, 128, 128]) / 255.  # 青色
        if item == 19:
            y[index] = np.array([128, 0, 128]) / 255.  # 紫色
        if item == 20:
            y[index] = np.array([210, 105, 30]) / 255.  # 巧克力色
        if item == 21:
            y[index] = np.array([32, 178, 170]) / 255.  # 淡青色
        if item == 22:
            y[index] = np.array([0, 255, 255]) / 255.  # 青色
    return y
def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = args.target_names
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

def sample_data_by_class(X, y, n_samples_per_class=50):
    unique_classes = np.unique(y)
    sampled_X, sampled_y = [], []

    for cls in unique_classes:
        if cls == 0:
            # 跳过类别 0
            continue

        indices = np.where(y == cls)[0]
        if len(indices) >= n_samples_per_class:
            indices = np.random.choice(indices, n_samples_per_class, replace=False)
        else:
            # 如果样本少于 n_samples_per_class，则使用所有可用样本
            indices = indices
        
        sampled_X.append(X[indices])
        sampled_y.append(y[indices])
        
        # 打印调试信息
        print(f"Class {cls}: Sampled {len(indices)} samples")

    # 将所有类别的数据和标签合并为一个数组
    sampled_X = np.concatenate(sampled_X, axis=0)
    sampled_y = np.concatenate(sampled_y, axis=0)

    return sampled_X, sampled_y

def tsne_visualization(X, y, n_samples_per_class=50, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
    X_sampled, y_sampled = sample_data_by_class(X, y, n_samples_per_class)
    
    # 进行 t-SNE 降维
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X_sampled)

    # 打印维度信息
    print(f"X_tsne shape: {X_tsne.shape}")
    print(f"y_sampled shape: {y_sampled.shape}")

    plt.figure(figsize=(8, 8))
    unique_classes = np.unique(y_sampled)
    for cls in unique_classes:
        # 绘制每个类别的点
        plt.scatter(X_tsne[y_sampled == cls, 0], X_tsne[y_sampled == cls, 1], label=f'Class {cls}', s=5)
    
    plt.legend()
    plt.title("t-SNE Visualization")
    plt.savefig("TSNE_" + args.dataset_name + "ss.png", dpi=300, format='png', bbox_inches='tight', pad_inches=0)



# def sample_data_by_class(X, y, n_samples_per_class=50):
#     unique_classes = np.unique(y)
#     sampled_X, sampled_y = [], []
    
#     for cls in unique_classes:
#         indices = np.where(y == cls)[0]
#         if len(indices) > n_samples_per_class:
#             indices = np.random.choice(indices, n_samples_per_class, replace=False)
        
#         sampled_X.append(X[indices])
#         sampled_y.append(y[indices])
    
#     return np.vstack(sampled_X), np.hstack(sampled_y)

# def tsne_visualization(X, y, n_samples_per_class=50, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
#     X_sampled, y_sampled = sample_data_by_class(X, y, n_samples_per_class)
#     tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
#     X_tsne = tsne.fit_transform(X_sampled)

#     plt.figure(figsize=(8, 8))
#     unique_classes = np.unique(y_sampled)
#     for cls in unique_classes:
#         plt.scatter(X_tsne[y_sampled == cls, 0], X_tsne[y_sampled == cls, 1], label=f'Class {cls}', s=5)
    
#     plt.legend()
#     plt.title("t-SNE Visualization")
#     plt.savefig("TSNE_" + args.dataset_name + "ss.png", dpi=300, format='png', bbox_inches='tight', pad_inches=0)
#     plt.show
    
def count_samples_per_class(y):
    unique_classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    return class_counts
if __name__ == '__main__':

    train_loader, test_loader, all_data_loader, y_all, Xtest = create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=args.epochs)

    # 只保存模型参数
    torch.save(net.state_dict(), args.model_pth)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    net = MyNet(mode="test").to(device)
    net.load_state_dict(torch.load(args.model_pth))

    y_pred_test, y_test = ytest(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    file_name = args.cls_report + args.dataset_name + ".txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    print(file_name)
    get_cls_map.get_cls_map(net, device, all_data_loader, y_all)
#    清空畫布
    plt.clf()
    testSizeNumber = args.testSizeNumber
    number = len(y_all)//testSizeNumber
    features = []
    # pred_y = []
#     labels = y_pred_test.data
    labels = y_all
    
    labels = np.array(labels)
#     labels = labels + 1
    class_counts = count_samples_per_class(labels)
    print("Samples per class:", class_counts)

    print(np.min(labels))
    print(np.max(labels))
    # print(pred_y)
    # print(len(pred_y), np.min(pred_y), np.max(pred_y))
    for i in range(number):
        temp = Xtest[i * testSizeNumber:(i + 1) * testSizeNumber]
        temp = torch.tensor(temp)
        temp = temp.to(dtype=torch.float)
        temp = temp.cuda()
        temp2 = net(temp)
        fea = temp2
        features.append(fea.cpu().detach().numpy())
        del temp, temp2

    if (i + 1) * testSizeNumber < len(y_test):
        temp = Xtest[(i + 1) * testSizeNumber:len(y_all)]
        temp = torch.tensor(temp)
        temp = temp.to(dtype=torch.float)
        temp = temp.cuda()
        temp2 = net(temp)
        fea = temp2
        features.append(fea.cpu().detach().numpy())
        del temp, temp2


    # Assuming `features` and `labels` are already defined
    features = np.concatenate(features)
    print("features = ", features.shape)
    Classes = args.cls
#     tsne_visualization(features, labels, n_samples_per_class=100)
    # Perform t-SNE
    embeddings = TSNE(n_components=2, n_jobs=24).fit_transform(features)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    # 自定义颜色列表，基于 `Classes` 的数量
#     自定义颜色列表，基于 `Classes` 的数量
    color_indices = np.arange(1, Classes + 1)
    colors = list_to_colormap(color_indices)
    colors = list_to_colormap(color_indices)
    cmap = ListedColormap(colors)

    # Define discrete colormap with BoundaryNorm
    norm = BoundaryNorm(np.arange(1, Classes + 2), cmap.N)

    # 调整图像尺寸
    plt.figure(figsize=(12, 10))  # 增大图像尺寸
    print("vis_x shape is {}".format(vis_x.size))
    # Scatter plot with custom colormap
    fig = plt.scatter(vis_x, vis_y, c=labels + 1, cmap=cmap, norm=norm, marker='.') 
        
    # Create a colorbar with discrete labels
    cbar = plt.colorbar(fig, ticks=np.arange(1, Classes + 1))
    print("features ")
    # 调整colorbar的字体大小
    cbar.ax.tick_params(labelsize=12)  # 将字体大小调整为12

    # Hide axes
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    # Save the figure
    plt.savefig("TSNE_" + args.dataset_name + ".png", dpi=300, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig("TSNE_" + args.dataset_name + ".eps", dpi=300, format='eps', bbox_inches='tight', pad_inches=0)
    plt.clf()
