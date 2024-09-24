from transformers import BertModel, BertTokenizer
import torch
from Memory import FeaturesMemory  # 确保FeaturesMemory类已正确定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from einops import rearrange,repeat
# 定义类别名称
target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']
def init_memory():

    # 初始化分词器和模型
    tokenizer = BertTokenizer.from_pretrained(r'/wenjiaxiang/Jiax/Semantic_de_cls/cls_semantic_de/bert')
    model = BertModel.from_pretrained(r'/wenjiaxiang/Jiax/Semantic_de_cls/cls_semantic_de/bert')

    # 定义FeaturesMemory实例所需的参数
    num_classes = len(target_names)  # 类别数量
    feats_channels = 1024  # 假设BERT输出的维度是1024
    transform_channels = 1024  # 调整为BERT输出的维度
    out_channels = 256
    num_feats_per_cls=4
    # 假设FeaturesMemory的构造函数接受以下参数
    memory_model = FeaturesMemory(num_classes, feats_channels,transform_channels,out_channels,num_feats_per_cls)

    # 对每个类别名称进行编码并获取BERT输出
    encoded_texts = tokenizer(target_names, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_texts)
        last_hidden_states = outputs.last_hidden_state
    # print(encoded_texts.shape)
    # for i in num_feats_per_cls:
    #     memory_model.memory.data.copy_()
    # 提取[CLS]特征




    cls_features = last_hidden_states[:, 0, :]  # (num_classes, feats_channels)

    # 调整cls_features的形状以匹配memory_model.memory的形状
    cls_features = rearrange(cls_features, 'c n -> c 1 n')  # 假设n是features的维度

    # 现在循环遍历每个类别和每个特征索引，并将cls_features的值赋给memory_model.memory
    for clsid in range(num_classes):
        for idx in range(num_feats_per_cls):
            # 使用rearrange调整形状以匹配memory_model.memory[clsid][idx].data的形状
            memoru_init = rearrange(cls_features[clsid], "c n -> (c n)")
            memoru_init = repeat(memoru_init, "n -> d n",d=4)
            print(memoru_init.shape)
            # 确保memoru_init的形状与memory_model.memory[clsid][idx].data的形状一致
            memory_model.memory[clsid].data.copy_(memoru_init)

    # 打印memory_model.memory.data的形状，验证赋值是否正确
    print(f"Shape of memory_model.memory.data: {memory_model.memory.data.shape}")

    # 检查cls_features和memory_model.memory.data是否相等
if __name__ =="__main__":

    init_memory()