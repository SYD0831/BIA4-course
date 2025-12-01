import os
import shutil
import random
import torch
import cv2
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

# 关键：从你的项目文件中导入所需的类和函数
# 脚本需要能找到 'model' 和 'data' 目录
from model.standard_net import StandardNetLightning
from data.mpidb_dataset import to7ch  # 这是核心的 7 通道转换函数

def preprocess_single_image(image_path: str, image_size: int = 100):
    """
    加载并预处理单张新图像，使其符合模型输入要求。
    此函数完整复刻了训练时的数据处理流程。

    Args:
        image_path (str): 待处理图像的文件路径。
        image_size (int): 模型输入的目标图像尺寸。

    Returns:
        torch.Tensor: 处理完成的、可直接输入模型的张量。
    """
    # 1. 使用 OpenCV 读取图像 (格式为 BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"错误: 无法读取图像文件 '{image_path}'。请检查路径是否正确。")
    # 2. 缩放图像到模型指定的尺寸 (100x100)
    img_resized = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    # 3. 将 BGR 图像转换为 7 通道表示 (项目的核心特征工程)
    img_7ch = to7ch(img_resized)

    # 4. 转换为 PyTorch 张量
    # ToTensorV2 会自动处理:
    #   - 维度转换: (H, W, C) -> (C, H, W)
    #   - 数据类型转换: numpy.ndarray -> torch.float32
    #   - 像素值归一化: [0, 255] -> [0.0, 1.0]
    tensor_converter = ToTensorV2()
    tensor = tensor_converter(image=img_7ch)['image']

    # 5. 增加批次维度 (Batch Dimension)
    # 模型期望的输入形状是 (N, C, H, W)，其中 N 是批次大小。
    # 对于单张图片推理，N=1。
    # [7, 100, 100] -> [1, 7, 100, 100]
    tensor = tensor.unsqueeze(0)

    return tensor

def predict(model_path: str, image_path: str):
    """
    加载模型并对单张图片进行预测。

    Args:
        model_path (str): 训练好的 .ckpt 模型的路径。
        image_path (str): 需要预测的图片的路径。
    """
    print("--- 开始预测 ---")
    
    # 1. 加载模型
    print(f"1. 正在从 '{model_path}' 加载模型...")
    try:
        model = StandardNetLightning.load_from_checkpoint(model_path)
    except FileNotFoundError:
        print(f"错误: 模型文件 '{model_path}' 未找到。请检查路径。")
        return

    # 将模型设置为评估模式（非常重要！）
    model.eval()

    # 检测并使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   模型已加载，并转移到设备: {device.type.upper()}")

    # 2. 预处理输入图像
    print(f"2. 正在预处理图像 '{image_path}'...")
    try:
        input_tensor = preprocess_single_image(image_path)
        input_tensor = input_tensor.to(device) # 将输入数据也转移到相应设备
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"图像处理过程中发生错误: {e}")
        return
    print("   图像预处理完成。输入张量形状:", input_tensor.shape)

    # 3. 执行推理
    print("3. 正在执行模型推理...")
    # 关闭梯度计算，以加速并节省内存
    with torch.no_grad():
        logits = model(input_tensor)
    
    # 4. 解析并输出结果
    # 将 Logits 转换为概率
    probabilities = torch.softmax(logits, dim=1).squeeze() # squeeze() 移除批次维度
    
    # 获取预测的类别索引
    predicted_index = torch.argmax(probabilities).item()

    # 类别名称映射 (需要与训练时的数据集标签顺序一致)
    class_map = {0: 'falciparum', 1: 'ovale', 2: 'vivax'}
    predicted_class_name = class_map.get(predicted_index, "未知类别")
    
    confidence = probabilities[predicted_index].item()

    print("\n--- 预测结果 ---")
    print(f"预测类别: {predicted_class_name.upper()}")
    print(f"置信度: {confidence:.2%}")
    print("\n详细概率分布:")
    for i, prob in enumerate(probabilities):
        class_name = class_map.get(i, f"类别 {i}")
        print(f"  - {class_name:<12}: {prob.item():.2%}")
        
def predict_batch(model_path: str, image_paths: list[Path]) -> pd.DataFrame:
    """
    加载模型并对一批图片进行预测，将结果返回为 DataFrame。

    Args:
        model_path (str): 训练好的 .ckpt 模型的路径。
        image_paths (list[Path]): 需要预测的图片路径列表。

    Returns:
        pd.DataFrame: 包含预测结果的 DataFrame。
    """
    print("--- 开始批量预测 ---")
    
    # 1. 加载模型 (仅加载一次)
    print(f"1. 正在从 '{model_path}' 加载模型...")
    try:
        model = StandardNetLightning.load_from_checkpoint(model_path)
    except FileNotFoundError:
        print(f"错误: 模型文件 '{model_path}' 未找到。请检查路径。")
        return pd.DataFrame() # 返回空的 DataFrame

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   模型已加载，并转移到设备: {device.type.upper()}")

    # 类别名称映射
    class_map = {0: 'falciparum', 1: 'ovale', 2: 'vivax'}
    
    # 存储所有预测结果的列表
    results = []

    # 2. 遍历图片列表进行预测
    print(f"2. 正在处理 {len(image_paths)} 张图片...")
    with torch.no_grad(): # 在循环外使用 no_grad，效率更高
        for img_path in tqdm(image_paths, desc="批量预测", unit="张"):
            # 预处理单张图像
            input_tensor = preprocess_single_image(str(img_path))
            
            # 如果图片读取失败，则跳过
            if input_tensor is None:
                print(f"警告: 无法读取或处理图片 '{img_path.name}'，已跳过。")
                results.append({
                    'filename': img_path.name,
                    'predicted_class': 'Error',
                    'confidence': 0.0,
                    'prob_falciparum': 0.0,
                    'prob_ovale': 0.0,
                    'prob_vivax': 0.0
                })
                continue
            
            input_tensor = input_tensor.to(device)

            # 执行推理
            logits = model(input_tensor)
            
            # 解析结果
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu()
            predicted_index = torch.argmax(probabilities).item()
            predicted_class_name = class_map.get(predicted_index, "未知类别")
            confidence = probabilities[predicted_index].item()
            
            # 将结果存入字典
            result_dict = {
                'filename': img_path.name,
                'predicted_class': predicted_class_name,
                'confidence': confidence,
                'prob_falciparum': probabilities[0].item(),
                'prob_ovale': probabilities[1].item(),
                'prob_vivax': probabilities[2].item()
            }
            results.append(result_dict)

    # 3. 将结果列表转换为 DataFrame
    print("3. 预测完成，正在生成结果 DataFrame...")
    results_df = pd.DataFrame(results)
    
    return results_df
    
# if __name__ == "__main__":
#     # 设置命令行参数解析器
#     parser = argparse.ArgumentParser(description="使用训练好的模型对疟原虫图像进行分类。")
#     parser.add_argument(
#         '-c', '--checkpoint', 
#         type=str, 
#         required=True, 
#         help="训练好的模型权重文件 (.ckpt) 的路径。"
#     )
#     parser.add_argument(
#         '-i', '--image', 
#         type=str, 
#         required=True, 
#         help="需要进行预测的单张图像的路径。"
#     )
    
#     args = parser.parse_args()
    
#     predict(model_path=args.checkpoint, image_path=args.image)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练好的模型对疟原虫图像进行批量分类。")
    parser.add_argument(
        '-c', '--checkpoint', 
        type=str, 
        required=True, 
        help="训练好的模型权重文件 (.ckpt) 的路径。"
    )
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True, 
        help="包含待预测图片的文件夹路径。"
    )
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        default="prediction_results.csv",
        help="保存预测结果的 CSV 文件路径。 (默认: prediction_results.csv)"
    )
    
    args = parser.parse_args()
    
    # 检查输入路径是文件夹
    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"错误: 输入路径 '{args.input}' 不是一个有效的文件夹。")
    else:
        # 收集文件夹下所有图片文件
        print(f"正在从文件夹 '{input_path}' 收集图片...")
        image_files = list(input_path.glob('*.jpg')) + \
                      list(input_path.glob('*.jpeg')) + \
                      list(input_path.glob('*.png'))
        
        if not image_files:
            print("错误: 在指定文件夹中没有找到任何图片文件 (.jpg, .jpeg, .png)。")
        else:
            # 调用批量预测函数
            results_dataframe = predict_batch(model_path=args.checkpoint, image_paths=image_files)

            if not results_dataframe.empty:
                # 显示 DataFrame 的前几行
                print("\n--- 预测结果预览 ---")
                print(results_dataframe.head())

                # 保存 DataFrame 到 CSV 文件
                try:
                    results_dataframe.to_csv(args.output_csv, index=False)
                    print(f"\n✅ 结果已成功保存到: {args.output_csv}")
                except Exception as e:
                    print(f"\n❌ 保存 CSV 文件失败: {e}")
