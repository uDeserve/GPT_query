import os

import datetime
import openai
import base64
from openai import OpenAI
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np

# 初始化固定随机种子
#RANDOM_SEED = 42  # 全局固定随机种子
#random.seed(RANDOM_SEED)

begin_num = 1  # 开始subject编号
end_num = 3  # 结束subject编号
dataset_dir = "CHS_way1"  # 数据集文件夹路径

def count_ses_num(idx, dataset_format):
    # 根据similarity编号计算sub_num\epoch_num\state_num编号
    for i in range(len(dataset_format)):
        # i从0开始，到len(dataset_format)-1结束
        idx

#def similarity_idx_to_base64(idx):
#    # 将相似度矩阵索引所指向的样本图像转换为 Base64 编码字符串
#    image_path = os.path.join(dataset_dir, f"subject{sub_num}_epoch{epoch_num}_{state_num}.0.png")

# ------------------ 新增预处理部分 ------------------

# 读取当前运行次数
run_count_file = "run_count.txt"
if os.path.exists(run_count_file):
    with open(run_count_file, "r") as f:
        run_count = int(f.read().strip())  # 读取现有的运行次数
else:
    run_count = 0  # 如果文件不存在，则从0开始

# 定义一个列表来存储所有测试结果
all_results = []

for num_i in range(begin_num, end_num):
    run_count += 1  # 增加运行次数
    output_filename = f'output_run_{run_count}.txt'
    # 将当前运行次数写回文件
    with open(run_count_file, "w") as f:
        f.write(str(run_count))

    # 定义API密钥等配置
    model_id = "gpt-4o"
    prompt_path = "english_prompt_5.txt"
    image_directory = os.path.join(dataset_dir, f"images_dx_sub{num_i}")
    epoch_quantities = -1
    if_use_fewshot = True  # 启用few-shot学习

    # 定义函数将图像文件转换为 Base64 编码字符串
    def image_to_base64(image_path):
        # 读取图像文件并将其转换为 Base64 编码字符串
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # 设置API密钥
    client = OpenAI(
        api_key="your_api_key_here"
    )
    # 读取文本内容
    with open(prompt_path, "r", encoding="utf-8") as text_file:
        text_content = text_file.read().strip()
    # 预定义的少量示例图片
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if epoch_quantities != -1: image_files = image_files[:epoch_quantities]  # 限制每次运行的图片数量
    file_name_info = {}
    output_info = {}
    results = []
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    # 依次处理每个图像文件
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        img_b64_str = image_to_base64(image_path)
        img_type = "image/jpeg" if image_file.endswith(".jpg") or image_file.endswith(".jpeg") else "image/png"
        # 获取文件名
        file_name = os.path.basename(image_path)

        # 检查文件名中是否包含“left”或“right”
        if "0.0" in file_name.lower():
            file_name_info[file_name] = 0
        elif "1.0" in file_name.lower():
            file_name_info[file_name] = 1
        else:
            file_name_info[file_name] = -1

        # 从文件夹中选择样本
        if if_use_fewshot:
            epoch_y = image_file.split('_')[1].replace('epoch', '')
            fewshot_directory = os.path.join("AL_method2", f"sub{num_i}", f"epoch_{epoch_y}")
            few_shot_images = []
            img_types = ['.png', '.jpg', '.jpeg']
            for img_type in img_types:
                img_files = [f for f in os.listdir(fewshot_directory) if f.endswith(img_type)]
                img_files.sort()
                for i in range(4):
                    if i < len(img_files):
                        img_path = os.path.join(fewshot_directory, img_files[i])
                        few_shot_images.append(img_path)
                        if len(few_shot_images) == 4:
                            break

            if len(few_shot_images) < 4:
                print(f"Few-shot images not found for {image_file}. Skipping few-shot learning.")
                few_shot_images = []
            else:
                # 转换为Base64
                img_fewshot_b64 = [image_to_base64(img) for img in few_shot_images]
                img_fewshot_1_b64_str, img_fewshot_2_b64_str, img_fewshot_3_b64_str, img_fewshot_4_b64_str = img_fewshot_b64

            if few_shot_images:
                response = client.chat.completions.create(
                    model=model_id,

                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text_content},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_fewshot_1_b64_str}"},
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_fewshot_2_b64_str}"},
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_fewshot_3_b64_str}"},
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_fewshot_4_b64_str}"},
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_b64_str}"},
                                },
                            ],
                        }
                    ],
                )
            else:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text_content},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                                },
                            ],
                        }
                    ],
                )

        # 打印响应
        response_content = response.choices[0].message.content
        # 在response内容中查找“T”和“F”出现的索引
        positive_index = response_content.rfind("T")
        negative_index = response_content.rfind("F")

        # 判断哪个字的索引更小，更新output_info中的key值
        if positive_index != -1 and (negative_index == -1 or positive_index < negative_index):
            output_info[file_name] = 0
        elif negative_index != -1:
            output_info[file_name] = 1
        else:
            output_info[file_name] = -1  # 如果都没找到，标记为unknown

        # 比较output_info和file_name_info
        if output_info.get(file_name) == file_name_info[file_name]:
            if output_info[file_name] == 0:
                TN += 1
            elif output_info[file_name] == 1:
                TP += 1
            results.append(True)  # 结果正确
        else:
            if output_info[file_name] == 0:
                FP += 1
            elif output_info[file_name] == 1:
                FN += 1
            results.append(False)  # 结果错误

        # 写入到文件
        with open(output_filename, 'a') as f:  # 'a'模式是追加写入
            fewshot_filenames = ", ".join(os.path.basename(img) for img in few_shot_images)  # 获取few-shot的文件名
            output_content = (
                #时间
                f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                # 运行信息
                f"Run Count: {run_count}\n"
                f"Model: {model_id}\n"
                f"Subject: {num_i}\n"
                f"If_use_fewshot: {if_use_fewshot}\n"
                f"Prompt_path: {prompt_path}\n"
                f"Image_directory: {image_directory}\n"
                f"Image: {image_file}\n"
                f"使用的few-shot文件名: {fewshot_filenames}\n\n"  # 添加few-shot文件名
                f"Query: {text_content},\n"
                f"Response: {response_content}\n"
                f"Image Type: {file_name_info[file_name]}\n"
                f"GPT_judge: {output_info[file_name]}\n"
                f"Judgment results: {results[len(results) - 1]}\n"
            )
            # 打印响应到控制台
            print(output_content)

            # 同时写入到文件
            f.write(output_content)
            f.write("\n")

    # 计算正确率
    correct_count = sum(results)  # 计算True的数量
    total_queries = len(results)  # 查询总数
    accuracy = correct_count / total_queries if total_queries > 0 else 0  # 防止除以零
    # 计算其他指标
    common_keys = [key for key in file_name_info.keys() if
                   key in output_info and file_name_info[key] != -1 and output_info[key] != -1]

    y_true = [file_name_info[key] for key in common_keys]
    y_pred = [output_info[key] for key in common_keys]
    y_score = [output_info[key] for key in common_keys]

    if len(y_true) > 0:
        auc = roc_auc_score(y_true, y_score)
        f1 = f1_score(y_true, y_pred, pos_label=1, average='binary')
        bca = balanced_accuracy_score(y_true, y_pred)
        sen = recall_score(y_true, y_pred, pos_label=1)
        spec = recall_score(y_true, y_pred, pos_label=0)
    else:
        auc = f1 = bca = sen = spec = 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 输出所有结果
    with open(output_filename, 'a') as f:  # 'a'模式是追加写入
        f.write(f"查询正确率: {accuracy * 100:.2f}%\n")
        f.write(f"AUC: {auc:.2f}\n")
        f.write(f"F1: {f1:.2f}\n")
        f.write(f"平衡准确率(BCA): {bca * 100:.2f}%\n")
        f.write(f"灵敏度(Sen): {sen * 100:.2f}%\n")
        f.write(f"特异性(Spec): {spec * 100:.2f}%\n")
        f.write(f"查准率(Precision): {precision * 100:.2f}%\n")
        f.write(f"查全率(Recall): {recall * 100:.2f}%\n")
        f.write(f"TP: {TP}\n")
        f.write(f"FP: {FP}\n")
        f.write(f"FN: {FN}\n")
        f.write(f"TN: {TN}\n")
        f.write(f"总计图片数量: {len(image_files)}\n")
        f.write(f"总计查询数量: {total_queries}\n")
        f.write(f"总计正确数量: {correct_count}\n")
        f.write(f"总计错误数量: {total_queries - correct_count}\n")

    print(f"查询正确率: {accuracy * 100:.2f}%")
    print(f"AUC: {auc:.2f}")
    print(f"F1: {f1:.2f}")
    print(f"平衡准确率(BCA): {bca * 100:.2f}%")
    print(f"灵敏度(Sen): {sen * 100:.2f}%")
    print(f"特异性(Spec): {spec * 100:.2f}%")
    print(f"查准率(Precision): {precision * 100:.2f}%")
    print(f"查全率(Recall): {recall * 100:.2f}%")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"TN: {TN}")
    print(f"总计图片数量: {len(image_files)}")
    print(f"总计查询数量: {total_queries}")
    print(f"总计正确数量: {correct_count}")
    print(f"总计错误数量: {total_queries - correct_count}")

    # 将当前测试结果存储到all_results列表中
    data = {
        '文件夹名': os.path.basename(image_directory),
        '查询正确率': accuracy * 100,
        'AUC': auc,
        'F1': f1,
        '平衡准确率(BCA)': bca * 100,
        '灵敏度(Sen)': sen * 100,
        '特异性(Spec)': spec * 100,
        '查准率(Precision)': precision * 100,
        '查全率(Recall)': recall * 100,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        '总计图片数量': total_queries,
        '总计查询数量': total_queries,
        '总计正确数量': correct_count,
        '总计错误数量': total_queries - correct_count
    }
    all_results.append(data)

# 生成汇总Excel文件名
summary_excel_filename = f'summary_results_run_{run_count}_sub{begin_num}-{end_num}.xlsx'

# 将所有测试结果写入汇总Excel文件
df_summary = pd.DataFrame(all_results)
df_summary.to_excel(summary_excel_filename, index=False)
