import sys
import matplotlib.pyplot as plt

# 初始化数据存储
thresholds = []
corr = []
err = []

class_data = {}

if len(sys.argv) < 2:
    print("Usage: python draw.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

# 读取文件
with open(filename, "r") as file:
    for line in file:
        if line.startswith("Threshold:"):
            threshold = float(line.split(": ")[1].strip())
            thresholds.append(threshold)
        elif line.startswith("Class"):
            parts = line.split(':')
            class_id = int(parts[0].split()[1])
            counts = parts[1].split(',')
            count = int(counts[0].split('=')[1].strip())
            corr_value = int(counts[1].split('=')[1].strip())
            err_value = int(counts[2].split('=')[1].strip())
            class_data[class_id] = {'count': count, 'corr': corr_value, 'err': err_value}
        elif line.startswith("total:"):
            parts = line.split(", ")
            total_count = int(parts[0].split("=")[1].strip())
            total_corr = int(parts[1].split("=")[1].strip())
            total_err = int(parts[2].split("=")[1].strip())
            corr.append(total_corr / total_count)
            err.append(total_err / total_count)

# 计算每个类别的Precision和Recall
precision_recall = {}
for class_id, values in class_data.items():
    precision = values['corr'] / (values['corr'] + values['err']) if (values['corr'] + values['err']) > 0 else 0
    recall = values['corr'] / values['count'] if values['count'] > 0 else 0
    precision_recall[class_id] = {'precision': precision, 'recall': recall}

# 计算mAP50
average_precision = []
for class_id, values in precision_recall.items():
    if values['recall'] >= 0.5:
        average_precision.append(values['precision'])

if average_precision:
    mean_average_precision = sum(average_precision) / len(average_precision)
else:
    mean_average_precision = 0

print(f"mAP50: {mean_average_precision:.4f}")

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(thresholds, corr, marker="o", linestyle="-", color="b", label="Correct Rate")
plt.plot(thresholds, err, marker="x", linestyle="--", color="r", label="Error Rate")

for i, (c, e) in enumerate(zip(corr, err)):
    plt.text(thresholds[i], c, f"{c:.2f}", ha='center', va='bottom')
    plt.text(thresholds[i], e, f"{e:.2f}", ha='center', va='top', color='red')

plt.title("Model Correct and Error Rates vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.legend()
plt.grid(True)
plt.xticks(thresholds)  # 设置x轴刻度为阈值
plt.show()