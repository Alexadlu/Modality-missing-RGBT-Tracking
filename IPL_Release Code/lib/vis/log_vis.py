# 可视化log数据

import numpy as np
import matplotlib.pyplot as plt
import torch

def moving_average(interval, windowsize=1):
    if windowsize>1:
        window = np.ones(int(windowsize)) / float(windowsize)

        # 消除边界影响
        a = np.ones(windowsize//2)*interval[0]
        b = np.ones(windowsize//2)*interval[-1]
        re = np.convolve(np.concatenate([a,interval,b]), window, 'same')

        return re[len(a) : -len(b)]
    
    else:
        return interval


log_name = 'attn_fusion-attn_same'
cycle_num = 10

log_name = 'attn_fusion-cross_attn_fusion'
cycle_num = 12

log_name = 'attn_fusion-cross_attn_fusion_6'
cycle_num = 16

log_name = 'attn_fusion_train-cross_attn_fusion'
cycle_num = 14

log_name = 'attn_fusion_train-cross_attn_fusion_ts'
cycle_num = 16

log_name = 'attn_fusion-cross_attn_fusion_ts'
cycle_num = 16

log_name = 'attn_fusion_uni-cross_attn_fusion'
cycle_num = 14

log_name = 'attn_fusion_train-cross_attn_fusion_uni'
cycle_num = 14

log_name = 'attn_fusion-uni_attn'
cycle_num = 12

log_name = 'attn_fusion-cross_attn_fusion_mh'
cycle_num = 14

log_name = 'attn_fusion-CAiA_v3'
cycle_num = 12

# log_name = 'attn_fusion_uni-cross_attn_fusion_old'
# cycle_num = 14

print('读取log文件...')
log_file = f"/home/zhaojiacong/ostrack_attnFusion/output/logs/{log_name}.log"
with open(log_file) as f:
    log_text = f.read()

print("文件读取完毕，分析中...")
log_text = log_text.split('\n')

# 内容筛选
key = 'val'
log_text = [item for item in log_text if key in item]
print('...')

# 查找内容
val_loss_total = []

cycle=0
for col in log_text:
    idx_0 = col.index("Loss/total:")+len("Loss/total:")
    # idx_0 = col.index("Loss/total/f:")+len("Loss/total/f:")
    idx_1 = col.index(",", idx_0)
    val = float(col[idx_0:idx_1].replace(' ',''))
    if cycle:
        val_loss_total[-1].append(val)
        cycle-=1
    else:
        val_loss_total.append([])
        val_loss_total[-1].append(val)
        cycle=cycle_num-1
        print('.', end='')
print()
print('分析完毕，绘图中...')

val_loss_total = torch.tensor(val_loss_total)
val_loss_total = moving_average(np.array(val_loss_total.mean(-1)), 3)
print(val_loss_total.shape)
print(val_loss_total)
print(val_loss_total.min())
plt.plot(val_loss_total)
plt.savefig(f"{log_name}_val_loss.png")