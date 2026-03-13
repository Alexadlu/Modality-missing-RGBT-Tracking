import numpy as np
import random
import json
import os

# 新增函数：读取RGBT数据集并返回格式化数据
def read_rgbt_dataset(dataset_path):
    dataset = {}

    # 获取数据集路径下的所有序列文件夹
    #sequence_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))] # for rgbt234
    squence_name_list = ['10runone', '11leftboy', '11runtwo', '1blackteacher', '1boycoming', '1stcol4thboy', '1strowleftboyturning', '1strowrightdrillmaster', '1strowrightgirl3540', '2girl', '2girlup', '2runseven', '3bike1', '3men', '3pinkleft', '3rdfatboy', '3rdgrouplastboy', '3thmoto', '4men', '4thboywithwhite', '7rightorangegirl', 'AQgirlwalkinrain', 'AQtruck2north', 'ab_bikeoccluded', 'ab_blkskirtgirl', 'ab_bolstershaking', 'ab_girlchoosesbike', 'ab_girlcrossroad', 'ab_pingpongball2', 'ab_rightlowerredcup_quezhen', 'ab_whiteboywithbluebag', 'advancedredcup', 'baggirl', 'ballshootatthebasket3times', 'basketball849', 'basketballathand', 'basketboy', 'bawgirl', 'belowdarkgirl', 'besom3', 'bike', 'bike2left', 'bike2trees', 'bikeboy', 'bikeboyintodark', 'bikeboyright', 'bikeboyturn', 'bikeboyturntimes', 'bikeboywithumbrella', 'bikefromlight', 'bikegoindark', 'bikeinrain', 'biketurnright', 'blackboy', 'blackboyoncall', 'blackcarturn', 'blackdown', 'blackgirl', 'blkboy`shead', 'blkboyback', 'blkboybetweenredandwhite', 'blkboydown', 'blkboyhead', 'blkboylefttheNo_21', 'blkboystand', 'blkboytakesumbrella', 'blkcaratfrontbluebus', 'blkgirlumbrella', 'blkhairgirltakingblkbag', 'blkmoto2north', 'blkstandboy', 'blktribikecome', 'blueboy', 'blueboy421', 'bluebuscoming', 'bluegirlbiketurn', 'bottlebetweenboy`sfeet', 'boy2basketballground', 'boy2buildings', 'boy2trees', 'boy2treesfindbike', 'boy`headwithouthat', 'boy`sheadingreycol', 'boyaftertree', 'boyaroundtrees', 'boyatdoorturnright', 'boydownplatform', 'boyfromdark', 'boyinlight', 'boyinplatform', 'boyinsnowfield3', 'boyleftblkrunning2crowd', 'boylefttheNo_9boy', 'boyoncall', 'boyplayphone', 'boyride2path', 'boyruninsnow', 'boyscomeleft', 'boyshead9684', 'boyss', 'boytakingbasketballfollowing', 'boytakingplate2left', 'boyunder2baskets', 'boywaitgirl', 'boywalkinginsnow2', 'broom', 'carbehindtrees', 'carcomeonlight', 'carcomingfromlight', 'carcominginlight', 'carlight2', 'carlightcome2', 'caronlight', 'carturn117', 'carwillturn', 'catbrown2', 'catbrownback2bush', 'couple', 'darkcarturn', 'darkgirl', 'darkouterwhiteboy', 'darktreesboy', 'drillmaster1117', 'drillmasterfollowingatright', 'farfatboy', 'firstexercisebook', 'foamatgirl`srighthand', 'foldedfolderatlefthand', 'girl2left3man1', 'girl`sblkbag', 'girlafterglassdoor', 'girldownstairfromlight', 'girlfromlight_quezhen', 'girlinrain', 'girllongskirt', 'girlof2leaders', 'girlrightthewautress', 'girlunderthestreetlamp', 'guardunderthecolumn', 'hugboy', 'hyalinepaperfrontface', 'large', 'lastleftgirl', 'leftblkTboy', 'leftbottle2hang', 'leftboy2jointhe4', 'leftboyoutofthetroop', 'leftchair', 'lefterbike', 'leftexcersicebookyellow', 'leftfarboycomingpicktheball', "leftgirl'swhitebag", 'lefthyalinepaper2rgb', 'lefthyalinepaperfrontpants', 'leftmirror', 'leftmirrorlikesky', 'leftmirrorside', 'leftopenexersicebook', 'leftpingpongball', 'leftrushingboy', 'leftunderbasket', 'leftuphand', 'littelbabycryingforahug', 'lowerfoamboard', 'mandownstair', 'manfromtoilet', 'mangetsoff', 'manoncall', 'mansimiliar', 'mantostartcar', 'midblkgirl', 'midboyNo_9', 'middrillmaster', 'midgreyboyrunningcoming', 'midof3girls', 'midredboy', 'midrunboywithwhite', 'minibus', 'minibusgoes2left', 'moto', 'motocomeonlight', 'motogoesaloongS', 'mototaking2boys306', 'mototurneast', 'motowithbluetop', 'pingpingpad3', 'pinkwithblktopcup', 'raincarturn', 'rainycarcome_ab', 'redboygoright', 'redcarcominginlight', 'redetricycle', 'redmidboy', 'redroadlatboy', 'redtricycle', 'right2ndflagformath', 'right5thflag', 'rightbike', 'rightbike-gai', 'rightblkboy4386', 'rightblkboystand', 'rightblkfatboyleftwhite', 'rightbluewhite', 'rightbottlecomes', 'rightboy504', 'rightcameraman', 'rightcar-chongT', 'rightcomingstrongboy', 'rightdarksingleman', 'rightgirltakingcup', 'rightwaiter1_quezhen', 'runningcameragirl', 'shinybikeboy2left', 'shinycarcoming', 'shinycarcoming2', 'silvercarturn', 'small-gai', 'standblkboy', 'swan_0109', 'truckgonorth', 'turning1strowleft2ndboy', 'umbreboyoncall', 'umbrella', 'umbrellabyboy', 'umbrellawillbefold', 'umbrellawillopen', 'waitresscoming', 'whitebikebelow', 'whiteboyrightcoccergoal', 'whitecarcomeinrain', 'whitecarturn683', 'whitecarturnleft', 'whitecarturnright', 'whitefardown', 'whitefargirl', 'whitegirlinlight', 'whitegirltakingchopsticks', 'whiteofboys', 'whiteridingbike', 'whiterunningboy', 'whiteskirtgirlcomingfromgoal', 'whitesuvturn', 'womanback2car', 'yellowgirl118', 'yellowskirt']

    for sequence_folder in squence_name_list:
        sequence_dir = os.path.join(dataset_path, sequence_folder)
        
        # 初始化序列信息
        sequence_info = {"frames": None, "modalities": []}
        
        # 遍历两种模态：visible 和 infrared
        modalities = ["visible", "infrared"]
        for modality in modalities:
            modality_dir = os.path.join(sequence_dir, modality)
            if os.path.exists(modality_dir):
                # 读取模态下的图像序列
                frame_files = [file for file in os.listdir(modality_dir)]
                sequence_info["frames"] = len(frame_files)
                # sequence_info["modalities"].append(modalities.index(modality))  # 使用0表示visible，1表示infrared

        # 将序列信息添加到数据集字典中
        dataset[sequence_folder] = sequence_info
    
    return dataset

# 随机缺失函数，每个随机缺失帧有三种缺失情况：只有RGB模态缺失、只有T模态缺失、RGB和T模态都缺失
def random_missing(frames, missing_ratio, modalities=2):
    num_missing_frames = int(frames * missing_ratio)
    missing_frames = random.sample(range(frames), num_missing_frames)
    missing_data = np.ones((frames, modalities))
    
    for frame in missing_frames:
        missing_case = random.choice([0, 1, 2])
        if missing_case == 0:
            missing_data[frame][0] = 0
        if missing_case == 1:
            missing_data[frame][1] = 0            
        if missing_case == 2:
            missing_data[frame][0] = 0 
            missing_data[frame][1] = 0
            
    return missing_data

# 长期缺失函数，缺失某个模态是随机选择的
def long_term_missing(frames, missing_ratio, modalities=2):
    num_missing_frames = int(frames * missing_ratio)
    start = random.randint(0, frames - num_missing_frames)
    missing_data = np.ones((frames, modalities))
    missing_modality = random.choice([0, 1]) # 随机选择一个模态缺失
    
    for i in range(start, start + num_missing_frames):
        missing_data[i][missing_modality] = 0
    return missing_data

# 交换缺失函数，缺失模态是随机选择的
def swap_missing(frames, missing_ratio, modalities=2):
    num_missing_frames = int(frames * missing_ratio)
    # unit_length = random.randint(1, frames // 2)
    unit_length = random.choice([frames // 6, frames // 4, frames // 2])
    missing_data = np.ones((frames, modalities))
    start = random.randint(0, frames - num_missing_frames)
    count = 0
    missing_modality = random.choice([0, 1]) 
    
    
    for i in range(start, start + num_missing_frames):
        count = count + 1
        missing_data[i][missing_modality] = 0
        
        if unit_length == count:
            if missing_modality == 0:
                missing_modality = 1
            else:
                missing_modality = 0
            count = 0 
        
    return missing_data

# 长期缺失+随机缺失函数
def long_term_and_random_missing(frames, missing_ratio, modalities=2):
    # 长期缺失操作
    missing_data = long_term_missing(frames, missing_ratio, modalities=2)
    
    # 找到第一次缺失操作执行的帧
    missing_frames = np.where(missing_data == 0)[0]
    
    # 随机挑选50%的帧并执行随机缺失操作
    num_random_missing_frames = int(len(missing_frames) * 0.5)
    random_missing_frames = random.sample(list(missing_frames), num_random_missing_frames)
    
    # for frame in random_missing_frames:
    #     missing_modality = random.choice(modalities)  # 随机选择一个模态缺失
    #     missing_data[frame][missing_modality] = 0
    for frame in random_missing_frames:
        # if missing_modality = 0:
            missing_data[frame][0] = 0
        # else:
            missing_data[frame][1] = 0

    
    return missing_data

# 交换缺失+随机缺失函数
def swap_and_random_missing(frames, missing_ratio, modalities):
    # 交换缺失操作
    missing_data = swap_missing(frames, missing_ratio, modalities)
    
    # 找到第一次缺失操作执行的帧
    missing_frames = np.where(missing_data == 0)[0]
    
    # 随机挑选50%的帧并执行随机缺失操作
    num_random_missing_frames = int(len(missing_frames) * 0.5)
    random_missing_frames = random.sample(list(missing_frames), num_random_missing_frames)
    
    for frame in random_missing_frames:
        # missing_modality = random.choice(modalities)  # 随机选择一个模态缺失
        # missing_data[frame][missing_modality] = 0
        missing_data[frame][0] = 0

        missing_data[frame][1] = 0
        
    
    return missing_data

# 定义分组函数
import random

def split_and_shuffle_dataset(dataset):
    # 将数据集按序列数量降序排列
    sorted_dataset = sorted(dataset.items(), key=lambda x: x[1]["frames"], reverse=True)
    num_sequences = len(sorted_dataset)
    print(num_sequences)
    # 缺失函数名称列表，按顺序与大分组对应
    missing_functions = ["random_missing", "long_term_missing", "swap_missing", "long_term_and_random_missing", "swap_and_random_missing"]
    
    # 初始化分组
    groups = {}

    # 每次取出5个序列并打乱顺序后分配到分组中
    batch_size = 5
    batch = []

    for i, (seq_name, seq_info) in enumerate(sorted_dataset):
        batch.append((seq_name, seq_info))
        
        if len(batch) == batch_size:
            random.shuffle(batch)  # 随机打乱顺序

            for j, (seq_name, seq_info) in enumerate(batch):
                group_idx = j % 5  # 计算序列应该分配到哪个大组
                group_name = f"Group{group_idx + 1}_{missing_functions[group_idx]}"  # 使用缺失函数名称
                
                if group_name not in groups:
                    groups[group_name] = {}
                    groups[group_name]["missing_function"] = missing_functions[group_idx]
                    groups[group_name]["sequences"] = []

                groups[group_name]["sequences"].append((seq_name, seq_info))
            
            batch = []  # 清空批次
    
    # 如果还有不足5个的剩余序列，同样要打乱顺序后分配到分组中
    if batch:
        random.shuffle(batch)

        for j, (seq_name, seq_info) in enumerate(batch):
            group_idx = j % 5  # 计算序列应该分配到哪个大组
            group_name = f"Group{group_idx + 1}_{missing_functions[group_idx]}"  # 使用缺失函数名称
            
            if group_name not in groups:
                groups[group_name] = {}
                groups[group_name]["missing_function"] = missing_functions[group_idx]
                groups[group_name]["sequences"] = []

            groups[group_name]["sequences"].append((seq_name, seq_info))
    
    # 生成五大组的JSON文件
    with open('groups_data_lasher245.json', 'w') as json_file:
        json.dump(groups, json_file, indent=4)

    return groups


import random

def split_group_subgroups(groups):
    subgroups_data = {}

    # 缺失比例列表
    missing_ratios = [0.3, 0.6, 0.9]

    for group_name, group_info in groups.items():
        # 初始化小组信息
        subgroups_info = {}
        sequences = group_info["sequences"]
        num_sequences = len(sequences)

        # 按帧数降序排列序列
        sorted_sequences = sorted(sequences, key=lambda x: x[1]["frames"], reverse=True)

        batch = []  # 用于存储每个 batch 中的序列

        for i, (seq_name, seq_info) in enumerate(sorted_sequences):
            batch.append((seq_name, seq_info))

            # 当 batch 中有三个序列时，执行随机打乱并分配到小组中
            if len(batch) == 3:
                random.shuffle(batch)  # 随机打乱顺序

                for j, (batch_seq_name, batch_seq_info) in enumerate(batch):
                    # 每个小组对应不同的缺失比例
                    subgroup_missing_ratio = missing_ratios[j]

                    # 分配到小组中
                    subgroup_name = f"{group_name}_SubGroup{j + 1}"
                    subgroup_info = group_info.copy()
                    subgroup_info["sequences"] = [(batch_seq_name, batch_seq_info, subgroup_missing_ratio)]

                    if subgroup_name not in subgroups_info:
                        subgroups_info[subgroup_name] = subgroup_info
                    else:
                        subgroups_info[subgroup_name]["sequences"].append((batch_seq_name, batch_seq_info, subgroup_missing_ratio))

                batch = []  # 清空 batch
            
        if batch:
            random.shuffle(batch)  # 随机打乱顺序
            for j, (batch_seq_name, batch_seq_info) in enumerate(batch):
                # 每个小组对应不同的缺失比例
                subgroup_missing_ratio = missing_ratios[2-j]

                # 分配到小组中
                subgroup_name = f"{group_name}_SubGroup{2-j + 1}"
                subgroup_info = group_info.copy()
                subgroup_info["sequences"] = [(batch_seq_name, batch_seq_info, subgroup_missing_ratio)]

                if subgroup_name not in subgroups_info:
                    subgroups_info[subgroup_name] = subgroup_info
                else:
                    subgroups_info[subgroup_name]["sequences"].append((batch_seq_name, batch_seq_info, subgroup_missing_ratio))

            batch = []  # 清空 batch            
        subgroups_data[group_name] = subgroups_info

    # 生成小组划分的 JSON 文件
    with open('subgroups_data_lasher245.json', 'w') as json_file:
        json.dump(subgroups_data, json_file, indent=4)

    return subgroups_data


# 新增函数：执行缺失操作并保存结果
def execute_missing_operations_and_save_results(subgroups_data):
    result_data = {}

    for group_name, subgroup_info in subgroups_data.items():
        for subgroup_name, subgroup_data in subgroup_info.items():
            missing_function = subgroup_data["missing_function"]
            sequences = subgroup_data["sequences"]

            for seq_name, seq_info, missing_ratio in sequences:
                frames = seq_info["frames"]
                #modalities = seq_info["modalities"]
                modalities=2
                missing_data = None

                if missing_function == "random_missing":
                    missing_data = random_missing(frames, missing_ratio, modalities)
                elif missing_function == "long_term_missing":
                    missing_data = long_term_missing(frames, missing_ratio, modalities)
                elif missing_function == "swap_missing":
                    missing_data = swap_missing(frames, missing_ratio, modalities)
                elif missing_function == "long_term_and_random_missing":
                    missing_data = long_term_and_random_missing(frames, missing_ratio, modalities)
                elif missing_function == "swap_and_random_missing":
                    missing_data = swap_and_random_missing(frames, missing_ratio, modalities)

                if missing_data is not None:
                    result_data[seq_name] = {
                        "frames": frames,
                        "data": missing_data.tolist()
                    }

    # 保存结果到JSON文件
    with open('missing_results_lasher245.json', 'w') as json_file:
        json.dump(result_data, json_file, indent=4)

# 主程序
if __name__ == "__main__":
    dataset_path = '/data/LasHeR/'  # 替换为您的数据集路径
    dataset = read_rgbt_dataset(dataset_path)
    
    # 划分数据集为五组，确定每个大组的缺失函数
    groups = split_and_shuffle_dataset(dataset)

    # 划分大组为小组，确定每个小组的缺失比例
    subgroups = split_group_subgroups(groups)

    # 执行缺失操作并保存结果
    execute_missing_operations_and_save_results(subgroups)
