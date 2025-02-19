import pandas as pd
import dgl
import torch
from itertools import combinations
from sklearn.metrics import jaccard_score
import os
from tqdm import tqdm
import re
import pickle
import numpy as np
from scipy.spatial import distance
import dgl
from tqdm import tqdm
from itertools import combinations
from stepmix import StepMix

# 讀取用戶評分數據
df_ratings = pd.read_csv('D:\\CODE\\multi-model knowledge graph multi-graph recommendation system\\data\\cleanuser_rating.csv')

# 讀取製作團隊信息數據
crew_info_df = pd.read_csv('D:\\CODE\\multi-model knowledge graph multi-graph recommendation system\\data\\final_data_cleaned.csv')

def extract_roles_and_initialize_graph(crew_info_df):
    """從crew_info數據中提取所有唯一的角色類型，並為每個角色創建唯一的ID映射。"""
    roles = {}  # 使用字典來存儲每個角色的ID映射
    edge_types = []  # 存儲邊類型
    for _, row in crew_info_df.iterrows():
        crew_info = parse_crew_info(row['crew_info'])  # 解析製作團隊信息
        for role, ids in crew_info.items():
            if role not in roles:
                roles[role] = {}
            for person_id in ids:
                if person_id not in roles[role]:
                    roles[role][person_id] = len(roles[role])
            edge_types.append(('movie', f'has_{role}', role))  # 添加邊類型
    return roles, edge_types

def parse_crew_info(crew_info_str):
    """解析製作團隊信息字符串，返回字典格式：{角色: [id, ...]}"""
    # 使用正則表達式匹配所有的角色和ID組合
    pattern = re.compile(r"(\w+): ([\w, ]+)")
    matches = pattern.findall(crew_info_str)
    crew_info = {}
    for role, ids in matches:
        # 將ID字符串分割並去除空格，然後轉換為列表
        crew_info[role] = ids.replace(' ', '').split(',')
    return crew_info

# 創建用戶-電影矩陣
user_movie_matrix = pd.pivot_table(df_ratings, index='userId', columns='movieId', aggfunc='size', fill_value=0)

# 將 pivot table 轉換為 0 或 1，表示是否觀看
user_movie_matrix = (user_movie_matrix > 0).astype(int)

# 轉換成適用於 StepMix 的數據格式
data = user_movie_matrix.values

def hellinger_distance(p, q):
    """計算兩個概率分佈的Hellinger距離"""
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

def compute_user_similarity(membership_probabilities, user_ids, threshold):
    """計算用戶相似度，返回相似用戶的邊列表"""
    num_users = len(user_ids)
    user_similarity_edges = []
    for i in range(num_users):
        for j in range(i + 1, num_users):
            dist = hellinger_distance(membership_probabilities[i], membership_probabilities[j])
            if dist <= threshold:
                user_similarity_edges.append((i, j))
    return user_similarity_edges

def create_heterogeneous_graph(df_ratings, crew_info_df, Uthreshold=0.5, threshold=0.7):
    """創建異構圖"""
    user_ids = df_ratings['userId'].unique()
    movie_ids = df_ratings['movieId'].unique()
    user_id_map = {uid: i for i, uid in enumerate(user_ids)}
    movie_id_map = {mid: i for i, mid in enumerate(movie_ids)}
    
    # 提取角色和初始化圖
    roles, edge_types = extract_roles_and_initialize_graph(crew_info_df)
    
    # 定義額外的邊類型
    edge_types.extend([
        ('movie', 'has_image', 'movieimage'),
        ('movie', 'has_text', 'movietext'),
        ('movie', 'has_audio', 'movieaudio'),
        ('user', 'rates', 'movie'),  # 用戶評分電影
        ('user', 'similar', 'user'),  # 用戶相似
        ('movie', 'similar', 'movie')  # 電影相似
    ])

    # 創建異構圖
    g = dgl.heterograph({etype: [] for etype in edge_types},
                        num_nodes_dict={
                            'user': len(user_ids), 
                            'movie': len(movie_ids),
                            **{role: len(ids) for role, ids in roles.items()},
                            'movieimage': len(movie_ids),  
                            'movietext': len(movie_ids),
                            'movieaudio': len(movie_ids)
                        })

    # 添加電影和角色之間的邊
    for idx, row in tqdm(crew_info_df.iterrows(), total=crew_info_df.shape[0], desc="Processing crew info"):
        movie_id = row['movieId']
        if movie_id in movie_id_map:
            crew_info = parse_crew_info(row['crew_info'])
            for role, ids in crew_info.items():
                src = []
                dst = []
                for person_id in ids:
                    src.append(movie_id_map[movie_id])
                    dst.append(roles[role][person_id])
                g.add_edges(src, dst, etype=('movie', f'has_{role}', role))

    # 創建用戶-電影交互矩陣
    user_movie_matrix = df_ratings.pivot_table(index='userId', columns='movieId', aggfunc='size', fill_value=0)
    
    # 構建StepMix模型並計算用戶相似度
    model = StepMix(n_components=3, measurement='categorical')
    model.fit(data)
    membership_probabilities = model.predict_proba(data)
    user_similarity = compute_user_similarity(membership_probabilities, user_ids, Uthreshold)

    # 計算電影相似度
    movie_similarity = [(movie_id_map[m1], movie_id_map[m2]) for m1, m2 in combinations(movie_ids, 2)
                        if jaccard_score(user_movie_matrix[m1].fillna(0), user_movie_matrix[m2].fillna(0)) >= threshold]
    
    # 添加用戶相似度邊
    if user_similarity:
        src, dst = zip(*user_similarity)
        g.add_edges(src, dst, etype=('user', 'similar', 'user'))
    
    # 添加電影相似度邊
    if movie_similarity:
        src, dst = zip(*movie_similarity)
        g.add_edges(src, dst, etype=('movie', 'similar', 'movie'))

    # 添加用戶評分電影的邊
    user_movie_interactions = [(user_id_map[row['userId']], movie_id_map[row['movieId']])
                               for index, row in df_ratings.iterrows()]
    if user_movie_interactions:
        src, dst = zip(*user_movie_interactions)
        g.add_edges(src, dst, etype=('user', 'rates', 'movie'))

    # 添加電影ID數據
    g.nodes['movie'].data['movie_id'] = torch.tensor([int(mid) for mid in movie_ids], dtype=torch.int64)
    
    return g

# 創建異構圖並保存
hetero_graph = create_heterogeneous_graph(df_ratings, crew_info_df)
with open('new_hetero_graph05.pkl', 'wb') as f:
    pickle.dump(hetero_graph, f)

# 輸出圖的信息
print(hetero_graph)
print(hetero_graph.ntypes, hetero_graph.etypes)
