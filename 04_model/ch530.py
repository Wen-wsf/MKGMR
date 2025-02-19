import pandas as pd
import matplotlib.pyplot as plt
# # Load the dataset
# df = pd.read_csv('D:\CODE\multi-model knowledge graph multi-graph recommendation system\data\cleanuser_rating.csv')

# # Display the first few rows of the dataframe to understand its structure
# print(df.head())

# # Check basic information about the dataset
# print(df.info())

# # Calculate the number of unique users
# unique_users = df['userId'].nunique()
# print("Number of unique users:", unique_users)

# # Calculate statistics about the number of historical records per user
# user_history_counts = df.groupby('userId').size()
# print("Statistics of user history counts:\n", user_history_counts.describe())

# # Display the distribution of the number of historical records per user

# user_history_counts.hist(bins=50)
# plt.title('Distribution of Historical Records per User')
# plt.xlabel('Number of Records')
# plt.ylabel('Number of Users')
# plt.show()


import dgl
import pickle

with open(r'D:\CODE\multi-model knowledge graph multi-graph recommendation system\code\mainmodel\graph\updated_hetero_graph03_with_V_T_A_features.pkl', 'rb') as f:
    hetero_graph = pickle.load(f)
# 列印所有電影節點的ID
movie_ids = hetero_graph.nodes('movie').numpy()
print("所有電影節點的ID:", movie_ids)

# 列印電影節點ID的最小和最大值
print("電影節點ID範圍:", movie_ids.min(), "到", movie_ids.max())

# 列印元圖，顯示所有的節點類型和邊類型
print("元圖結構:")
print(hetero_graph.metagraph().edges())

print("\n節點和邊的數量:")
for ntype in hetero_graph.ntypes:
    print(f"{ntype}節點數量: {hetero_graph.number_of_nodes(ntype)}")
for etype in hetero_graph.canonical_etypes:
    print(f"{etype}邊數量: {hetero_graph.number_of_edges(etype)}")

# 检檢查特定的邊類型是否存在
expected_edges = [
    ('user', 'rates', 'movie'),
    ('movie', 'similar', 'movie'),
    ('user', 'similar', 'user'),
    ('movie', 'has_image', 'movieimage'),
    ('movie', 'has_text', 'movietext'),
    ('movie', 'has_audio', 'movieaudio')
]

print("\n檢查預期的邊類型:")
for edge in expected_edges:
    if edge in hetero_graph.canonical_etypes:
        print(f"邊 {edge} 存在.")
    else:
        print(f"邊 {edge} 不存在.")


import dgl
import pickle


# 列印元圖，顯示所有的節點類型和邊類型
print("元圖結構:")
print(hetero_graph.metagraph().edges())

print("\n節點和邊的數量:")
for ntype in hetero_graph.ntypes:
    print(f"{ntype}節點數量: {hetero_graph.number_of_nodes(ntype)}")
for etype in hetero_graph.canonical_etypes:
    print(f"{etype}邊數量: {hetero_graph.number_of_edges(etype)}")

# 檢查特定電影節點是否有與製作團隊成員的邊
movie_id = 5996 # 假設我們檢查ID為0的電影
print(f"\n電影節點ID {movie_id} 的邊檢查:")

# 導演
if ('movie', 'has_director', 'director') in hetero_graph.canonical_etypes:
    directors = hetero_graph.successors(movie_id, etype=('movie', 'has_director', 'director'))
    print(f"電影節點 {movie_id} 的導演: {directors}")

# 編劇
if ('movie', 'has_writer', 'writer') in hetero_graph.canonical_etypes:
    writers = hetero_graph.successors(movie_id, etype=('movie', 'has_writer', 'writer'))
    print(f"電影節點 {movie_id} 的編劇: {writers}")

# 演員
if ('movie', 'has_actor', 'actor') in hetero_graph.canonical_etypes:
    actors = hetero_graph.successors(movie_id, etype=('movie', 'has_actor', 'actor'))
    print(f"電影節點 {movie_id} 的演員: {actors}")

# 製作人
if ('movie', 'has_producer', 'producer') in hetero_graph.canonical_etypes:
    producers = hetero_graph.successors(movie_id, etype=('movie', 'has_producer', 'producer'))
    print(f"電影節點 {movie_id} 的製作人: {producers}")

# 類似地，您可以檢查其他製作團隊成員

# 檢查整體邊分佈（每種邊類型的前幾個例子）
print("\n每種邊類型的前幾個例子:")
for etype in hetero_graph.canonical_etypes:
    src, dst = hetero_graph.edges(etype=etype)
    print(f"{etype} 邊類型的前 5 個邊: {list(zip(src.numpy(), dst.numpy()))[:5]}")
