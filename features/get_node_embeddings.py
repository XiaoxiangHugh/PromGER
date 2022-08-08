import numpy as np
import pandas as pd
import networkx as nx


def get_LNS(feature_matrix, neighbor_num):
    feature_matrix = np.matrix(feature_matrix) #注意：matrix不同于array
    iteration_max = 40  # same as 2018 bibm
    mu = 3  # same as 2018 bibm
    X = feature_matrix
    alpha = np.power(X, 2).sum(axis=1) # power(X, 2)将特征矩阵中的每一个元素x进行平方运算， sum(axis=1)将结果逐行相加
    distance_matrix = np.sqrt(alpha + alpha.T - 2 * X * X.T) # matrix中的 .T 和 * 与array不同。  由Frobenius范数得到距离矩阵，为主对角线为0的对称矩阵
    row_num = X.shape[0]      # row_num是数据点的个数
    e = np.ones((row_num, 1)) # e是所有元素值都等于1的列向量，(row_num, 1)——>行数为row_num，列数为1
    distance_matrix = np.array(distance_matrix + np.diag(np.diag(e * e.T * np.inf))) # 优化距离矩阵，将主对角线元素置为无穷大。np.diag(array):array是一个一维数组时，输出一个以一维数组为对角线元素的矩阵；array是一个二维矩阵时，输出矩阵的对角线元素
    sort_index = np.argsort(distance_matrix, kind='mergesort')# argsort返回矩阵中的每一行从小到大排列的值对应的索引
    nearest_neighbor_index = sort_index[:, :neighbor_num].flatten()
    nearest_neighbor_matrix = np.zeros((row_num, row_num))
    nearest_neighbor_matrix[np.arange(row_num).repeat(neighbor_num), nearest_neighbor_index] = 1
    C = nearest_neighbor_matrix # C是一个指标矩阵
    np.random.seed(0) #使设有相同参数的seed函数之后的随机数都相等
    W = np.mat(np.random.rand(row_num, row_num), dtype=float)
    W = np.multiply(C, W)      # 对应位置相乘，Hadamard算子； mat生成的数组的点乘只能用np.multiply()
    lamda = mu * e             # miu * e
    P = X * X.T + lamda * e.T  # X * XT + miu * e * eT
    for q in range(iteration_max):
        Q = W * P
        W = np.multiply(W, P) / Q
        W = np.nan_to_num(W) #用0替换NaN， 用最大的有限数替换inf
    return np.array(W)


def get_Graph(df, neighbor_num=20):
    print("neighbor_num=", neighbor_num)
    # 获取Graph
    feature_matrix = list(df.values)
    p_n_fs_w = get_LNS((np.array(feature_matrix)), neighbor_num=neighbor_num)
    p_n_fs_w_df = pd.DataFrame(p_n_fs_w, columns=list(range(len(df))),
                               index=list(range(len(df))))
    # print(p_n_fs_w_df)

    G = nx.from_pandas_adjacency(p_n_fs_w_df, create_using=nx.Graph())
    if not nx.is_connected(G):
        print("Error! The initial graph is not connected.")
    t = 0.9
    print("get Graph")
    p_n_fs_w_df_new = p_n_fs_w_df
    p_n_fs_w_df_new[p_n_fs_w_df > t] = 1
    p_n_fs_w_df_new[p_n_fs_w_df <= t] = 0
    # G = nx.from_pandas_adjacency(p_n_fs_w_df_new, create_using=nx.Graph())
    while not nx.is_connected(G):
        p_n_fs_w_df_new = p_n_fs_w_df
        t = t - 0.001
        p_n_fs_w_df_new[p_n_fs_w_df > t] = 1
        p_n_fs_w_df_new[p_n_fs_w_df <= t] = 0
        G = nx.from_pandas_adjacency(p_n_fs_w_df_new, create_using=nx.Graph())
    # print(p_n_fs_w_df_new)
    print("G is connected", nx.is_connected(G))
    return G


def get_Graph2(df, neighbor_num=20, init_t=None):
    print("neighbor_num =", neighbor_num)
    feature_matrix = list(df.values)
    p_n_fs_w = get_LNS((np.array(feature_matrix)), neighbor_num=neighbor_num)
    p_n_fs_w_df = pd.DataFrame(p_n_fs_w, columns=list(range(len(df))),
                               index=list(range(len(df))))
    # print(p_n_fs_w_df)

    # 判断初始Graph是否是连通图，若不是，则增加neighbor_num，直到G为连通图
    G = nx.from_pandas_adjacency(p_n_fs_w_df, create_using=nx.Graph())
    while not nx.is_connected(G):
        neighbor_num += 10
        print("neighbor_num is added to", neighbor_num)
        p_n_fs_w = get_LNS((np.array(feature_matrix)), neighbor_num=neighbor_num)
        p_n_fs_w_df = pd.DataFrame(p_n_fs_w, columns=list(range(len(df))),
                                   index=list(range(len(df))))
        G = nx.from_pandas_adjacency(p_n_fs_w_df, create_using=nx.Graph())

    t = 1 / neighbor_num
    if init_t is not None:
        t = init_t
    print("get Graph")
    p_n_fs_w_df_new = p_n_fs_w_df.copy()
    p_n_fs_w_df_new[p_n_fs_w_df > t] = 1
    p_n_fs_w_df_new[p_n_fs_w_df <= t] = 0
    G = nx.from_pandas_adjacency(p_n_fs_w_df_new, create_using=nx.Graph())

    while not nx.is_connected(G):
        p_n_fs_w_df_new = p_n_fs_w_df.copy()
        t = t - 1 / (neighbor_num * 100)
        p_n_fs_w_df_new[p_n_fs_w_df > t] = 1
        p_n_fs_w_df_new[p_n_fs_w_df <= t] = 0
        G = nx.from_pandas_adjacency(p_n_fs_w_df_new, create_using=nx.Graph())
    # print(p_n_fs_w_df_new)
    print("The threshold value ", t)
    print("G is connected", nx.is_connected(G))
    return G


def get_Graph3(df, neighbor_num=20, init_t=None):
    print("neighbor_num=", neighbor_num)
    feature_matrix = list(df.values)
    p_n_fs_w = get_LNS((np.array(feature_matrix)), neighbor_num=neighbor_num)
    p_n_fs_w_df = pd.DataFrame(p_n_fs_w, columns=list(range(len(df))),
                               index=list(range(len(df))))
    # print(p_n_fs_w_df)

    t = 1 / neighbor_num
    if init_t is not None:
        t = init_t
    print("get Graph")
    p_n_fs_w_df_new = p_n_fs_w_df.copy()
    p_n_fs_w_df_new[p_n_fs_w_df <= t] = 0
    G = nx.from_pandas_adjacency(p_n_fs_w_df_new, create_using=nx.Graph())

    while not nx.is_connected(G):
        p_n_fs_w_df_new = p_n_fs_w_df.copy()
        t = t - 1 / (neighbor_num * 100)
        p_n_fs_w_df_new[p_n_fs_w_df <= t] = 0
        G = nx.from_pandas_adjacency(p_n_fs_w_df_new, create_using=nx.Graph())
    # print(p_n_fs_w_df_new)
    print("The threshold value ", t)
    print("G is connected", nx.is_connected(G))
    return G


def get_node_embeddings(feature_df, G, ne_methods_dict):
    # G = get_Graph(feature_df, neighbor_num=neighbor_num)
    node_embeddings = [[] for node in range(len(feature_df))]
    from karateclub.node_embedding import Node2Vec, GraRep, SocioDim
    for ne_methods in ne_methods_dict.keys():
        if ne_methods not in ["Node2Vec", "GraRep", "LaplacianEigenmaps", "SocioDim"]:
            print("Error！The ne_methods_dicts may be set incorrectly.")

    if "Node2Vec" in ne_methods_dict.keys():
        print("Node2Vec")
        if isinstance(ne_methods_dict["Node2Vec"], list):
            for Node2Vec_parameter in ne_methods_dict["Node2Vec"]:
                node2vec = Node2Vec(
                    **Node2Vec_parameter)  # 参数有dimensions(128), epochs(1), workers(4), walk_number(10),walk_length(80),learning_rate(0.05),window_size(5)
                node2vec.fit(G)
                for i, vec in enumerate(node2vec.get_embedding()):
                    node_embeddings[i].extend(list(vec))
        elif isinstance(ne_methods_dict["Node2Vec"], dict):
            node2vec = Node2Vec(**ne_methods_dict["Node2Vec"])
            node2vec.fit(G)
            for i, vec in enumerate(node2vec.get_embedding()):
                node_embeddings[i].extend(list(vec))
        else:
            print("错误！")

    if "GraRep" in ne_methods_dict.keys():
        print("GraRep")
        if isinstance(ne_methods_dict["GraRep"], list):
            for GraRep_parameter in ne_methods_dict["GraRep"]:
                grarep = GraRep(**GraRep_parameter)  # 参数有dimensions(128), iteration(10), order(5)
                grarep.fit(G)
                for i, vec in enumerate(grarep.get_embedding()):
                    node_embeddings[i].extend(list(vec))
        elif isinstance(ne_methods_dict["GraRep"], dict):
            grarep = GraRep(
                **ne_methods_dict["GraRep"])
            grarep.fit(G)
            for i, vec in enumerate(grarep.get_embedding()):
                node_embeddings[i].extend(list(vec))
        else:
            print("错误！")

    if "SocioDim" in ne_methods_dict.keys():
        print("SocioDim")
        if isinstance(ne_methods_dict["SocioDim"], list):
            for SocioDim_parameter in ne_methods_dict["SocioDim"]:
                sociodim = SocioDim(
                    **SocioDim_parameter)  # 参数有dimensions(128)
                sociodim.fit(G)
                for i, vec in enumerate(sociodim.get_embedding()):
                    node_embeddings[i].extend(list(vec))
        elif isinstance(ne_methods_dict["SocioDim"], dict):
            sociodim = SocioDim(
                **ne_methods_dict["SocioDim"])
            sociodim.fit(G)
            for i, vec in enumerate(sociodim.get_embedding()):
                node_embeddings[i].extend(list(vec))
        else:
            print("Error！")

    print("finish graph embeddings training")

    node_embeddings_df = pd.DataFrame(node_embeddings)
    node_embeddings_df.index = feature_df.index

    if len(node_embeddings_df) == 0:
        print("Error！The ne_methods_dicts may be set incorrectly.")
    return node_embeddings_df
