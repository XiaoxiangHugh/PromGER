cpm_dict = {'A': (1., 1., 1.), 'C': (0., 0., 1.), 'G': (1., 0., 0.), 'T': (0., 1., 0.)}
cpm_dict2 = {'A': [0., 0., 0., 1.], 'C': [0., 0., 1., 0.], 'G': [0., 1., 0., 0.], 'T': [1., 0., 0., 0.]}
seq_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def density(seq):
    res = []
    d = {'A': 0., 'T': 0., 'C': 0., 'G': 0.} # d是字典，在当前的序列位置上，记录四个核苷酸之前出现的总次数
    for i in range(len(seq)):
        d[seq[i]] += 1
        res.append(d[seq[i]] / (i + 1))
    return res


def get_NCP_ND_feature(seq):
    res = []
    # seq = remove_center_GAC(seq)
    den = density(seq)
    for n, i in zip(seq, range(len(den))):  #zip可以将多个序列（列表、元组、字典、range()区间构成的列表）压缩成一个zip对象，就是将这些序列中对应位置的元素重新组合成一个个新的元素
        res.extend(cpm_dict[n]) #extend把一个序列的内容添加到列表中。若序列为列表，则将列表元素在尾部逐一加入；若序列为字典，则将字典的key值在尾部逐一加入
        res.append(den[i])
    return res


if __name__ == "__main__":
    seq = "TCGTTCATGG"
    print(density(seq))
    print(get_NCP_ND_feature(seq))
    print(seq)
