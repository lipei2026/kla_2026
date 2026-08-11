import numpy as np
global nop, non
from pandas import DataFrame
from pandas import Series


def readweight(weight_file):
    weight = None
    with open(weight_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 2 - 1:
                weight = np.array([float(x) for x in line.rstrip().split('\t')])
    return weight


def acf(list30):  # acl,aal
    # import pandas as pd
    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':acf00'+'\n')
    global acl, aal, acn, aan, acsl, aasl
    pos = DataFrame(list30)
    all_ = pos
    # with open('../aaindex1.txt', 'r') as fout:
    #    line=fout.readline()
    #    line=line.rstrip()[5:]
    #    aalist=line.split('\t')
    ll, j = 0, 0
    na = {}  # AAindex字典：特征->氨基酸对应值
    vvl = []  # top10特征名字列表,用于在na字典中搜索
    with open('top10.txt', 'r') as fout:
        line = fout.readline()
        line = line.rstrip()[5:]
        aalist = line.split('\t')  # aa名称列表
        for line in fout:
            ll = ll + 1
            line = line.rstrip()
            na[line.split('\t')[0]] = line.split('\t')[1:]
            vvl.append(line.split('\t')[0])

    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':acf100')
    # with open('../aaindex1.txt', 'r') as fout:
    #    line=fout.readline()
    #    line=line.rstrip()[5:]
    #    aalist=line.split('\t')
    #    for line in fout:
    #        j=j+1
    #        line=line.rstrip()
    #        na[j]=line.split('\t')[1:]
    # vvl1.sort(reverse=True)
    # vvl = []
    # con=range(11,31)
    # ww=open('top10.txt','w')
    # ww.write('name\t'+'\t'.join(aalist)+'\n')
    # for co in (range(30)):
    #     if co + 1 in con:
    #         llo = 0
    #     else:
    #         vvl.append(vvl1[co])
    #         ww.write(str(vvlist[vvl1[co]])+'\t'+'\t'.join(na[vvlist[vvl1[co]]])+'\n')

    def doc2num1(ss):
        sss, AAindex_Encode = [], []
        ss = ss.replace('*', '0')
        ss = ss.replace('X', '0')
        ss = ss.replace('B', '0')
        ss = ss.replace('U', '0')
        for k in vvl:
            s = list(ss)
            for i, ii in enumerate(aalist):
                s = [(na[k][i]) if x == ii else x for x in s]
            s = [float(x) for x in s]
            sss = sss + s
        return sss

    def doc2num(ss):  # ss=all_[0]=usp序列
        # ss=ss[20:-20]
        AAindex_Encode = []
        ss = ss.replace('*', '0')
        ss = ss.replace('X', '0')
        ss = ss.replace('B', '0')
        ss = ss.replace('U', '0')
        for k in vvl:
            s = list(ss)  # 将ss中的序列从dataframe转化为list存在s中
            for i, ii in enumerate(aalist):  # enumerate枚举，i是索引，ii是值：此处，i是aa名称的索引，ii是aa名称
                s = [(na[k][i]) if x == ii else x for x in s]  # k:vvl（特征） + i:aalist（aa） 一一对应
            s = [float(x) for x in s]
            AAindex_Encode.append(s)
        ACF_Encode = np.zeros((np.array(AAindex_Encode).shape[0], np.array(AAindex_Encode).shape[1]))
        for i, seq in enumerate(AAindex_Encode):
            for k, kv in enumerate(ss):
                sumValue = 0
                for j in range(0, len(seq) - k):
                    singleValue = seq[j] * seq[j + k]
                    sumValue = sumValue + singleValue
                ACF_Encode[i][k] = round(sumValue / (len(seq) - k), 2)
        ACF_Encode = ACF_Encode.flatten().tolist()
        return ACF_Encode

    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':acf11'+'\n')
    all_['doc2num'] = all_[0].apply(lambda ss: doc2num(ss))  # 将存在all_[0]中的序列以ss的名字导入doc2num，每次取一条序列
    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':acf12'+'\n')
    all_['doc2num1'] = all_[0].apply(lambda ss: doc2num1(ss))  # 将存在all_[0]中的序列以ss的名字导入doc2num1，每次取一条序列
    # wt.write(time.strftime( ISOTIMEFORMAT, time.localtime( time.time() ) )+':acf13'+'\n')
    # for i in list(all_['doc2num']):
    #     print(len(i))
    xy = np.array(list(all_['doc2num']), dtype=np.float64)
    x2 = xy.tolist()
    x = np.array(list(all_['doc2num1']), dtype=np.float64)
    x3 = x.tolist()

    # acl,aal=acf(list0[0])
    acl, aal = x2, x3
    # aal = x3
    # acn=acl
    # if  spes==9:aan=[a[100:-100] for a in aal]
    # elif spes==0 or s==8:aan=[a[200:-200] for a in aal]
    # else:aan=aal
    # acsl=ls(acl,'ACF')
    # aasl=ls(aal,'AAindex')
    return acl, aal  # acf,AAindex
    # return x3


def kmors(list10, km, m='l'):
    # global kmn
    pos = DataFrame(list10)
    all_ = pos
    aalist = []
    if m == 'd':
        ablist = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    else:
        ablist = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X',
                  'Y']
    for aa in ablist:
        for bb in ablist:
            aalist.append(aa + bb)

    def doc2num(s):
        ssss = []
        for k in range(km):
            alist = []
            for i, a in enumerate(s):
                if i + k + 1 < len(s):
                    alist.append(a + s[i + k + 1])
                else:
                    continue
            ss = [float(alist.count(i)) for i in aalist]
            ssss = ssss + ss[:]
        return list(ssss)

    all_['doc2num'] = all_[0].apply(lambda s: doc2num(s))
    x = np.array(list(all_['doc2num']), dtype=np.int64)
    x2 = x.tolist()
    return x2


# 氨基酸组成
def aac1(list10):
    pos = DataFrame(list10)
    all_ = pos
    aalist = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    ll = len(list10[0])

    def doc2num(s):
        s = [float(s.count(i) / ll) for i in aalist]
        s = s[:]
        return list(s)

    all_['doc2num'] = all_[0].apply(lambda s: doc2num(s))

    x = np.array(list(all_['doc2num']), dtype=np.float64)
    x2 = x.tolist()
    return x2


# list中肽段转为2进制 d控制是否序列中有U
def be1(list10, d):
    all_ = DataFrame(list10)
    abc = Series(range(0, 21),
                 index=['K', 'L', 'A', 'E', 'V', 'G', 'S', 'D', 'I', 'T', 'R', '*', 'P', 'Q', 'N', 'F', 'Y', 'M', 'H',
                        'C', 'W'])
    if d == 0: abc = Series(range(0, 22),
                            index=['K', 'L', 'A', 'E', 'V', 'G', 'S', 'D', 'I', 'T', 'R', '*', 'P', 'Q', 'N', 'F', 'Y',
                                   'M', 'H', 'C', 'W', 'U'])
    abc[:] = range(len(abc))
    word_set = set(abc.index)

    def doc2num(s):
        s = s.replace('X', '*')
        s = s.replace('B', '*')
        s = [i for i in s if i in word_set]
        return list(abc[s])

    all_['doc2num'] = all_[0].apply(lambda s: doc2num(s))
    for i in range(len(list(all_['doc2num']))):
        if len(list(all_['doc2num'][i])) != 51:
            print(i)
    x = np.array(list(all_['doc2num']), dtype=np.int64)
    #gen_matrix = lambda z: tf.keras.utils.to_categorical(z, num_classes=len(abc)).flatten()
    def gen_matrix(z):
        one_hot = np.eye(len(abc))[z]
        return one_hot.flatten()
    def data_generator(data, batch_size):
        batches = [range(batch_size * i, min(len(data), batch_size * (i + 1))) for i in
                   range(int(len(data) / batch_size + 1))]
        while True:
            for i in batches:
                xx = np.array(list(map(gen_matrix, data[i])))
            return (xx)

    x = data_generator(x[:], len(x) + 1)
    bin = x.tolist()
    return bin


# list:[k-mer]
def gps(list):
    global gpn
    for i, ii in enumerate(list):
        ii = ii.replace('U', '*')
        list[i] = ii

    # from keras.models import load_model

    def generateMMData(querylist, plist, pls_weight, mm_weight, loo=True, positive=False):
        gp = GpsPredictor(plist, pls_weight, mm_weight)

        d = []

        for query_peptide in querylist:
            d.append(gp.generateMMdata(query_peptide, loo).tolist())
        return d

    mm_weight = readweight('BLOSUM62R.txt')  # 1th is intercept

    ll = len(list[0])
    gpn = generateMMData(list, list, np.repeat(1, ll), mm_weight, loo=False, positive=False)
    # plist = readPeptide('./Protein/0.peptide',int(ll/2))

    # gpn = generateMMData(list, plist,np.repeat(1, ll), mm_weight, loo=False, positive=False)# for p in nlist]
    return gpn


class GpsPredictor(object):
    def __init__(self, plist, pls_weight, mm_weight):
        '''
        initial GPS predictor using positive training set, pls_weight vector and mm_weight vector
        :param plist: (list) positive peptides list
        :param pls_weight:  (list) pls_weight vector
        :param mm_weight:   (list) mm_weight vector
        '''
        self.alist = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                      'V', 'B', 'Z', 'X', '*']
        self.plist = plist
        self.pls_weight = np.array(pls_weight).flatten()
        self.mm_weight = np.array(mm_weight).flatten()

        self.__count_matrix = self._plist_index()
        self.__mm_matrix, self.__mm_intercept = self._mmweight2matrix()

    def predict(self, query_peptide, loo=False):
        '''
        return the gps score for the query peptide
        :param query_peptide: (str) query peptide
        :param loo: (bool) if true, count_matrix will minus 1 according to the amino acid in each position in query peptide
        :return: gps score
        '''
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(self.__count_matrix)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        pls_count_matrix = (count_clone.T * self.pls_weight).T / (len(self.plist) - rm_num)
        return np.sum(matrix * pls_count_matrix) + self.__mm_intercept

    def generatePLSdata(self, query_peptide, loo=False):
        '''
        generate the pls vector of query peptide
        :param query_peptide: (str) query peptide
        :param loo: (bool) if true, the count_matrix will minus 1 according to the amino acid in each position in query peptide
        :return: (np.ndarray) the vector of feature for each position
        '''
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo:
                count_clone[i, self.alist.index(a)] -= 1

            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        count_clone = (count_clone.T * self.pls_weight).T
        return np.sum(matrix * count_clone / (len(self.plist) - rm_num), 1)

    def generateMMdata(self, query_peptide, loo=False):
        count_clone = self.__count_matrix * len(self.plist)

        indicator_matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            indicator_matrix[i, self.alist.index(a)] = 1

        rm_num = 1 if loo else 0

        count_clone /= (len(self.plist) - rm_num)

        pls_count_matrix = (count_clone.T * self.pls_weight).T

        m = np.dot(indicator_matrix.T, pls_count_matrix) * self.__mm_matrix

        m += m.T

        np.fill_diagonal(m, np.diag(m) / float(2))

        iu1 = np.triu_indices(m.shape[0])

        return m[iu1]

    def getcutoff(self, randompeplist, sp=[0.98, 0.95, 0.85]):
        '''
        return cutoffs using 10000 random peptides as negative
        :param randompeplist: (list) random generated peptides
        :param sp: (float list) sp to be used for cutoff setting
        :return: (float list) cutoffs, same lens with sp
        '''
        rand_scores = sorted([self.predict(p) for p in randompeplist])
        cutoffs = np.zeros(len(sp))
        for i, s in enumerate(sp):
            index = np.floor(len(rand_scores) * s).astype(int)
            cutoffs[i] = rand_scores[index]
        return cutoffs

    def _plist_index(self):
        '''
        return the amino acid frequency on each position, row: position, column: self.alist, 61 x 24
        :return: count matrix
        '''
        n, m = len(self.plist[0]), len(self.alist)
        count_matrix = np.zeros((n, m))
        for i in range(n):
            for p in self.plist:
                count_matrix[i][self.alist.index(p[i])] += 1
        return count_matrix / float(len(self.plist))

    def _mmweight2matrix(self):
        '''
        convert matrix weight vector to similarity matrix, 24 x 24, index order is self.alist
        :return:
        '''
        aalist = self.getaalist()
        mm_matrix = np.zeros((len(self.alist), len(self.alist)))
        for n, d in enumerate(aalist):
            value = self.mm_weight[n + 1]  # mm weight contain intercept
            i, j = self.alist.index(d[0]), self.alist.index(d[1])
            mm_matrix[i, j] = value
            mm_matrix[j, i] = value
        return mm_matrix, self.mm_weight[0]

    def getaalist(self):
        '''return aa-aa list
        AA: 0
        AR: 1
        '''
        aa = [self.alist[i] + self.alist[j] for i in range(len(self.alist)) for j in range(i, len(self.alist))]
        return aa


def storage10features(
    storehouse,
    sample_ids,
    protein_ids,
    group_ids,
    folds,
    match_statuses,
    acf,
    aaindex,
    obc,
    cksaap,
    pseaac,
    label,
):
    storehouse = storehouse
    print('ACF', len(acf[0]))
    print('AAINDEX', len(aaindex[0]))
    print('OBC', len(obc[0]))
    print('CKSAAP', len(cksaap[0]))
    print('PSEAAC', len(pseaac[0]))
    feature_data = {
        'ACF': [],
        'AAINDEX': [],
        'OBC': [],
        'CKSAAP': [],
        'PSEAAC': []
        }

    def process_feature(
        feature_name,
        sample_id,
        protein_id,
        group_id,
        fold,
        match_status,
        feature_values,
        label,
    ):
        row = {
            'Sample_ID': sample_id,
            'UniProt_ID': protein_id,
            'Group_ID': group_id,
            'Fold': fold,
            'Match_Status': match_status,
        }
        for i, val in enumerate(feature_values):
            row[f'{i + 1}'] = val
        row['Label'] = label
        return row

    for iter, sample_id in enumerate(sample_ids):
        metadata = (
            sample_id,
            protein_ids[iter],
            group_ids[iter],
            folds[iter],
            match_statuses[iter],
        )
        feature_data['ACF'].append(process_feature('ACF', *metadata, acf[iter], label[iter]))
        feature_data['AAINDEX'].append(process_feature('AAINDEX', *metadata, aaindex[iter], label[iter]))
        feature_data['OBC'].append(process_feature('OBC', *metadata, obc[iter], label[iter]))
        feature_data['CKSAAP'].append(process_feature('CKSAAP', *metadata, cksaap[iter], label[iter]))
        feature_data['PSEAAC'].append(process_feature('PSEAAC', *metadata, pseaac[iter], label[iter]))

    for feature_name, data in feature_data.items():
        df = pd.DataFrame(data)
        df.to_excel(f"./six_features/{feature_name}.xlsx", index=False)



import pandas as pd
file_path = './train_data_with_ids.xlsx'
pepdic = pd.read_excel(file_path)
peplist = list(pepdic['Sequence'])
sample_ids = list(pepdic['Sample_ID'])
protein_ids = list(pepdic['UniProt_ID'])
group_ids = list(pepdic['Group_ID'])
folds = list(pepdic['Fold'])
match_statuses = list(pepdic['Match_Status'])
Label = list(pepdic['Label'])
ACF, AAindex=acf(peplist)
d=0
binary=be1(peplist,d)
km=1
CKSAAPs=kmors(peplist,km,m='l')
PseAAC=aac1(peplist)
storage10features(
    './',
    sample_ids,
    protein_ids,
    group_ids,
    folds,
    match_statuses,
    ACF,
    AAindex,
    binary,
    CKSAAPs,
    PseAAC,
    Label,
)
