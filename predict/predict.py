import numpy as np
import tqdm
import os
import pandas as pd
global nop, non
from pandas import DataFrame
from pandas import Series
import torch.nn as nn
from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def readweight(weight_file):
    weight = None
    with open(weight_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 2 - 1:
                weight = np.array([float(x) for x in line.rstrip().split('\t')])
    return weight


def acf(list30):
    global acl, aal, acn, aan, acsl, aasl
    pos = DataFrame(list30)
    all_ = pos
    ll, j = 0, 0
    na = {}
    vvl = []
    with open('./models/top10.txt', 'r') as fout:
        line = fout.readline()
        line = line.rstrip()[5:]
        aalist = line.split('\t')
        for line in fout:
            ll = ll + 1
            line = line.rstrip()
            na[line.split('\t')[0]] = line.split('\t')[1:]
            vvl.append(line.split('\t')[0])

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
            s = list(ss)
            for i, ii in enumerate(aalist):
                s = [(na[k][i]) if x == ii else x for x in s]
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

    all_['doc2num'] = all_[0].apply(lambda ss: doc2num(ss))
    all_['doc2num1'] = all_[0].apply(lambda ss: doc2num1(ss))
    xy = np.array(list(all_['doc2num']), dtype=np.float64)
    x2 = xy.tolist()
    x = np.array(list(all_['doc2num1']), dtype=np.float64)
    x3 = x.tolist()
    acl, aal = x2, x3
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

    mm_weight = readweight('./models/BLOSUM62R.txt')  # 1th is intercept

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


class CustomClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(CustomClassificationHead, self).__init__()
        self.dense1 = nn.Linear(hidden_size, 512)
        self.ln = nn.LayerNorm(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(512, num_labels)

    def forward(self, features):
        # Match the training script by classifying from the ESM CLS token.
        x = features[:, 0, :]
        x = self.dense1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2, max_seq_length=51):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(22, input_size)
        self.position_embedding = nn.Embedding(max_seq_length, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        word_embeddings = self.embedding(x)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = word_embeddings + position_embeddings
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(embeddings, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def seq2num(seqlist):
    out = []
    transdic = {'A': 8, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 0, 'L': 9, 'M': 10,
                'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '*': 20, 'B': 21}
    for seq in seqlist:
        seq = seq.replace('U', '*').replace('X', '*')
        vec = [transdic[i] for i in seq]
        out.append(vec)
    out = np.array(out)
    return out


class DNN(nn.Module):
    def __init__(self, size, hidden_sizes, dropout_rate, use_bn):
        super(DNN, self).__init__()
        layers = []
        input_size = size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class hybridKla(nn.Module):
    def __init__(self, size):
        super(hybridKla, self).__init__()
        self.fc1 = nn.Linear(size, 128)
        nn.init.ones_(self.fc1.bias)
        self.fc2 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(self.fc2(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc3(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc4(x))
        return x


def load_models(hidden_config, model_weights_path):
    models = {}
    device = torch.device("cpu")
    for feature_name, config in hidden_config.items():
        model = DNN(
            size=size_dist[feature_name],
            hidden_sizes=config['hidden'],
            dropout_rate=config['dropout'],
            use_bn=config['bn']
        )

        weights_path = f"{model_weights_path}/{feature_name}.pth"
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()  # Set to evaluation mode
        models[feature_name] = model

    return models


def inference(models, feature_data):
    predictions = {}
    device = torch.device("cpu")
    with torch.no_grad():
        for feature_name, model in models.items():
            if feature_name in feature_data:
                x = torch.FloatTensor(feature_data[feature_name]).to(device)
                pred = model(x)
                predictions[feature_name] = pred.cpu().numpy()

    return predictions


size_dist = {
    'ACF': 510,
    'AAINDEX': 510,
    'CKSAAP': 441,
    'OBC': 1122,
    'PSEAAC': 20
}
hidden_config = {
    'ACF': {'hidden': [1024, 512], 'dropout': 0.3, 'bn': True},
    'AAINDEX': {'hidden': [512, 256], 'dropout': 0.4, 'bn': True},
    'OBC': {'hidden': [2048, 512], 'dropout': 0.5, 'bn': True},
    'CKSAAP': {'hidden': [768, 256], 'dropout': 0.4, 'bn': True},
    'PSEAAC': {'hidden': [64, 32], 'dropout': 0.2, 'bn': False}
}

def predict_with_batches(sequences, batch_size=128, show_progress=True):
    """
    Perform prediction on sequences in batches with progress tracking
    
    Args:
        sequences: List of protein sequences to predict
        batch_size: Size of each batch (default: 128)
        show_progress: Whether to show progress bar (default: True)
    
    Returns:
        numpy array of prediction scores
    """
    device = torch.device("cpu")
    total_sequences = len(sequences)
    all_predictions = []
    
    # Load models
    print("Loading models...")
    
    # Load LSTM model
    input_size = 128
    hidden_size = 256
    num_layers = 6
    num_classes = 2
    dropout_rate = 0.2
    lstmmodel = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout_rate)
    lstmmodel_path = f'./models/LSTM.pth'
    lstmmodel.load_state_dict(torch.load(lstmmodel_path, map_location=device))
    lstmmodel.eval()
    
    # Load ESM2 model
    model_checkpoint = "./models/ESM2/"
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels,
                                                           hidden_dropout_prob=0.3, classifier_dropout=0.4)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    hidden_size = model.config.hidden_size
    model.classifier = CustomClassificationHead(hidden_size, num_labels)

    safetensors_path = f"./models/ESM2/model.safetensors"
    weights = load_file(safetensors_path)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    
    # Load feature models
    model_weights_path = "./models"
    models = load_models(hidden_config, model_weights_path)
    
    # Load hybrid model
    feature_size = 7  # ESM2 + LSTM + 5 handcrafted-feature models
    hybrid_model = hybridKla(feature_size).to(device)
    hybrid_model.load_state_dict(torch.load('./models/Meta_model.pth', map_location=device))
    hybrid_model.eval()
    
    print("Models loaded successfully.")
    
    # Create batches
    num_batches = (total_sequences + batch_size - 1) // batch_size
    batch_iterator = range(num_batches)
    
    if show_progress:
        batch_iterator = tqdm.tqdm(batch_iterator, desc="Predicting", total=num_batches)
    
    # Process each batch
    for batch_idx in batch_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_sequences)
        batch_peplist = sequences[start_idx:end_idx]
        
        # Feature extraction
        s_feature = []
        acf_result = acf(batch_peplist)
        s_feature.extend(acf_result)
        d = 0
        s_feature.append(be1(batch_peplist, d))
        km = 1
        s_feature.append(kmors(batch_peplist, km, m='l'))
        s_feature.append(aac1(batch_peplist))
        
        # LSTM prediction
        with torch.no_grad():
            lstmout = lstmmodel(torch.tensor(seq2num(batch_peplist), device=device))
            lstmoutput = torch.softmax(lstmout, dim=1)[:, 1]
        
        # ESM2 prediction
        tokenized = tokenizer(
            batch_peplist,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.no_grad():
            outputs_direct = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_direct = outputs_direct.logits
        esm_fold_predictions_direct = F.softmax(logits_direct, dim=-1)[:, 1]
        
        # Feature model predictions
        feature_data = {
            'ACF': s_feature[0],
            'AAINDEX': s_feature[1],
            'OBC': s_feature[2],
            'CKSAAP': s_feature[4],
            'PSEAAC': s_feature[5]
        }
        predictions = inference(models, feature_data)
        
        # Combine predictions
        lstmoutput = lstmoutput.unsqueeze(1).numpy()
        esm_fold_predictions_direct = esm_fold_predictions_direct.unsqueeze(1).numpy()
        for key in predictions:
            predictions[key] = torch.tensor(predictions[key]).numpy()
            
        new_array = []
        for j in range(len(lstmoutput)):
            row = [esm_fold_predictions_direct[j][0]]
            row.append(lstmoutput[j][0])
            for key in predictions:
                row.append(predictions[key][j][0])
            new_array.append(row)
            
        new_array = torch.tensor(new_array, dtype=torch.float32).to(device)
        
        # Final prediction
        with torch.no_grad():
            batch_predictions = hybrid_model(new_array).cpu().numpy().flatten()
        
        all_predictions.extend(batch_predictions.tolist())
        
        if show_progress:
            batch_iterator.set_postfix({"Processed": f"{end_idx}/{total_sequences}"})
    
    return np.array(all_predictions)

if __name__ == "__main__":
    # df = pd.read_excel('./data.xlsx')
    # peplist = list(df['sequence'])
    peplist = [
        'VVRTWRLNERHYGGLTGLNKAETAAKHGEAQVKIWRRSYDVPPPPMEPDHP',
        'RKFLKEEQQLRCQEREQQLRQDRDRKFREEEQQLSRQERDRKFREEEQQVR'
    ]

    results = predict_with_batches(peplist, batch_size=128, show_progress=True)
    print(f"Predictions for {len(peplist)} sequences:")
    print(results)
    
    # Save results to Excel file
    # output_df = pd.DataFrame({
    #     'Sequence': peplist,
    #     'Prediction_Score': results
    # })
    #
    # output_path = './prediction_results.xlsx'
    # output_df.to_excel(output_path, index=False)
    # print(f"Results saved to {output_path}")
