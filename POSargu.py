# Author: Robert Guthrie
# 作者：Robert Guthrie

import torch
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能
import torch.optim as optim       # 模型优化器模块
import numpy as np

training_data=[]
file=open('/Users/zoe/Documents/event_extraction/CRF++-0.58/example/sequence/all_argu.txt','r')
lines=file.readlines()
file.close()
sent=[]
label=[]
count=0
for line in lines:
    if len(line)<2:
        training_data.append((sent,label))
        sent=[]
        label=[]
    else:
        line = line.split()
        sent.append(line[0])
        label.append(line[2])
word_to_ix = {} # 单词的索引字典
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
print(len(training_data))

tag_to_ix = {'A_Agent': 0, 'A_Recipient': 1, 'A_Price': 2, 'A_Time-At-Beginning': 3, 'A_Time-Within': 4, 'A_Origin': 5, 'A_Defendant': 6, 'A_Time-Before': 7, 'A_Place': 8, 'A_Plaintiff': 9, 'A_Buyer': 10, 'A_Entity': 11, 'A_Destination': 12, 'A_Beneficiary': 13, 'A_Person': 14, 'A_Artifact': 15, 'A_Position': 16, 'A_Adjudicator': 17, 'A_Victim': 18, 'A_Org': 19, 'A_Crime': 20, 'A_Sentence': 21, 'A_Time-At-End': 22, 'A_Instrument': 23, 'A_Money': 24, 'A_Time-Holds': 25, 'A_Giver': 26, 'A_Prosecutor': 27, 'A_Seller': 28, 'A_Target': 29, 'A_Time-After': 30, 'A_Attacker': 31, 'A_Time-Ending': 32, 'A_Vehicle': 33, 'A_Time-Starting': 34, 'O': 35}
ix_to_tag = {0: 'A_Agent', 1: 'A_Recipient', 2: 'A_Price', 3: 'A_Time-At-Beginning', 4: 'A_Time-Within', 5: 'A_Origin', 6: 'A_Defendant', 7: 'A_Time-Before', 8: 'A_Place', 9: 'A_Plaintiff', 10: 'A_Buyer', 11: 'A_Entity', 12: 'A_Destination', 13: 'A_Beneficiary', 14: 'A_Person', 15: 'A_Artifact', 16: 'A_Position', 17: 'A_Adjudicator', 18: 'A_Victim', 19: 'A_Org', 20: 'A_Crime', 21: 'A_Sentence', 22: 'A_Time-At-End', 23: 'A_Instrument', 24: 'A_Money', 25: 'A_Time-Holds', 26: 'A_Giver', 27: 'A_Prosecutor', 28: 'A_Seller', 29: 'A_Target', 30: 'A_Time-After', 31: 'A_Attacker', 32: 'A_Time-Ending', 33: 'A_Vehicle', 34: 'A_Time-Starting', 35: 'O'}

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)
test_data=training_data[1400:]
training_data=training_data[1:1400]
EMBEDDING_DIM=100
HIDDEN_DIM=128
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
print(training_data[0])
print('Training data:',training_data[0][0])
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)
for epoch in range(20):  # 我们要训练50次，可以根据任务量的大小酌情修改次数。
    print(epoch)
    for sentence, tags in training_data:
        # 清除网络先前的梯度值，梯度值是Pytorch的变量才有的数据，Pytorch张量没有
        model.zero_grad()
        # 重新初始化隐藏层数据，避免受之前运行代码的干扰
        model.hidden = model.init_hidden()
        # 准备网络可以接受的的输入数据和真实标签数据，这是一个监督式学习
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        # 运行我们的模型，直接将模型名作为方法名看待即可
        tag_scores = model(sentence_in)
        # 计算损失，S9反向传递梯度及更新模型参数
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# 来检验下模型训练的结果
file=open('raw_file/argu_result.txt','w')
tcount=0
count=0
for sentence,tags in test_data:
    inputs = prepare_sequence(sentence, word_to_ix)
    tag_scores = model(inputs)
    for i in range(len(sentence)):
        count+=1
        x=tag_scores[i,:]
        y=x.data.cpu().numpy()
        file.write(ix_to_tag[np.argmax(y)])
        file.write('\t')
        file.write(tags[i])
        if ix_to_tag[np.argmax(y)] == tags[i]:
            tcount += 1
        file.write('\n')
    file.write('\n')
file.close()
print(tcount*1.0/count)
