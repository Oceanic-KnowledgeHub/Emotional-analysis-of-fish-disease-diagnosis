import time
import torch
import numpy as np
import argparse
from importlib import import_module
import utils
import train
import train_ATAE
import train_our

parser=argparse.ArgumentParser(description='Aspect-Based')
parser.add_argument('--model',type=str,default='Bert-Linear',help='choose a model')     #BERT_Linear
# parser.add_argument('--model',type=str,default='Bert-BiLSTM',help='choose a model')       #BERT_BiLSTM
# parser.add_argument('--model',type=str,default='Bert-CNN',help='choose a model')           #BERT_CNN
# parser.add_argument('--model',type=str,default='Bert-BiLSTM_AG_CNN',help='choose a model')           #Bert_BiLSTM_AG_CNN
# parser.add_argument('--model',type=str,default='MAP',help='choose a model')
# parser.add_argument('--model',type=str,default='ATAE_LSTM',help='choose a model')
# parser.add_argument('--model',type=str,default='IAN',help='choose a model')
# parser.add_argument('--model',type=str,default='RAM',help='choose a model')
# parser.add_argument('--model',type=str,default='MAP_Feature_IAN',help='choose a model')
# parser.add_argument('--model',type=str,default='TNet',help='choose a model')
args=parser.parse_args()



if __name__=='__main__':
    dataset='Data'
    model_name=args.model
    x=import_module('Model.'+model_name)
    config=x.Config(dataset)
    # print(config.class_list)


    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic=True

    start_time=time.time()
    print('加载数据集')
    train_data,dev_data,test_data=utils.build_data(config)
    train_iter=utils.build_iterator(train_data,config)
    dev_iter=utils.build_iterator(dev_data,config)
    test_iter=utils.build_iterator(test_data,config)
    #
    # time_spend=utils.get_time_spend(start_time)
    # print('准备数据时间：',time_spend)
    #
    # # print(len(train_data)).,
    #
    #
    # # print(gen_iter)
    #
    #
    #训练
    # model = x.Bert_BiLSTM(config).to(config.device[0])         #BERT_BiLSTM
    model = x.Bert_Linear(config).to(config.device[0])         #BERT_Linear
    # model = x.Bert_CNN(config).to(config.device[0])         #BERT_CNN
    # model = x.Bert_BiLSTM_AG_CNN(config).to(config.device[0])           #Bert_BiLSTM_AG_CNN
    # model = x.MAP(config).to(config.device[0])         #MAE
    # model = x.MAP_Feature(config).to(config.device[0])         #MAE_Feature
    # model = x.RAM(config).to(config.device[0])         #RAM
    # model = x.MAP_Feature_IAN(config).to(config.device[0])    #Our
    # model = x.ATAE_LSTM(config).to(config.device[0])
    # model = x.IAN(config).to(config.device[0])
    # model = x.TNet(config).to(config.device[0])

    train_ATAE.train(config, model, train_iter, dev_iter, test_iter)
    # train_our.train(config, model, train_iter, dev_iter, test_iter)

    train_ATAE.test(config,model,test_iter)
    # train_our.test(config, model, test_iter)

    # #加载
    # model_test = model.load_state_dict(torch.load('../MAP/modelpath/MAP.ckpt'))
