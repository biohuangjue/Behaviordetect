import argparse
from model_1 import train_and_eval
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Train and evaluate a text classification model.')
    parser.add_argument('-m', '--model_type', choices=['CNN', 'RNN_LSTM', 'MLP', 'RNN_GRU', 'BERT'], default='CNN', help='The type of the model.')
    parser.add_argument('-i', '--init_method', choices=['zero', 'normal', 'kaiming', 'ortho', 'Xavier'], default='Xavier', help='The initialization method for the model parameters.')
    parser.add_argument('-e', '--early_stopping', choices=['true', 'false'], default='false', help='Whether to use early stopping.')
    #parser.add_argument('-d', '--data_path', type=str, required=True, help='Path to the data file.')
    # 解析命令行参数
    args = parser.parse_args()

    # 加载数据
    data, labels = load_data("C:/Users/A/Desktop/guangyichuan/dataset/cuoshou")

    # 训练和评估模型
    train_and_eval(args.model_type, args.init_method, args.early_stopping, data, labels)

import pickle

def load_data(data_path):
    with open(data_path + '/data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open(data_path + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    return data, labels

if __name__ == '__main__':
    main()