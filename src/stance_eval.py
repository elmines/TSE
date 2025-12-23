import torch
import random
import os
import argparse
import numpy as np
import warnings

import test_utils.data_helper as test_dh
import train_utils.data_helper as train_dh


from test_utils import modeling
from train_utils import metrics, model_utils

# import train_utils.preprocessing as pp
# import train_utils.data_helper as dh
# from train_utils import modeling, metrics, model_utils

warnings.filterwarnings('ignore')


def train():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-s', '--seed', help='Random seed', required=False)
    parser.add_argument('-m', '--model_select', help='Model name', required=False)
    parser.add_argument('-mod_dir', '--model_dir', help='Saved model dir', required=False)
    parser.add_argument('-test', '--test_data', help='Name of the test data file', default=None, required=False)
    args = vars(parser.parse_args())

    # gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    
    # load config file
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    
    # print parameters in the log file
    random_seeds = []
    random_seeds.append(int(args['seed']))
    outdir = args['model_dir']
    model_select = args['model_select']
    batch_size = int(config['batch_size'])
    print("Model: ",model_select)
    print("Batch size: ",config['batch_size'])
    print(60*"#")

    # load test set
    file = [args['test_data']] * 3
    if model_select.startswith('bert'):
        x_train_all, x_val_all, x_test_all, x_train_aux_all, _ = train_dh.load_dataset(file, model_select, config)
    else:
        raise ValueError("Only support bert")

    if model_select.startswith('bert'):
        _, _, _, y_test, _, testloader = train_dh.data_loader(x_test_all, batch_size, 'test', model_select)   
    else:
        raise ValueError("Only BERT supported now")
    y_test = y_test.to(device)


    # test
    for seed in random_seeds:    
        print("current random seed: ", seed)
        
        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # model setup
        weight = os.path.join(outdir,model_select+'_seed{}.pt'.format(seed))
        if model_select in ['bert','bertweet']:
            model = modeling.bert_classifier(config, model_select).to(device)
        else:
            raise ValueError("Only bert models supported")
        model.load_state_dict(torch.load(weight), strict=False)

        # evaluation
        model.eval()
        
        best_test_micro, best_test_macro = [], []
        with torch.no_grad():
            preds = model_utils.model_preds(testloader, model, device, model_select)

            # micro-averaged F1
            f1_average = metrics.train_compute_f1(preds, y_test)
            best_test_micro.append(f1_average)

            # macro-averaged F1
            preds_list = train_dh.sep_test_set(preds) 
            y_test_list = train_dh.sep_test_set(y_test)
            temp_list = []
            for ind in range(len(y_test_list)):
                f1_average = metrics.train_compute_f1(preds_list[ind], y_test_list[ind])
                temp_list.append(f1_average)
            best_test_macro.append(sum(temp_list)/len(temp_list))
                
        print("Best macro          test results on SemEval-2016: " + ",".join(map(str, [temp_list[0]])))
        print("Best macro          test results on COVID-19: " + ",".join(map(str, [temp_list[1]])))
        print("Best macro          test results on argmin: " + ",".join(map(str, [temp_list[2]])))
        print("Best macro          test results on PStance: " + ",".join(map(str, [temp_list[3]])))
        print("Best macro-of-macro test results: " + ",".join(map(str, best_test_macro)))
        print("Best micro          test results: " + ",".join(map(str, best_test_micro)))
    
if __name__ == "__main__":
    train()