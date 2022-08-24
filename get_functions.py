import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional
import time
import math
from typing import Dict, List, Mapping, Optional, Union
import os
import platform

from pororo.tasks.utils.download_utils import download_or_load
import torch
from fairseq.models.roberta import RobertaHubInterface, RobertaModel
from fairseq import hub_utils
import torch.nn.functional as F

    

def load_input():
    """load input dataframe
    Args:
    
    Returns:
        df(DataFrame): preprocesed news text dataframe
    """
    folder = 'Data' # Your Current directory
    file_name = f'dfs.txt' # Your input data name
    file_path = os.path.join(folder, file_name)
    
    df = pd.read_csv(file_path, delimiter = '\t', header = None)
    df[0] = df[0].str.replace(pat =  r'[{}<>]', repl = r'', regex = True)
    print('-----DATA LOADED-----')
    return df

def load_model():
    """load pretrained Pororo NER model using RobertaHubInterface
    
    Args:
    
    Returns:
        Pororo Model for NER
    """
    task='ner'
    lang: str = "ko"
    model: Optional[str] = "charbert.base.ko.ner"
    model_name: str = f"bert/{model}"
    kwargs = {}

    ckpt_dir = download_or_load(model_name, lang)

    x = hub_utils.from_pretrained(
        ckpt_dir,
        "model.pt",
        ckpt_dir,
        **kwargs,
    )

    model = RobertaHubInterface(
        x["args"],
        x["task"],
        x["models"][0],
    )
    print('-----MODEL LOADED-----')
    return model

def batch_data_preparation(dataframe, model):
    """Take encoded tokens input and return max padded ones
    
    Args:
        dataframe(DataFrame): Raw news text input
        model : pretrained ner model
        
    Returns:
        batch_token(list): max padded tokens
    """
    def tokenizer(sent, add_special_tokens=True):
        """Take text input and return tokenized ones
    
        Args:
            sent(str) : preprocessed news text input (by sentence)
            add_special_tokens(bool): special token flag
            model : pretrained ner model
        
        Returns:
            result(str): tokenized input
            tokens(list): encoded tokens
        """
        # tokenize
        x = sent
        x = x.strip()

        if len(x) == 0:
            result = ""
        else:
            # x = [c for c in re.sub("\s+", " ", x)]
            x = list(x)

            result = list()
            for i in range(len(x)):
                if x[i] == " ":
                    x[i + 1] = f"▁{x[i+1]}"
                    continue # 
                else:
                    result.append(x[i])
            result[0] = f"▁{result[0]}"
            result = " ".join(result)
            bpe_sentence = result

        if add_special_tokens:
            bpe_sentence = f"<s> {result} </s>"

        # tokens in number code
    
        tokens = model.task.source_dictionary.encode_line(
            bpe_sentence,
            append_eos=False,
            add_if_not_exist=False,
        )
        tokens = tokens[:512]
        tokens = tokens.long()
        return result, tokens
    
    train_x = []
    for i in range(len(dataframe[0])):
        xs = np.array(tokenizer(''.join(dataframe[0][i]), model), dtype='object').T[1]
        train_x.append(xs)
    print('-----BATCH MAKING DONE-----')
    max_token_len = max(list(map(len, train_x)))
    list_token_pad = list(map(lambda token: torch.nn.functional.pad(token, (0, max_token_len - len(token)), value=1), train_x))
    batch_token = torch.stack(list_token_pad)
    
    print('-----BATCH DATA PREPARED-----')
    return batch_token,
    
def get_predicted_label(batch, num_start, num_end, model):
    """get predicted label using pretrained ner model to make answer sheet
    
    Args:
        batch(list): max padded batch tokens
        num_start(int): starting number
        num_end(int): finishing number
        model: loaded pretraied model
    
    Returns;
        pr(list): model-predicted labels list
    """
    pr = []
    print('--- This Process might take SOME time ---')
    for i in range(num_start, num_end):
        prs = []
        output = model.predict("sequence_tagging_head", batch[i])[:, 1:-1, :]
        logits = output.detach().cpu().numpy()
        if i % 10 == 0:
            print(f'---{i}th / {num_end} text is predicted---')
            
        
        for j in range(len(logits[0])):
            prs.append(int(np.argmax(F.softmax(torch.tensor(logits[0][j])))))
        pr.append(prs)
        
    return pr

def find_all(str_, sub):
    start = 0
    while True:
        str_ = str_.replace(" ", "")
        start = str_.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)
        
def change_events(start_symbol, end_symbol, origin, predicted):
    '''
    Args:
        start_symbol : 이벤트/제품 라벨링 phrase의 시작 기호
        end_symbol : 이벤트/제품 라벨링 phrase의 마지막 기호
    
    Output:
        모델이 예측한 태그 중 이벤트/제품 라벨링 되어야 할 자리에 이벤트/제품 태그로 replace
    '''
    locs = [list(find_all(origin, start_symbol)), list(find_all(origin, end_symbol))]
    events = []
    for i in range(len(locs[0])):
        indexes1 = locs[0]
        indexes2 = locs[1]
        predicted2 = predicted
        predicted2[indexes1[i] + (-4)*i] = 19
        predicted2[(indexes1[i] + (-4)*i + 1) : (indexes2[i] -3 + (-4)*i + 1)] = [14] * (indexes2[i] - indexes1[i] -3)
        predicted = predicted2
        
    return predicted

def change_products(start_symbol, end_symbol, origin, predicted):
    '''
    Args:
        start_symbol : 이벤트/제품 라벨링 phrase의 시작 기호
        end_symbol : 이벤트/제품 라벨링 phrase의 마지막 기호
    
    Output:
        모델이 예측한 태그 중 이벤트/제품 라벨링 되어야 할 자리에 이벤트/제품 태그로 replace
    '''
    locs = [list(find_all(origin, start_symbol)), list(find_all(origin, end_symbol))]
    events = []
    for i in range(len(locs[0])):
        indexes1 = locs[0]
        indexes2 = locs[1]
        predicted2 = predicted
        predicted2[indexes1[i] + (-4)*i] = 16
        predicted2[(indexes1[i] + (-4)*i + 1) : (indexes2[i] -3 + (-4)*i + 1)] = [10] * (indexes2[i] - indexes1[i] -3)
        predicted = predicted2
        
    return predicted