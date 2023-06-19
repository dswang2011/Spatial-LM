import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import json
import Levenshtein as lev
import re
import string
import sys
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(ground_truth, prediction):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1




class MyInferencer:
    def __init__(self,opt):
        self.opt = opt


    def inference_for_classification(self, opt, model, mydata):
        trainable_ds = mydata.trainable_ds
        raw_ds = mydata.raw_ds
        train_dataloader = DataLoader(trainable_ds, batch_size=opt.batch_size)
        print('-- start to infer --')
        all_preds = []
        img_paths = []
        model.eval()
        with torch.no_grad():
            for batch_index, inputs in enumerate(train_dataloader):
                # move to the device
                inputs = {key: value.to(opt.device) for key, value in inputs.items()}
                # infer
                outputs = model(**inputs)
                # get label id
                batch_predictions = torch.argmax(outputs.logits, dim=-1).tolist()
                batch_imgs = [raw_ds[i + opt.batch_size*batch_index]['image'] for i in range(len(batch_predictions))]
                all_preds.extend(batch_predictions)
                img_paths.extend(batch_imgs)
        return img_paths,all_preds


    def inference_for_QA(self, model, mydata, save_to_file):
        # 1 load dataset
        # test_dataset = mydata.test_dataset
        loader_test = DataLoader(mydata.test_ds, batch_size=self.opt.batch_size*3)

        model.eval()
        res = []
        # 2. iterate and inference
        # for row in test_dataset:
        for batch_index, batch in enumerate(loader_test):
            print(batch_index)
            questionId = batch['questionId']
            input_ids = batch['input_ids'].to(self.opt.device)
            position_ids = batch['position_ids'].to(self.opt.device)
            attention_mask = batch['attention_mask'].to(self.opt.device)
            token_type_ids = batch['token_type_ids'].to(self.opt.device)
            bbox = batch['bbox'].to(self.opt.device)
            # start_positions = row['ans_start_positions'].to(opt.device)
            # end_positions = row['ans_end_positions'].to(opt.device)
            pixel_values = batch['pixel_values'].to(self.opt.device)

            with torch.no_grad():
                outputs = model(input_ids = input_ids, position_ids=position_ids,
                        token_type_ids=token_type_ids, bbox = bbox, attention_mask = attention_mask, 
                        pixel_values = pixel_values)
                # 3.1 get start_logits and end_logits
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                # 3.2: get largest logit for both
                for idx, (s,e) in enumerate(zip(start_logits,end_logits)):
                    predicted_start_idx = s.argmax(-1).item()    # item() is to get scalars
                    predicted_end_idx = e.argmax(-1).item()
                    # print("Predicted start idx:", predicted_start_idx)
                    # print("Predicted end idx:", predicted_end_idx)
                    # 4. get the text answer
                    answer = mydata.tokenizer.decode(input_ids[idx][predicted_start_idx:predicted_end_idx+1]) 
                    res.append({"questionId":questionId[idx].item(), "answer":answer})
        # save it
        res = json.dumps(res)
        self.save_res(save_to_file,res)
        return res


    def inference_and_evaluate(self, model, mydata):
        # 1 load dataset
        # test_dataset = mydata.test_dataset
        loader_test = DataLoader(mydata.trainable_ds['test'], batch_size=self.opt.batch_size*3)

        model.eval()
        res = []
        # 2. iterate and inference
        # for row in test_dataset:
        for batch_index, batch in enumerate(loader_test):
            # print(batch_index)
            
            with torch.no_grad():
                batch = {k:v.to(self.opt.device) for k,v in batch.items()}
                outputs = model(**batch)
                # 3.1 get start_logits and end_logits
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                # 3.2: get largest logit for both
                for idx, (s,e) in enumerate(zip(start_logits,end_logits)):
                    predicted_start_idx = s.argmax(-1).item()    # item() is to get scalars
                    predicted_end_idx = e.argmax(-1).item()
                    # print("Predicted start idx:", predicted_start_idx)
                    # print("Predicted end idx:", predicted_end_idx)
                    # 4. get the text answer
                    answer = mydata.tokenizer.decode(batch['input_ids'][idx][predicted_start_idx:predicted_end_idx+1]) 
                    res.append(answer)

        # test
        L_scores, F_scores = [],[]
        for cand_ans,pred in zip(mydata.raw_test['ans_strs'], res):
            print(pred, ' -v.s.- ',str(cand_ans))
            temp_l_scores, temp_f_scores = [],[]
            for ans in cand_ans:
                l_score = lev.ratio(ans.lower(),pred.lower()) # Levenshein sim
                f_score = f1_score(ans, pred) # f1 score
                temp_l_scores.append(l_score)
                temp_f_scores.append(f_score)

            L_scores.append(max(temp_l_scores))
            F_scores.append(max(temp_f_scores))

        avg_l_score = sum(L_scores) / len(L_scores)
        avg_f_score = sum(F_scores) / len(F_scores)

        res_dict = {'ANLS':avg_l_score,'F1': avg_f_score}
        print(str(res_dict))
        self.write_res('temp.txt', model.__class__.__name__ + '\n' )
        self.write_res('temp.txt',str(res_dict))
        return res_dict


    def write_res(self,path,content):
        with open(path,'a') as fw:
            fw.write(str(content))

