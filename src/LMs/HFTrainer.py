from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,classification_report

from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
import numpy as np
from transformers import DataCollatorForLanguageModeling


def pretrain(opt, model, mydata):
    # mlm= True uses masked language model; otherwise, causal LM (NTP); 
    # mydata.tokenizer.pad_token = tokenizer.eos_token  # no idea why??
    data_collator = DataCollatorForLanguageModeling(tokenizer=mydata.tokenizer, mlm=True, mlm_probability=opt.mlm_probability)

    # logging_steps = len(mydata.train_dataset)  //opt.batch_size
    trainable_ds = mydata.trainable_ds.shuffle(seed=88).train_test_split(test_size=opt.test_size)

    training_args = TrainingArguments(v
        output_dir = opt.output_dir,
        num_train_epochs = opt.epochs,
        learning_rate = opt.lr,
        per_device_train_batch_size = opt.batch_size,
        per_device_eval_batch_size = opt.batch_size,
        weight_decay = 0.01,
        warmup_ratio = 0.05,
        fp16 = True,    # make it train fast
        push_to_hub = False,
        # push_to_hub_model_id = f"layoutlmv3-finetuned-cord"        
        evaluation_strategy = "epoch",
        save_strategy="epoch",
        overwrite_output_dir=True,  # use only one dir
        prediction_loss_only = True,
        logging_dir='./logs',  
        # save_steps=5000,
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = trainable_ds['train'],
        eval_dataset = trainable_ds['test'],
        data_collator = data_collator,
        # config = model.config,
        # tokenizer = mydata.tokenizer
    )
    trainer.train()
    trainer.save_model(opt.save_path)


def train(opt, model, mydata):
    # mlm= True uses masked language model; otherwise, causal LM (NTP); 
    # logging_steps = len(mydata.train_dataset)  //opt.batch_size
    trainable_ds = mydata.trainable_ds

    training_args = TrainingArguments(v
        output_dir = opt.output_dir,
        num_train_epochs = opt.epochs,
        learning_rate = opt.lr,
        per_device_train_batch_size = opt.batch_size,
        per_device_eval_batch_size = opt.batch_size,
        # weight_decay = 0.01,
        # warmup_ratio = 0.05,
        fp16 = True,    # make it train fast
        push_to_hub = False,
        # push_to_hub_model_id = f"layoutlmv3-finetuned-cord"        
        evaluation_strategy = "epoch",
        save_strategy="epoch",
        overwrite_output_dir=True,  # use only one dir
        # prediction_loss_only = True,
        # logging_dir='./logs',  
        # save_steps=5000,
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = trainable_ds['train'],
        eval_dataset = trainable_ds['test'],
        compute_metrics = compute_metrics,
        # data_collator = data_collator,
        # config = model.config,
        # tokenizer = mydata.tokenizer
    )
    trainer.train()
    trainer.save_model(opt.save_path)


def compute_metrics(p,return_entity_level_metrics=False):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

