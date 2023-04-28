from datasets import load_from_disk, load_dataset ,Features, Sequence, Value, Array2D, Array3D, Dataset, DatasetDict
from datasets.features import ClassLabel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import os
import json
import transformers
from PIL import Image

class RVL:
    def __init__(self,opt) -> None:
        self.opt = opt

        # step1: create config, tokenizer, and processor
        self.config = AutoConfig.from_pretrained(opt.layoutlm_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)  # get sub
        self.processor = AutoProcessor.from_pretrained(opt.layoutlm_dir,tokenizer=self.tokenizer, apply_ocr=False)    # wrap of featureExtract & tokenizer

        self.cpu_num = opt.num_cpu
        # step 2.1: get raw ds (already normalized bbox, img object)
        raw_train, raw_test = self.get_raw_ds(opt.rvl_train), self.get_raw_ds(opt.rvl_test)

        # step 2.2, get labeled ds (get label list, and map label dataset)
        _,_, opt.label_list = self._get_label_map(raw_train)
        opt.num_labels = len(opt.label_list)    # 54; 27 pairs
        print('labels:',opt.label_list)
        self.class_label = ClassLabel(num_classes=opt.num_labels, names = opt.label_list)

        # map label ds
        train_label_ds, test_label_ds = self.get_label_ds(raw_train), self.get_label_ds(raw_test)

        # step 3: prepare for getting trainable data (encode and define features)
        
        trainable_train_ds, trainable_test_ds = self.get_preprocessed_ds(train_label_ds),self.get_preprocessed_ds(test_label_ds)
        
        print(trainable_train_ds)
        # print(trainable_train_ds[0])
        self.trainable_ds = DatasetDict({
            "train" : trainable_train_ds , 
            "test" : trainable_test_ds 
        })


    # load raw dataset (including image object)
    def get_raw_ds(self, ds_path):
        def _load_imgs_obj(sample):
            # 1) load img obj
            sample['images'],size = self._load_image(sample['image'])
            # 2) normalize bboxes using the img size 
            sample['bboxes'] = [self._normalize_bbox(bbox, size) for bbox in sample['bboxes']]
            return sample

        # 1 load raw data
        raw_ds = load_from_disk(ds_path) # {'tokens': [], 'tboxes': [], 'bboxes': [], 'block_ids':[], 'image': image_path}
        raw_ds = Dataset.from_dict(raw_ds[:300])    # obtain subset for experiment/debugging use
        # 2 load img obj and norm bboxes 
        ds = raw_ds.map(_load_imgs_obj, num_proc=self.cpu_num, remove_columns=['tboxes']) # load image objects

        return ds


    def get_preprocessed_ds(self,ds):
        def _preprocess(batch):
            # 1) encode words and imgs
            encodings = self.processor(images=batch['images'],text=batch['tokens'], boxes=batch['bboxes'],
                                       truncation=True, padding='max_length', max_length=self.opt.max_seq_len)
            # 2) add position_ids
            position_ids = []
            for i, block_ids in enumerate(batch['block_ids']):
                word_ids = encodings.word_ids(i)
                rel_pos = self._get_rel_pos(word_ids, block_ids)
                position_ids.append(rel_pos)
            encodings['position_ids'] = position_ids
            # 3) add labels separately for sequence classification
            encodings['labels'] = batch['label']
            return encodings
        
        features = Features({
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'position_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            # 'labels': Value(dtype='int64'),
            'labels': self.class_label, # this is sequence classification, so only value!
        })
        # processed_ds = ds.map(_preprocess, batched=True, num_proc=self.cpu_num, 
        #     remove_columns=['id','tokens', 'bboxes','ner_tags','block_ids','image'], features=features).with_format("torch")
        processed_ds = ds.map(_preprocess, batched=True, num_proc=self.cpu_num, 
            remove_columns=ds.column_names, features=features).with_format("torch")
    
        # process to: 'input_ids', 'position_ids','attention_mask', 'bbox', 'labels', 'pixel_values']
        return processed_ds


    def get_label_ds(self,ds):
        label_ds = ds.cast_column('label', self.class_label)
        # def map_label2id(sample):
        #     sample['label'] = self.class_label.str2int(sample['label'])
        #     return sample
        # label_ds = ds.map(map_label2id, num_proc=self.cpu_num)
        return label_ds


    # get label list
    def _get_label_map(self,ds):
        # features = ds.features
        # column_names = ds.column_names
        # if isinstance(features['label'].feature, ClassLabel):
        #     label_list = features['label'].feature.names
        #     # No need to convert the labels since they are already ints.
        #     id2label = {k: v for k,v in enumerate(label_list)}
        #     label2id = {v: k for k,v in enumerate(label_list)}
        # else:
        label_list = list(set(ds['label']))   # this invokes another function
        label_list.sort()
        id2label = {k: v for k,v in enumerate(label_list)}
        label2id = {v: k for k,v in enumerate(label_list)}
        return id2label, label2id, label_list

    def _load_image(self,image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)

    def _normalize_bbox(self,bbox, size):
        return [
            int(1000 * bbox[0] / size[0]),
            int(1000 * bbox[1] / size[1]),
            int(1000 * bbox[2] / size[0]),
            int(1000 * bbox[3] / size[1]),
        ]
    def _get_line_bbox(self,bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox
    
    def _quad_to_box(self, quad):
    # test 87 is wrongly annotated
        box = (
            max(0, quad["x1"]),
            max(0, quad["y1"]),
            quad["x3"],
            quad["y3"]
        )
        if box[3] < box[1]:
            bbox = list(box)
            tmp = bbox[3]
            bbox[3] = bbox[1]
            bbox[1] = tmp
            box = tuple(bbox)
        if box[2] < box[0]:
            bbox = list(box)
            tmp = bbox[2]
            bbox[2] = bbox[0]
            bbox[0] = tmp
            box = tuple(bbox)
        return box

    def _get_rel_pos(self,word_ids, block_ids):   # [None, 0, 1, 2, 2, 3, None]; [1,1,2,2] which is dict {word_idx: block_num}
        res = []
        rel_cnt = self.config.pad_token_id+1
        prev_block = 1
        for word_id in word_ids:
            if word_id is None:
                res.append(self.config.pad_token_id)
                continue
            else:
                curr_block = block_ids[word_id]   # word_id is the 0,1,2,3,.. word index;
                if curr_block != prev_block:
                    # set back to 0; 
                    rel_cnt = self.config.pad_token_id+1
                    res.append(rel_cnt) # operate
                    # reset prev_block
                    prev_block = curr_block
                else:
                    res.append(rel_cnt)
            rel_cnt+=1
        return res

if __name__ == '__main__':
    # Section 1, parse parameters
    mydata = FUNSD(None)
    test_dataset = mydata.get_data(split='test')
    print(test_dataset)
    doc1 = test_dataset[0]
    print(doc1['input_ids'])

    '''
    Dataset({
        features: ['input_ids', 'attention_mask', 'bbox', 'labels', 'pixel_values'],
        num_rows: 100
    })
    '''
