
# from LMs.Roberta import RobertaClassifier
from LMs.LayoutLM import LayoutLMTokenclassifier
from LMs.LayoutLM import LayoutLM4DocVQA
from LMs.Roberta import RobertaTokenClassifier, RobertaForBinaryQA, RobertaForQA
from LMs.SpatialLM import SpatialLMForMaskedLM, SpatialLMForTokenclassifier, SpatialLMConfig, SpatialLMForSequenceClassification, SpatialLMForDocVQA
from transformers import AutoConfig, AutoModel
from LMs.layoutlmv3_disent import LayoutLMv3ForMaskedLM
from LMs.layoutlmv3_disent import LayoutLMv3ForTokenClassification as DiscentTokClassifier
from LMs.bert import BertTokenClassifier,BertForQA, BertSequenceClassifier, BertForBinaryQA
from LMs.layoutlmv2 import LayoutLMv2ForTokenClassification, LayoutLMv2Config
from LMs.layoutlmv1 import LayoutLMForTokenClassification, LayoutLMForSequenceClassification, LayoutLMForQuestionAnswering, LayoutLMForBinaryQA
import transformers
from LMs import layoutlmv3_disent
from LMs import layoutlmv3
from LMs import layoutlmv2

def setup(opt):
    print('network:' + opt.network_type)
    # if opt.network_type == 'roberta':
    #     model = RobertaClassifier(opt)
    # if opt.network_type in ['layoutlm', 'layoutlmv1']:
    #     if opt.task_type == 'token-classifier':
    #         model = LayoutLMTokenclassifier(opt)
    #     elif opt.task_type == 'docvqa':
    #         model = LayoutLM4DocVQA(opt)
    if opt.network_type == 'layoutlmv1':
        config = AutoConfig.from_pretrained(opt.layoutlm_dir)   # borrow config
        config.num_labels=opt.num_labels    # set label num
        if opt.task_type == 'token-classifier':
            model = LayoutLMForTokenClassification.from_pretrained(opt.layoutlm_dir,config=config)
        elif opt.task_type == 'sequence-classifier':
            model = LayoutLMForSequenceClassification.from_pretrained(opt.layoutlm_dir, config = config)
        elif opt.task_type == 'docvqa':
            model = LayoutLMForQuestionAnswering.from_pretrained(opt.layoutlm_dir,config=config)
        elif opt.task_type == 'docbqa':
            model = LayoutLMForBinaryQA.from_pretrained(opt.layoutlm_dir,config=config)
    elif opt.network_type == 'layoutlmv2':
        config = LayoutLMv2Config.from_pretrained(opt.layoutlm_dir)   # borrow config
        if opt.task_type == 'token-classifier':
            config.num_labels=opt.num_labels    # set label num
            model = LayoutLMv2ForTokenClassification.from_pretrained(opt.layoutlm_dir,config=config)
        elif opt.task_type == 'docvqa':
            model = layoutlmv2.LayoutLMv2ForQuestionAnswering.from_pretrained(opt.layoutlm_dir, config=config)
    elif opt.network_type == 'layoutlmv3':
        config = AutoConfig.from_pretrained(opt.layoutlm_dir)   # borrow config
        config.num_labels=opt.num_labels    # set label num
        if opt.task_type == 'token-classifier':
            model = transformers.LayoutLMv3ForTokenClassification.from_pretrained(opt.layoutlm_dir,config = config)
        elif opt.task_type == 'sequence-classifier':
            model = transformers.LayoutLMv3ForSequenceClassification.from_pretrained(opt.layoutlm_dir,config = config)
        elif opt.task_type == 'docvqa':
            config.num_labels = 2   # change back to 2
            model = transformers.LayoutLMv3ForQuestionAnswering.from_pretrained(opt.layoutlm_dir, config=config)
        elif opt.task_type == 'docbqa':
            config.num_labels = 2   # change back to 2
            model = layoutlmv3.LayoutLMv3ForBinaryQA.from_pretrained(opt.layoutlm_dir, config=config)
    elif opt.network_type == 'bert':
        if opt.task_type == 'token-classifier':
            model = BertTokenClassifier(opt)
        elif opt.task_type == 'sequence-classifier':
            config = AutoConfig.from_pretrained(opt.bert_dir)
            config.num_labels = opt.num_labels
            model = transformers.BertForSequenceClassification.from_pretrained(opt.bert_dir, config=config)
        elif opt.task_type == 'docvqa':
            model = BertForQA(opt)
        elif opt.task_type == 'docbqa':
            model = BertForBinaryQA(opt)
    elif opt.network_type == 'roberta':
        config = AutoConfig.from_pretrained(opt.roberta_dir)
        if opt.task_type == 'token-classifier': 
            model = RobertaTokenClassifier(opt)
        elif opt.task_type == 'sequence-classifier':
            config.num_labels = opt.num_labels
            model = transformers.RobertaForSequenceClassification.from_pretrained(opt.roberta_dir, config=config)
        elif opt.task_type == 'docvqa':
            model = RobertaForQA(opt)
            # model = transformers.RobertaForQuestionAnswering.from_pretrained(opt.roberta_dir, config=config)
        elif opt.task_type == 'docbqa':
            RobertaForBinaryQA(opt)
    elif opt.network_type == 'layoutlmv3_disent':
        if opt.task_type in ['mlm','blm']:
            # from_pretrained is put inside or outside
            if 'checkpoint_path' in opt.__dict__.keys():
                print('== load from the checkpoint === ', opt.checkpoint_path)
                config = AutoConfig.from_pretrained(opt.checkpoint_path)   # borrow config
                # config.spatial_attention_update = opt.spatial_attention_update
                model = LayoutLMv3ForMaskedLM.from_pretrained(opt.checkpoint_path, config = config)
            else:
                # the first time, we first start from layoutlm; put layoutlm_dir
                print('=== load the first time from layoutlmv3 ===')
                config = AutoConfig.from_pretrained(opt.layoutlm_dir)   # borrow config
                config.spatial_attention_update = opt.spatial_attention_update
                model = LayoutLMv3ForMaskedLM(config=config, start_dir_path=opt.layoutlm_dir)
        elif opt.task_type == 'token-classifier':
            config = AutoConfig.from_pretrained(opt.checkpoint_path)
            config.num_labels=opt.num_labels    # set label num from mydataset
            model = DiscentTokClassifier.from_pretrained(opt.checkpoint_path, config = config)
        elif opt.task_type == 'sequence-classifier':
            config = AutoConfig.from_pretrained(opt.checkpoint_path)
            config.num_labels=opt.num_labels    # set label num from mydataset
            model = layoutlmv3_disent.LayoutLMv3ForSequenceClassification.from_pretrained(opt.checkpoint_path,config = config)
        elif opt.task_type == 'docbqa':
            config = AutoConfig.from_pretrained(opt.checkpoint_path)
            config.num_labels=opt.num_labels    # set label num from mydataset
            model = layoutlmv3_disent.DesentForBinaryQA.from_pretrained(opt.checkpoint_path,config=config)
        elif opt.task_type == 'docvqa':
            config = AutoConfig.from_pretrained(opt.checkpoint_path)
            config.num_labels=opt.num_labels    # set label num from mydataset
            model = layoutlmv3_disent.LayoutLMv3ForQuestionAnswering.from_pretrained(opt.checkpoint_path,config=config)
        print('attention mode:',config.spatial_attention_update)
    elif opt.network_type == 'spatial_lm':
        if opt.task_type in ['mlm','blm']:
            # from_pretrained is put inside or outside
            if 'checkpoint_path' in opt.__dict__.keys():
                print('== load from the checkpoint === ', opt.checkpoint_path)
                config = SpatialLMConfig.from_pretrained(opt.checkpoint_path)   # borrow config
                model = SpatialLMForMaskedLM.from_pretrained(opt.checkpoint_path, config = config)
            else:
                # the first time, we first start from layoutlm; put layoutlm_dir
                print('=== load the first time from layoutlmv3 ===')
                config = SpatialLMConfig.from_pretrained(opt.layoutlm_dir)   # borrow config
                model = SpatialLMForMaskedLM(config=config, start_dir_path=opt.layoutlm_dir)

        elif opt.task_type == 'token-classifier':
            config = SpatialLMConfig.from_pretrained(opt.checkpoint_path)
            config.num_labels=opt.num_labels    # set label num
            config.spatial_attention = opt.spatial_attention
            model = SpatialLMForTokenclassifier.from_pretrained(opt.checkpoint_path, config = config)
        elif opt.task_type == 'sequence-classifier':
            config = SpatialLMConfig.from_pretrained(opt.checkpoint_path)
            config.spatial_attention = opt.spatial_attention
            if not bool(opt.inference_only): # if it is train mode
                config.num_labels, config.id2label, config.label2id = opt.num_labels, opt.id2label, opt.label2id  # set label num
            model = SpatialLMForSequenceClassification.from_pretrained(opt.checkpoint_path, config=config)
        elif opt.task_type == 'docvqa':
            config = SpatialLMConfig.from_pretrained(opt.checkpoint_path)
            config.spatial_attention = opt.spatial_attention
            model = SpatialLMForDocVQA.from_pretrained(opt.checkpoint_path, config = config)
    else:
        raise Exception('model not supported:{}'.format(opt.network_type))

    return model


