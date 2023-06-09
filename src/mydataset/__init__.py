
def setup(opt):
    if opt.dataset_name.lower() == 'rvlcdip':
        from mydataset.rvlcdip import RVLCDIP as MyData
    elif opt.dataset_name.lower() == 'funsd_cord_sorie':
        from mydataset.funsd_cord_sorie import FUNSD_CORD_SORIE as MyData
    elif opt.dataset_name.lower() == 'funsd':
        from mydataset.funsd4lm import FUNSD as MyData
    elif opt.dataset_name.lower() == 'funsdplus':
        from mydataset.funsdplus4lm import FUNSDPLUS as MyData
    elif opt.dataset_name.lower() == 'cord':
        from mydataset.cord4lm import CORD as MyData
    elif opt.dataset_name.lower() == 'rvl':
        from mydataset.rvl import RVL as MyData
    elif opt.dataset_name.lower() == 'sroie':
        from mydataset.sroie import SROIE as MyData
    elif opt.dataset_name.lower() == 'cdip':
        from mydataset.cdip import CDIP as MyData 
    elif opt.dataset_name.lower() == 'docvqa_ocr':
        from mydataset.docvqa_ocr import DocVQA as MyData
    elif opt.dataset_name.lower() == 'findoc_ner':
        from mydataset.findoc_ner import FinDoc as MyData
    elif opt.dataset_name.lower() == 'findoc_cat':
        from mydataset.findoc_cat import FinDoc as MyData
    elif opt.dataset_name.lower() == 'findoc_vqa':
        from mydataset.findoc_vqa import FinDoc as MyData
    elif opt.dataset_name.lower() == 'findoc_bqa':
        from mydataset.findoc_bqa import FinDoc as MyData
    else:
        raise Exception('dataset not supported:{}'.format(opt.dataset_name))

    mydataset = MyData(opt=opt)
    return mydataset
