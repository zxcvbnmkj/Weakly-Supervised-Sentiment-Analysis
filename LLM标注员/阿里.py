from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_aoe = pipeline(Tasks.siamese_uie, 'damo/nlp_structbert_siamese-aoe_chinese-base', model_revision='v1.0')

semantic_aoe(
	input='柚子味道很好，很香，果肉饱满，水分足，没有坏果，商家发货速度很快，包装严实完整，就是这边送货态度恶劣，商家也负责任处理解决',
  	schema={
        '属性词': {
            "正向情感(情感词)": None,
            "负向情感(情感词)": None,
            "中性情感(情感词)": None
        }
    }
)