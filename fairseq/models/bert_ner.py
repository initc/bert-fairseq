from fairseq.modules.bert_modeling import PreTrainedBertModel, BertModel, BertEmbeddings, BertPreTrainedModel, BertLayerNorm
import torch.nn as nn
import  torch.nn.functional  as F
import torch
from . import BaseFairseqModel, register_model, register_model_architecture



@register_model('bert_NER_model')
class BertNERModel(BaseFairseqModel):


    def __init__(self, model, target_dictionary):
        super().__init__()
        self.model = model
        self.target_dictionary = target_dictionary

    def forward(self, input_ids, token_type_ids, attention_mask):
        cls_score = self.model(input_ids, token_type_ids, attention_mask)
        return cls_score

    def get_normalized_probs(self, net_output, log_probs=False):
        if log_probs:
            return F.log_softmax(net_output)
        else:
            return F.softmax(net_output)

    def info_acc(self, predict, target):
        return self.target_dictionary.info_acc(predict, target)



    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--pre-dir', type=str, help="where to load bert model")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        if not hasattr(args, "pre_dir"):
            args.pre_dir = args.tokenizer_dir
        model = NERModel.build_model(args.pre_dir, tgt_dict=task.target_dictionary)
        # fix_layers(model.bert, list(range(0,20))+["embeddings"])


        return BertNERModel(model, task.target_dictionary)

class NERModel(PreTrainedBertModel):

    def __init__(self, config, *inputs, **kwargs):
        super(NERModel, self).__init__(config)

        self.tgt_dict = kwargs["tgt_dict"]
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, len(self.tgt_dict))
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids, attention_mask):

        encoder_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        encoder_output = self.dropout(encoder_output)
        logits = self.classifier(encoder_output[:,1:])


        return logits

def fix_layers(bert_model, layers):
    for name, params in bert_model.named_parameters():
        if any([True if str(l) in name else False for l in layers]):
            params.require_grad = False
            # print("| name fix : {}".format(name))



@register_model_architecture('bert_NER_model', 'bert_NER_model')
def caiyun_base_architecture(args):
    args.AB_times = getattr(args, 'AB_times', 10)
    














