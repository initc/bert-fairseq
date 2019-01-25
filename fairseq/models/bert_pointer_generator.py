from fairseq.modules.bert_modeling import PreTrainedBertModel, BertModel, BertEmbeddings, BertPreTrainedModel, BertLayerNorm
import torch.nn as nn
import  torch.nn.functional  as F
import torch
from . import BaseFairseqModel, register_model, register_model_architecture
from fairseq.modules.multi_pointer_modeling import TransformerDecoder, DecoderAttention, Feedforward


@register_model('bert_transformer')
class BertTransformerModel(BaseFairseqModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids, token_type_ids, attention_mask, prev_output_tokens):
        encoder_outs = self.encoder(input_ids, token_type_ids, attention_mask)
        deocder_out = self.decoder(prev_output_tokens, encoder_outs)
        return deocder_out



    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--pre-dir', type=str, help="where to load bert model")
        parser.add_argument('--reduce-dim', default=-1, type=int, metavar='N',
                            help='convert encoder hidden')
        parser.add_argument('--decoder-layer', default=2, type=int, metavar='N',
                            help='the number of decoder layer')
        parser.add_argument('--token-types', default=2, type=int, metavar='N',
                            help='the number of tokens number')
        parser.add_argument('--defined-position', action="store_true", default=False, 
                            help='user-defined position in embedding')
        parser.add_argument('--decoder-layers', type=int, metavar='N', default=2,
                            help='num decoder layers')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true', default=False,
                            help='share decoder input and output embeddings')
        parser.add_argument("--decoder-lr", default=3e-3, type=float, help="The initial learning rate for decoder layers")
        parser.add_argument("--decoder-lr-scale", default=100, type=float, help="The scale learning rate for decoder layers")
        parser.add_argument("--encoder-lr-scale", default=1, type=float, help="The initial learning rate for encoder layers")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        if not hasattr(args, "pre_dir"):
            args.pre_dir = args.tokenizer_dir

        encoder = BertTransformerEncoder.build_model(args.pre_dir, args=args)
        decoder_embedding = build_decoder_embedding(encoder)
        init_encoder_token_type(encoder, token_nums=args.token_types)
        decoder_dictionary = task.tokenizer
        decoder = BertTransformerDecoder(args, encoder.config, decoder_dictionary, decoder_embedding)
        return BertTransformerModel(encoder, decoder)

class BertTransformerEncoder(PreTrainedBertModel):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config)
        self.config = config
        args = kwargs["args"]
        self.is_defined_position = args.defined_position
        self.max_source_positions = args.max_source_positions
        self.bert = BertPreTrainedModel(config)

        decoder_dim = config.hidden_size
        self.reduce_dim = args.reduce_dim
        if self.reduce_dim > 0:
            decoder_dim = self.reduce_dim
            # self.linear_answer = nn.Linear(config.hidden_size, decoder_dim)
            self.linear_context = nn.Linear(config.hidden_size, decoder_dim)
            # self.ln_answer = LayerNorm(decoder_dim)
            self.ln_context = nn.LayerNorm(decoder_dim)
        args.decoder_dim = decoder_dim
        args.hidden_size = config.hidden_size

        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids, attention_mask, position_ids):
        if not self.is_defined_position:
            position_ids = None
        all_encoder_layers= self.bert(input_ids, token_type_ids, attention_mask)

        sequence_output = all_encoder_layers[-1]
        sequence_output = self.ln_context(self.linear_context(sequence_output)) if self.reduce_dim>0 else sequence_output
        return {"encoder_output":sequence_output, "encoder_mask":attention_mask, "input_ids":input_ids}

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return min(self.max_source_positions, self.config.max_position_embeddings)

class BertTransformerDecoder(nn.Module):

    def __init__(self, args, config, dictionary, embedding_token):
        super().__init__()
        self.args = args
        self.config = config
        self.dict = dictionary
        self.embedding_token = embedding_token
        decoder_dim = args.decoder_dim
        hidden_size = args.hidden_size
        if args.reduce_dim > 0:
            self.linear_answer = nn.Linear(hidden_size, decoder_dim)
            self.ln_answer = nn.LayerNorm(decoder_dim)

        self.attention_layer = TransformerDecoder(decoder_dim, decoder_dim//64, decoder_dim*4, args.decoder_layers, 0.2)
        self.pointer_layer = PointerDecoder(decoder_dim, decoder_dim, dropout=0.2)

        self.vocab_size = len(dictionary)
        self.out = nn.Linear(decoder_dim, self.vocab_size)

        self.apply(self.init_bert_weights)

        if args.share_decoder_input_output_embed:
            self.project = nn.Linear(decoder_dim, hidden_size)
            self.out.weight = embedding_token.word_embeddings.weight

    def forward(self, prev_output_tokens, encoder_outs):
        encoder_out, encoder_mask, encoder_input_ids = encoder_outs["encoder_output"], encoder_outs["encoder_mask"], encoder_outs["input_ids"]

        answer_mask = prev_output_tokens.data != self.dict.pad()
        answer_padding = prev_output_tokens.data == self.dict.pad()
        encoder_out_padding = encoder_mask.data == self.dict.pad()

        # self.pointer_layer.applyMasks(encoder_out_padding)

        answer_embedding = self.embedding_token(prev_output_tokens)
        answer_embedding = self.ln_answer(self.linear_answer(answer_embedding)) if self.args.reduce_dim>0 else answer_embedding

        decoder_out = self.attention_layer(answer_embedding, encoder_out, context_padding=encoder_out_padding, answer_padding=answer_padding, positional_encodings=False)

        decoder_out = self.pointer_layer(decoder_out, encoder_out, atten_mask=encoder_out_padding)

        context_question_outputs, context_question_weight, vocab_pointer_switches = decoder_out

        probs = self.probs(self.out, context_question_outputs, vocab_pointer_switches, context_question_weight, encoder_input_ids)
        return probs
        # else:
        #     return None, self.greedy(encoder_out, encoder_input_ids, answer_ids = prev_output_tokens)


    def probs(self, generator, outputs, vocab_pointer_switches, context_question_weight, context_question_indices, oov_to_limited_idx=None):

        size = list(outputs.size())

        size[-1] = self.vocab_size
        if self.args.share_decoder_input_output_embed:
            outputs = self.project(outputs)
        # B * Ta * V
        scores = generator(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim()-1)
        # B * Ta * V
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = self.vocab_size # + len(oov_to_limited_idx) TODO: add the oov
        if self.vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = scaled_p_vocab.new_full(size, EPSILON)
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim()-1)

        # p_context_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim()-1, context_question_indices.unsqueeze(1).expand_as(context_question_weight),
            (1 - vocab_pointer_switches).expand_as(context_question_weight) * context_question_weight)

        return scaled_p_vocab

    def greedy_decoder(self, encoder_out, eos_idx, pad_idx, max_lens=15):
        encoder_out, encoder_mask, encoder_input_ids = encoder_outs["encoder_output"], encoder_outs["encoder_mask"], encoder_outs["input_ids"]
        encoder_out_padding = encoder_mask.data == pad_idx

        B, TC, C = encoder_out.size()
        T = max_lens # suppose the max length of answer is 15

        outs = encoder_out.new_full((B, T), pad_idx, dtype=torch.long)

        hiddens = [encoder_out.new_zeros((B, T, C)) for l in range(len(self.attention_layer.layers) + 1)]
        # hiddens[0] = hiddens[0] #+ positional_encodings_like(hiddens[0])
        eos_yet = encoder_out.new_zeros((B, )).byte()
        answer_scores = encoder_out.new_zeros((B, ))
        for t in range(T):
            if t == 0:
                embedding = self.embedding_token(encoder_out.new_full((B, 1), eos_idx, dtype=torch.long), position=t)
            else:
                embedding = self.embedding_token(outs[:, t - 1].unsqueeze(1), position=t)
            embedding = self.ln_answer(self.linear_answer(embedding)) if self.args.reduce_dim>0 else embedding
            
            hiddens[0][:, t] = hiddens[0][:, t] + embedding.squeeze(1)
            for l in range(len(self.attention_layer.layers)):
                hiddens[l + 1][:, t] = self.attention_layer.layers[l].feedforward(
                    self.attention_layer.layers[l].attention(
                    self.attention_layer.layers[l].selfattn(hiddens[l][:, t], hiddens[l][:, :t+1], hiddens[l][:, :t+1])
                  , encoder_out, encoder_out, padding=encoder_out_padding))
            
            decoder_outputs = self.pointer_layer(hiddens[-1][:, t].unsqueeze(1), encoder_out, atten_mask=encoder_out_padding)
            context_question_outputs, context_question_weight, vocab_pointer_switches = decoder_out

            probs = self.probs(self.out, context_question_outputs, vocab_pointer_switches, context_question_weight, encoder_input_ids)

            pred_probs, preds_index = probs.max(-1)
            preds_index = preds_index.squeeze(1)
            pred_probs = pred_probs.log().squeeze(1)
            eos_yet = eos_yet | (preds_index == eos_idx)  # the index of "[SEP]" is 102
            answer_scores += pred_probs * (1-eos_yet.float())
            outs[:, t] = preds_index.cpu()
            if eos_yet.all():
                break
        # if self.reshape:
        #     return outs.view((self.ori_size[0], self.ori_size[1], -1)), answer_scores.view((self.ori_size[0], self.ori_size[1], -1))
        return outs, answer_scores


    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output.float()
        return torch.log(logits)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return min(self.max_target_positions, self.config.max_position_embeddings)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class PointerDecoder(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.0, num_layers=1):
        super().__init__()
        self.d_hid = d_hid
        self.d_in = d_in
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.context_question_attn = DecoderAttention(d_hid, dot=True)  # TJ
        self.vocab_pointer_switch = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.d_hid + d_in, 1), nn.Sigmoid()) # XD

    # input = B * T * C
    def forward(self, input, context_question, atten_mask):

        context_question_outputs, context_question_weight = self.context_question_attn(input, context_question, atten_mask=atten_mask)
        vocab_pointer_switches = self.vocab_pointer_switch(torch.cat([input, context_question_outputs], -1))
        context_question_outputs = self.dropout(context_question_outputs)

        return context_question_outputs, context_question_weight, vocab_pointer_switches



def build_decoder_embedding(encoder):
    decoder_embedding = BertEmbeddings(encoder.config)
    encoder_embedding = encoder.bert.embeddings
    decoder_embedding.word_embeddings.weight.data.copy_(encoder_embedding.word_embeddings.weight.data)
    decoder_embedding.position_embeddings.weight.data.copy_(encoder_embedding.position_embeddings.weight.data)
    decoder_embedding.token_type_embeddings.weight.data.copy_(encoder_embedding.token_type_embeddings.weight.data)
    decoder_embedding.LayerNorm.gamma.data.copy_(encoder_embedding.LayerNorm.gamma.data)
    decoder_embedding.LayerNorm.beta.data.copy_(encoder_embedding.LayerNorm.beta.data)
    return decoder_embedding

def init_encoder_token_type(encoder_model, token_nums=3):
    config = encoder_model.config

    token_type_embeddings_appended = nn.Embedding(token_nums, config.hidden_size)
    token_type_embeddings_appended.weight.data.normal_(mean=0.0, std=config.initializer_range)
    # token_type_embeddings_appended.bias.data.zero_()

    encoder_token_type_embedding = encoder_model.bert.embeddings.token_type_embeddings

    token_type_embeddings_appended.weight.data[:2,:] = encoder_token_type_embedding.weight.data[:,:]
    encoder_token_type_embedding.weight = token_type_embeddings_appended.weight



    
@register_model_architecture('bert_transformer', 'bert_transformer_base')
def caiyun_base_architecture(args):
    # args.AB_times = getattr(args, 'AB_times', 10)
    pass
    

