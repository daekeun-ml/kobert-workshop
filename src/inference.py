import os
import json
import glob
import time

# Install/Update Packages
subprocess.call([sys.executable, '-m', 'pip', 'install', 'gluonnlp', 'torch', 'sentencepiece', 
                 'onnxruntime', 'transformers', 'git+https://git@github.com/SKTBrain/KoBERT.git@master'])

import mxnet as mx
import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer
from gluonnlp.model import BERTModel, BERTEncoder
from kobert.utils import get_tokenizer


def initialize_model(vocab_file, use_pooler, use_decoder, use_classifier, ctx=mx.cpu(0)):
    
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                         padding_token='[PAD]')    
    predefined_args = {
        'num_layers': 12,
        'units': 768,
        'hidden_size': 3072,
        'max_length': 512,
        'num_heads': 12,
        'dropout': 0.1,
        'embed_size': 768,
        'token_type_vocab_size': 2,
        'word_embed': None,
    }
    
    encoder = BERTEncoder(num_layers=predefined_args['num_layers'],
                          units=predefined_args['units'],
                          hidden_size=predefined_args['hidden_size'],
                          max_length=predefined_args['max_length'],
                          num_heads=predefined_args['num_heads'],
                          dropout=predefined_args['dropout'],
                          output_attention=False,
                          output_all_encodings=False)
    # BERT
    net = BERTModel(
        encoder,
        len(vocab_b_obj.idx_to_token),
        token_type_vocab_size=predefined_args['token_type_vocab_size'],
        units=predefined_args['units'],
        embed_size=predefined_args['embed_size'],
        word_embed=predefined_args['word_embed'],
        use_pooler=use_pooler,
        use_decoder=use_decoder,
        use_classifier=use_classifier)  
    
    
    net.initialize(ctx=ctx)
    return vocab_b_obj, net

def get_kobert_pretrained_model(model_file,
                     vocab_file,
                     use_pooler=True,
                     use_decoder=False,
                     use_classifier=False,
                     num_classes=2,
                     ctx=mx.cpu(0)):
    vocab_b_obj, net = initialize_model(vocab_file, use_pooler, use_decoder, use_classifier, ctx)
    
    # Load fine-tuning model
    classifier = nlp.model.BERTClassifier(net, num_classes=num_classes, dropout=0.5)
    classifier.classifier.initialize(ctx=ctx)
    classifier.hybridize(static_alloc=True)
    classifier.load_parameters(model_file)

    return (classifier, vocab_b_obj)

def model_fn(model_dir):
    voc_file_name = glob.glob('{}/*.spiece'.format(model_dir))[0]
    model_param_file_name = glob.glob('{}/*.params'.format(model_dir))[0]
    
    # check if GPU is available
    if mx.context.num_gpus() > 0:
        print('Use GPU')
        ctx = mx.gpu()
    else:
        print('Use CPU')
    
    model, vocab = get_kobert_pretrained_model(model_param_file_name, voc_file_name, ctx=ctx)
    tok = SentencepieceTokenizer(voc_file_name)

    return model, vocab, tok, ctx

def transform_fn(model, request_body, content_type, accept_type):
    model, vocab, tok, ctx = model
    
    sent = request_body.encode('utf-8')
    sent = sent.decode('unicode_escape')[1:]
    sent = sent[:-1]
    toked = tok(sent)
    
    t0 = time.time()
    input_ids = mx.nd.array([vocab[vocab.bos_token]]  + vocab[toked]).expand_dims(axis=0)
    token_type_ids = mx.nd.zeros(input_ids.shape)
    input_ids = input_ids.as_in_context(ctx)
    token_type_ids = token_type_ids.as_in_context(ctx)
    pred = mx.nd.softmax(model(input_ids, token_type_ids)[0])
    
    response_body = json.dumps({
        'score': pred.asnumpy().tolist(),
        'time': time.time() - t0
    })    

    return response_body, content_type