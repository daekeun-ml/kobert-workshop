import argparse
import logging
import io, os
import random
import time
import datetime 
import itertools
import numpy as np
import mxnet as mx
from mxnet import nd
import subprocess
import sys

# # Install/Update Packages
# subprocess.call([sys.executable, '-m', 'pip', 'install', 'gluonnlp', 'torch', 'sentencepiece', 
#                  'onnxruntime', 'transformers', 'git+https://git@github.com/SKTBrain/KoBERT.git@master'])


from mxnet.gluon import nn, rnn
from mxnet import gluon, autograd
import gluonnlp as nlp

#from kobert.mxnet_kobert import get_mxnet_kobert_model
from model import get_mxnet_kobert_model

from kobert.utils import get_tokenizer
from bert import BERTDatasetTransform, BERTDataset, BERTClassifier

import warnings
warnings.filterwarnings('ignore')

def get_dataloaders(dataset_train, dataset_test, vocab, batch_size, 
                    max_len=64, class_labels=['0', '1']):
    from kobert.mxnet_kobert import get_mxnet_kobert_model
    from kobert.utils import get_tokenizer
    
    tokenizer = get_tokenizer()
    bert_tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    # for single sentence classification, set pair=False
    # for regression task, set class_labels=None
    # for inference without label available, set has_label=False
    transform = BERTDatasetTransform(bert_tokenizer, max_len,
                                                    class_labels=class_labels,
                                                    has_label=True,
                                                    pad=True,
                                                    pair=False)
    data_train = dataset_train.transform(transform)
    data_test = dataset_test.transform(transform)    
    

    train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_train],
                                                batch_size=batch_size,
                                                shuffle=True)
    train_dataloader = gluon.data.DataLoader(data_train, batch_sampler=train_sampler)

    test_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_test],
                                                batch_size=batch_size,
                                                shuffle=True)
    test_dataloader = mx.gluon.data.DataLoader(data_test, batch_sampler=test_sampler)
    
    return train_dataloader, test_dataloader, bert_tokenizer


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def evaluate_accuracy(model, data_iter, loss_function, ctx):
    acc = mx.metric.Accuracy()
    total_loss = 0

    for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(data_iter):

        token_ids = token_ids.as_in_context(ctx)
        valid_length = valid_length.as_in_context(ctx)
        segment_ids = segment_ids.as_in_context(ctx)
        label = label.as_in_context(ctx)

        out = model(token_ids, segment_ids, valid_length.astype('float32'))
        ls = loss_function(out, label).mean()
        total_loss += ls.asscalar()

        acc.update(preds=out, labels=label)

    avg_acc = acc.get()[1]
    avg_loss = total_loss / batch_id

    print('Validation loss={:.4f}, acc={:.3f}'.format(avg_loss, avg_acc))

    return avg_acc, avg_loss    


def train_model(num_epochs, batch_size, lr, log_interval, train_dir, validation_dir, model_output_dir):

    ctx = mx.gpu()
      
    print("=== Load pre-trained KoBERT model ===")
    
    # Load pre-trained KoBERT model
    bert_base, vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False, ctx=ctx)

    # Create KoBERT classifition model for fine-tuning
    bert_classifier = nlp.model.BERTClassifier(bert_base, num_classes=2, dropout=0.5)

    # Only need to initialize the classifier layer.
    bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    bert_classifier.hybridize(static_alloc=True)

    # softmax cross entropy loss for classification
    loss_function = gluon.loss.SoftmaxCELoss()
    loss_function.hybridize(static_alloc=True)
    metric = mx.metric.Accuracy()
    
    print("=== Getting Data ===")
 
    train_file = os.path.join(train_dir, 'train.txt')
    validation_file = os.path.join(validation_dir, 'validation.txt')
    print(train_file)
    print(validation_file)
    dataset_train = nlp.data.TSVDataset(train_file, field_indices=[1,2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset(validation_file, field_indices=[1,2], num_discard_samples=1)

    train_dataloader, test_dataloader, bert_tokenizer = get_dataloaders(dataset_train, dataset_test, vocab, batch_size)

    num_train_examples = len(dataset_train)
    step_num = 0
    all_model_params = bert_classifier.collect_params()

    trainer = gluon.Trainer(all_model_params, 'adam',
                               {'learning_rate': lr, 'epsilon': 1e-9})

    
    # Weight Decay is not applied to LayerNorm and Bias.
    for _, v in bert_classifier.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    # Collect all differentiable parameters
    # `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
    # The gradients for these params are clipped later
    params = [p for p in bert_classifier.collect_params().values() 
              if p.grad_req != 'null']


    ### 
    print("=== Start Training ===")

    training_stats = []
    step_num = 0

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # Start training loop
    for epoch_id in range(num_epochs):

        # === Training phase ===

        # Measure how long the training epoch takes.
        t0 = time.time()    

        metric.reset()
        step_loss = 0
        total_loss = 0
        
        for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(train_dataloader):

            with mx.autograd.record():

                # Load the data to the GPU.
                token_ids = token_ids.as_in_context(ctx)
                valid_length = valid_length.as_in_context(ctx)
                segment_ids = segment_ids.as_in_context(ctx)
                label = label.as_in_context(ctx)

                # Forward computation
                out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))

                # Compute loss
                ls = loss_function(out, label).mean()

            # Perform a backward pass to calculate the gradients
            ls.backward()
            
            # Gradient clipping
            # step() can be used for normal parameter updates, but if we apply gradient clipping, 
            # you need to manaully call allreduce_grads() and update() separately.            
            trainer.allreduce_grads()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer.update(1)

            step_loss += ls.asscalar()
            total_loss += ls.asscalar()
            metric.update([label], [out])

            # Printing vital information
            if (batch_id + 1) % (log_interval) == 0:
                print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                             .format(epoch_id, batch_id + 1, len(train_dataloader),
                                     step_loss / log_interval,
                                     trainer.learning_rate, metric.get()[1]))
                step_loss = 0

            train_avg_acc = metric.get()[1]
            train_avg_loss = total_loss / batch_id
            total_loss = 0

        # Measure how long this epoch took.
        train_time = format_time(time.time() - t0)

        # === Validation phase ===
        print('=== Start Validation ===')

        # Measure how long the validation epoch takes.    
        t0 = time.time()    
        valid_avg_acc, valid_avg_loss = evaluate_accuracy(bert_classifier, test_dataloader, loss_function, ctx)

        # Measure how long this epoch took.
        valid_time = format_time(time.time() - t0)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        # Record all statistics from this epoch.
        print('Epoch {} evaluation result: train_acc {}, train_loss {}, valid_acc {}, valid_loss {}'
                     .format(epoch_id, train_avg_acc, train_avg_loss, valid_avg_acc, valid_avg_loss))
                     
        training_stats.append(
            {
                'epoch': epoch_id + 1,
                'train_acc': train_avg_acc,
                'train_loss': train_avg_loss,
                'train_time': train_time,
                'valid_acc': valid_avg_acc,
                'valid_loss': valid_avg_loss,
                'valid_time': valid_time
            }
        )

        # === Save Model Parameters ===
        subprocess.call('cp ~/kobert/kobert_news_wiki_ko_cased-1087f8699e.spiece {}'.format(model_output_dir), shell=True)     
        bert_classifier.save_parameters(os.path.join(model_output_dir, 'kobert-nsmc.params'))
        print("Model successfully saved at: {}".format(model_output_dir))        
        
        
if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)  
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--log_interval', type=int, default=50) 
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))      
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))      
    parser.add_argument('--model_output_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    #parse arguments
    args = parser.parse_args() 
    train_model(num_epochs=args.num_epochs, 
                batch_size=args.batch_size, 
                lr=args.lr, 
                log_interval=args.log_interval, 
                train_dir=args.train,
                validation_dir=args.validation,
                model_output_dir=args.model_output_dir)