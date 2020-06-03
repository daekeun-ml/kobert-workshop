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

import logging
import logging.handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)

def get_dataloaders(dataset_train, dataset_test, vocab, batch_size, 
                    max_len=64, class_labels=['0', '1']):

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

        # Load the data to the GPUs
        token_ids_ = gluon.utils.split_and_load(token_ids, ctx, even_split=False)
        valid_length_ = gluon.utils.split_and_load(valid_length, ctx, even_split=False)
        segment_ids_ = gluon.utils.split_and_load(segment_ids, ctx, even_split=False)
        label_ = gluon.utils.split_and_load(label, ctx, even_split=False)

        for t, v, s, l in zip(token_ids_, valid_length_, segment_ids_, label_):
            # Forward computation
            out = model(t, s, v.astype('float32'))
            ls = loss_function(out, l).mean()
            total_loss += ls.asscalar()
            acc.update(preds=out, labels=l)

    avg_acc = acc.get()[1]
    avg_loss = total_loss / batch_id

    logger.info('Validation loss={:.4f}, acc={:.3f}'.format(avg_loss, avg_acc))

    return avg_acc, avg_loss    


def train_model(num_epochs, batch_size, lr, log_interval, train_dir, validation_dir, model_output_dir):

    num_gpus = mx.context.num_gpus()
    num_gpus = 1
    
    batch_size *= num_gpus
    ctx = [mx.gpu(i) for i in range(num_gpus)]
      
    logger.info("=== Load pre-trained KoBERT model ===")
    
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
    
    logger.info("=== Getting Data ===")
   
    train_file = os.path.join(train_dir, 'train.txt')
    validation_file = os.path.join(validation_dir, 'validation.txt')

    dataset_train = nlp.data.TSVDataset(train_file, field_indices=[1,2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset(validation_file, field_indices=[1,2], num_discard_samples=1)

    train_dataloader, test_dataloader, bert_tokenizer = get_dataloaders(dataset_train, dataset_test, vocab, batch_size)

    all_model_params = bert_classifier.collect_params()
    trainer = gluon.Trainer(all_model_params, 'bertadam',
                            {'learning_rate': lr, 'epsilon': 1e-9, 'wd':0.01, 'clip_gradient': 1},
                           kvstore='device')

    # Weight Decay is not applied to LayerNorm and Bias.
    for _, v in bert_classifier.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    # Collect all differentiable parameters
    # `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
    # The gradients for these params are clipped later
    params = [p for p in all_model_params.values() if p.grad_req != 'null']  

    # Learning rate warmup parameters
    num_train_examples = len(dataset_train)
    num_train_steps = int(num_train_examples / batch_size * num_epochs) 
    warmup_ratio = 0.1
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    logger.info("Training steps={}, warmup_steps={}".format(num_train_steps, num_warmup_steps))

    logger.info("=== Start Training ===")
    training_stats = []
    step_num = 0
    set_seed()  

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
            
            # Learning rate warmup
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                non_warmup_steps = step_num - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                new_lr = lr - offset * lr
            trainer.set_learning_rate(new_lr)
#             new_lr = scheduler(step_num)
#             trainer.set_learning_rate(new_lr)
    
            losses = []
            with mx.autograd.record():
                # Load the data to the GPUs.
                token_ids_ = gluon.utils.split_and_load(token_ids, ctx, even_split=False)
                valid_length_ = gluon.utils.split_and_load(valid_length, ctx, even_split=False)
                segment_ids_ = gluon.utils.split_and_load(segment_ids, ctx, even_split=False)
                label_ = gluon.utils.split_and_load(label, ctx, even_split=False)                

                for t, v, s, l in zip(token_ids_, valid_length_, segment_ids_, label_):
                    # Forward computation
                    out = bert_classifier(t, s, v.astype('float32'))
                    ls = loss_function(out, l)
                    losses.append(ls)
                    metric.update([l], [out])             

            # Perform a backward pass to calculate the gradients        
            for ls in losses:
                ls.backward()  

            trainer.step(batch_size)
            
            # Sum losses over all devices
            sum_loss = (sum([l.sum().asscalar() for l in losses]))/batch_size
            step_loss += sum_loss
            total_loss += sum_loss
            step_num += 1            
            #metric.update([label], [out])             

            # Printing vital information
            if (batch_id + 1) % (log_interval) == 0:
                logger.info('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                             .format(epoch_id, batch_id + 1, len(train_dataloader),
                                     step_loss / log_interval,
                                     trainer.learning_rate, metric.get()[1]))
                step_loss = 0

            train_avg_acc = metric.get()[1]
            train_avg_loss = total_loss
            total_loss = 0

        # Measure how long this epoch took.
        train_time = format_time(time.time() - t0)

        # === Validation phase ===
        logger.info('=== Start Validation ===')

        # Measure how long the validation epoch takes.    
        t0 = time.time()    
        valid_avg_acc, valid_avg_loss = evaluate_accuracy(bert_classifier, test_dataloader, loss_function, ctx)

        # Measure how long this epoch took.
        valid_time = format_time(time.time() - t0)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        # Record all statistics from this epoch.
        logger.info('Epoch {} evaluation result: train_acc {}, train_loss {}, valid_acc {}, valid_loss {}'
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
        logger.info("Model successfully saved at: {}".format(model_output_dir))        
        
        
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