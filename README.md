# Fine Tuning Naver Movie Review Sentiment Classification with KoBERT using GluonNLP

In this workshop, you will learn how to fine-tune pre-trained KoBERT models on Amazon SageMaker and how to host trained models in the SageMaker Endpoint.

The workshop consists of the following three modules: Module 1 and Module 2 deal with fine-tuning. For reference, you do not need to do both Module 1 and Module 2, you only need to perform one selectively. However, in order to proceed with module 3, either module 1 or module 2 must be performed.


**[1. Fine-tuning Naver Movie Review Sentiment Classification with KoBERT](module1_kobert_nsmc_finetuning.ipynb)**: In this module, we focus on fine-tuning with the pre-trained BERT model to classify Naver movie review sentiment. Since this module does not use the SageMaker API, you need to have at least GPU as well as CUDA 10.1/10.2. After this module, you will get familiar with BERT fine-tuning.

**[2. Fine-tuning Naver Movie Review Sentiment Classification with KoBERT on Amazon SageMaker](module2_kobert_nsmc_finetuning_sagemaker.ipynb)**: This module covers how to learn with SageMaker but excluded the detailed explanation. After this module, you will familiar with BERT fine-tuning on SageMaker.

**[3. Deploying fine-tuned model to SageMaker Endpoint to perform Inference](module3_kobert_nsmc_deployment.ipynb)**: In this module, you will learn how to deploy a fine-tuned kobert model to the SageMaker endpoint. A great tutorial has already been introduced in the AWS Korea AIML blog and GitHub by Amazon Machine Learning Solutions Lab(https://github.com/aws-samples/kogpt2-sagemaker/blob/master/sagemaker-deploy-en.md). Based on this method, it is easy to perform endpoint deployment by making minor modifications.

