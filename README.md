# Fine Tuning Naver Movie Review Sentiment Classification with KoBERT using GluonNLP

***[Before getting started] This workshop assumes you have basic knowledge of Deep Learning and experience with Amazon SageMaker.***

<br>

In this workshop, you will learn how to fine-tune pre-trained KoBERT models on Amazon SageMaker and how to host trained models in the SageMaker Endpoint.

The workshop consists of the following three modules: Module 1 and Module 2 deal with fine-tuning. For reference, you do not need to do both Module 1 and Module 2, you only need to perform one selectively. However, in order to proceed with module 3, either module 1 or module 2 must be performed.


**[1. Fine-tuning Naver Movie Review Sentiment Classification with KoBERT](module1_kobert_nsmc_finetuning.ipynb)**: In this module, we focus on fine-tuning with the pre-trained BERT model to classify Naver movie review sentiment. Since this module does not use the SageMaker API, you need to have at least GPU as well as CUDA toolkit >= 10.1. After this module, you will get familiar with BERT fine-tuning.

*[Note] 
The CUDA toolkit version of SageMaker as of May 2020 is 10.0, so CUDA upgrade is required. No CUDA upgrade is required for SageMaker Studio, since CUDA version is 10.1 as of May 2020. If you want to run the notebook on SageMaker, please run the shell script below in terminal mode.*
```shell
$ wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run 
$ sudo sh cuda_10.2.89_440.33.01_linux.run
```

**[2. Fine-tuning Naver Movie Review Sentiment Classification with KoBERT on Amazon SageMaker](module2_kobert_nsmc_finetuning_sagemaker.ipynb)**: This module covers how to learn with SageMaker but excluded the detailed explanation (See module 1 for details). After this module, you will familiar with BERT fine-tuning on SageMaker.

**[3. Deploying fine-tuned model to SageMaker Endpoint to perform Inference](module3.1_kobert_nsmc_deployment_local.ipynb)**: In this module, you will learn how to deploy a fine-tuned kobert model to the SageMaker endpoint. 
We recommend that you test your deployment code in local mode first, without provisioning for deployment. If you have done enough testing, you can deploy EC2 instances for deployment. You can also Build Your Own Container(BYOC) if needed, and a great tutorial on this can be found on the [AWS Korea AIML blog](https://aws.amazon.com/ko/blogs/korea/deploy-kogpt2-model-mxnet-amazon-sagemaker/) and [GitHub](https://github.com/aws-samples/kogpt2-sagemaker/blob/master/sagemaker-deploy-en.md). Based on this tutorial, you can easily deploy Endpoint on SageMaker with minor modifications.

- [Local Mode](module3.1_kobert_nsmc_deployment_local.ipynb)
- [Script Mode](module3.2_kobert_nsmc_deployment_script.ipynb)
- [Bring Your Own Container(BYOC)](module3.3_kobert_nsmc_deployment_byoc.ipynb)

<br>

---

<br>

***[시작하기 전] 이 워크샵은 딥러닝에 대한 기본적인 지식이 있고 Amazon SageMaker를 사용해 본 경험이 있다고 가정합니다. 딥러닝에 대한 지식이 있지만 Amazon SageMaker 사용 경험이 없다면, 아래의 동영상들을 참조해 주세요.***
- [Amazon SageMaker 오버뷰](https://www.youtube.com/watch?v=jF2BN98KBlg)
- [Amazon SageMaker 데모](https://www.youtube.com/watch?v=miIVGlq6OUk)

<br>

이 워크샵에서 여려분은 Amazon SageMaker에서 사전 훈련된 KoBERT 모델을 fine-tuning하는 방법과 훈련된 모델을 SageMaker Endpoint에 호스팅하는 방법을 배우게 됩니다.

워크샵은 다음 세 가지 모듈로 구성됩니다. 모듈 1 및 모듈 2는 fine-tuning을 수행합니다. 참고로 모듈 1과 모듈 2를 모두 수행할 필요는 없으며, 상황에 따라 1개 모듈만 선택적으로 수행할 수 있습니다. 그러나 모듈 3을 진행하려면, 모듈 1 또는 모듈 2 중 하나를 반드시 수행해야 합니다.

**[1. KoBERT를 이용한 네이버 영화 리뷰 감성 분류 fine-tuning](module1_kobert_nsmc_finetuning.ipynb)**: 이 모듈에서는 미리 훈련된 BERT 모델을 사용하여 네이버 영화 리뷰 감성을 분류하는 데 초점을 맞춥니다. 이 모듈은 SageMaker API를 사용하지 않으므로 GPU와 CUDA 10.1 이상의 toolkit이 필요합니다. 이 모듈을 수행하신 후에 BERT의 fine-tuning 방법에 친숙해질 수 있습니다.

*[Note] 2020년 5월 시점에서의 SageMaker 노트북 인스턴스 CUDA toolkit 버전은 10.0이므로 CUDA 업그레이드가 필요합니다. 단, SageMaker Studio는 이미 CUDA버전이 10.1이기 때문에 CUDA 업그레이드가 필요 없습니다. 본 노트북을 SageMaker 상에서 실행하고 싶다면, 터미널 모드에서 아래 셀 스크립트를 실행해 주세요.*
```shell
$ wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run 
$ sudo sh cuda_10.2.89_440.33.01_linux.run
```

**[2. Amazon SageMaker에서 KoBERT를 이용한 네이버 영화 리뷰 감성 분류 fine-tuning](module2_kobert_nsmc_finetuning_sagemaker.ipynb)**: 이 모듈은 SageMaker를 사용하여 학습을 수행하지만 자세한 설명은 생략하였습니다. (자세한 설명은 모듈 1을 참조하세요.) 이 모듈을 수행하신 후에 SageMaker 상에서의 BERT fine-tuning에 친숙해질 수 있습니다.

**[3. 추론을 수행하기 위해 fine-tuning된 모델을 SageMaker Endpoint에 배포](module3_kobert_nsmc_deployment.ipynb)**: 이 모듈에서는 fine-tuning된 KoBERT 모델을 SageMaker Endpoint에 배포하는 방법을 배웁니다. 먼저 배포용 프로비저닝 없이 로컬 모드에서 먼저 배포 코드를 테스트해 보고, 충분한 테스트를 거쳤다면 배포용 EC2 인스턴스를 프로비저닝하는 방법을 추천합니다. 또한 필요 시,여러분의 도커 컨테이너를 직접 빌드할 수 있으며 이에 대한 훌륭한 튜토리얼이 [AWS Korea AIML 블로그](https://aws.amazon.com/ko/blogs/korea/deploy-kogpt2-model-mxnet-amazon-sagemaker/) 및 [GitHub](https://github.com/aws-samples/kogpt2-sagemaker/blob/master/sagemaker-deploy-en.md) 에 소개되었습니다. 이 튜토리얼을 기반으로 약간만 수정하면 여러분은 SageMaker 상에서 Endpoint 배포를 쉽게 수행할 수 있습니다.

- [Local Mode](module3.1_kobert_nsmc_deployment_local.ipynb)
- [Script Mode](module3.2_kobert_nsmc_deployment_script.ipynb)
- [Bring Your Own Container(BYOC)](module3.3_kobert_nsmc_deployment_byoc.ipynb)



<br>

[Privacy](https://aws.amazon.com/privacy/) | [Site terms](https://aws.amazon.com/terms/) | © 2020, Amazon Web Services, Inc. or its affiliates. All rights reserved.