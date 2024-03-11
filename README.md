# Sentiment Analysis Using Transform

[BERT base model (uncased)](https://huggingface.co/google-bert/bert-base-uncased)

Pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in this paper and first released in this repository. This model is uncased: it does not make a difference between english and English.

[IMDB Dataset](https://huggingface.co/datasets/imdb)

Large Movie Review Dataset. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well.

## Quick start

### Train

```shell
python train.py 
```


### Predict

```shell
python predict.py 
```

### Serve

```shell
python server.py 
```

```text 
This film is terrible! predicted result: negative, predicted probability: 0.994263231754303
This film is great! predicted result: positive, predicted probability: 0.9927541613578796
This film is not terrible! predicted result: positive, predicted probability: 0.7570403218269348
This film is not great! predicted result: negative, predicted probability: 0.9897279143333435
```

## Reference

- https://github.com/bentrevett/pytorch-sentiment-analysis/tree/main
