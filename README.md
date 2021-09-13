# reuters-21578-text-categorization

The Reuters Corpus is a collection of documents with news articles which totals over ten thousand documents. The documents in the Reuters-21578 collection appeared on the Reuters newswire in 1987. It is widely used for text categorization research. The dataset can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection).

## Getting Started

Clone this git repository in [SageMaker Studio](https://aws.amazon.com/sagemaker/studio/). This allows us to run notebooks in SageMaker and use S3 for storage. By default, use the `Python3 (Data Science)` kernel which has most dependencies installed.

## Helpful Documents

* [How are the Reuters files organized?](https://link.springer.com/content/pdf/bbm%3A978-3-642-04533-2%2F1.pdf)
* [How to create a baseline model?](https://towardsdatascience.com/how-to-build-a-baseline-model-be6ce42389fc)
* [How can NLP be extended to different business use cases?](https://towardsdatascience.com/leveraging-on-nlp-to-gain-insights-in-social-media-news-broadcasting-ca89752ef638)

## Notebooks

1. `nb01_raw_to_clean.ipynb` - parses SGML files and restructures their content as JSON.
2. `nb02_features.ipynb` - does preprocessing by vectorizing document text and splitting between train and test datasets.
3. `nb03_baseline_models.ipynb` - creates baseline machine learning models for topic classification.
