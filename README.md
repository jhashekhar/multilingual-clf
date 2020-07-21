# multilingual-clf

### Data
The data has been used from Kaggle cometion [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/overview)

### Workings

Refer to my notebook to see how all of the stuff works out. [Kaggle Notebook](https://www.kaggle.com/xanthate/xlm-roberta-large-pytorch-xla-tpu)

* Use PyTorch nightly. PyTorch and torch_xla seems to be unstable a lot of times. 

* ```bert-multilingual-uncased``` models works very easily. There are no ```SIGKILL``` and other memory issues. 

* ```xlm-roberta-base``` model works too with ```batch_size=8```.

* ```xlm-roberta-large``` is a lot trickier. Garbage collection, loading the data to once is required.

### Todo

- [] Add Multiple Sample Dropout
- [] Mixed precision training

