# multilingual-clf

### Data
The data has been used from Kaggle cometion [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/overview)

### Workings

Refer to my notebook to see how all of the stuff works out. [Kaggle Notebook](https://www.kaggle.com/xanthate/xlm-roberta-large-pytorch-xla-tpu)

* Use PyTorch nightly. PyTorch and torch_xla seems to be unstable a lot of times. 

* ```bert-multilingual-uncased``` models works very easily. There are no ```SIGKILL``` or other memory issues. 

* ```xlm-roberta-base``` model works too with ```batch_size=8```.

* ```xlm-roberta-large``` is a lot trickier. Garbage collection, limiting the loading of dataloader to once is required.

    * Model needs to be called only once and wrapped with a wrapper function provided in ```torch_xla``` library.
    

### Todo

- [ ] Add Multiple Sample Dropout
- [ ] Mixed precision training

