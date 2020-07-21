from transformers import BertModel, BertTokenizer


BATCH_SIZE = 64
EPOCHS = 10
MAX_LENGTH = 192
MODEL_PATH = 'bert-base-multilingual-uncased'

TRAIN_FILE_PATH = ''
VALID_FILE_PATH = ''


TOKENIZER = BertTokenizer.from_pretrained(MODEL_PATH, max_len=MAX_LENGTH)
MODEL = BertModel.from_pretrained(MODEL_PATH)