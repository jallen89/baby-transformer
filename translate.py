from src.utils import Tokenizer
import torch

config = {
    'data_path': 'data/fra.txt',
    'train_split': 0.8,
    'val_split': 0.1,
    'batch_size': 64,
    'num_epochs': 1,
    'lr': 0.001,
    'model_output': 'weights/english-french.pt',
    'src_output_tokenizers': 'weights/english-tokenizer.json',
    'tgt_output_tokenizers': 'weights/french-tokenizer.json',
}


english_tokenizer = Tokenizer.load(config['src_output_tokenizers'])
french_tokenizer = Tokenizer.load(config['tgt_output_tokenizers'])

model = torch.load(config['model_output'])



# 1. Load in model. 
# 2. Setup Translation loop. 
# 3. We also need the decoders/encoders loaded. 