from src.utils import Tokenizer
from src.model import Transformer
import torch
import torch.nn.functional as F


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


english_tokenizer = Tokenizer.load_vocab(config['src_output_tokenizers'])
french_tokenizer = Tokenizer.load_vocab(config['tgt_output_tokenizers'])


model = Transformer(
    english_tokenizer.vocab_size,
    french_tokenizer.vocab_size,
)

model.load_state_dict(torch.load(config['model_output'], weights_only=True))
model.eval()

# Create encoder input.
sentence = "I am swimming in a lake."
encoded_sentence = english_tokenizer.encode_sentence(sentence)
encoder_input = encoded_sentence.unsqueeze(0) # Shape: 1 x EncSeqLength

# Create decoder input.
decoder_sentence = french_tokenizer.encode_sentence("<sos>", add_sos_eos=False)
decoder_input = decoder_sentence.unsqueeze(0) # Shape: 1 x 1

# Start decoding loop
with torch.no_grad():
    for _ in range(20): # Using _ since i is not needed
        # 1. Forward pass
        # logits shape: 1 x CurrentSeqLength x VocabSize
        logits = model(encoder_input, decoder_input)
        
        # 2. Get the logits for the *last* token position (the one we are predicting)
        # next_token_logits shape: 1 x VocabSize
        next_token_logits = logits[:, -1, :]
        
        # 3. Greedy step: get the token ID with the highest probability
        # next_token_id shape: 1 (or 1x1)
        next_token_id = next_token_logits.argmax(dim=-1)
        
        # 4. Check for end-of-sequence
        if next_token_id.item() == french_tokenizer.word_to_id['<eos>']:
            print("Translation finished with <eos>")
            break
            
        # 5. Append the new token ID to the decoder input for the next step
        # Using torch.cat and dim=1 (the sequence dimension)
        decoder_input = torch.cat((decoder_input, next_token_id.unsqueeze(0)), dim=1)
        
        # For debugging/tracking
        print(f"Current Translation: {french_tokenizer.decode(decoder_input.squeeze(0))}")
        
    print("\nFinal Decoder Input Tensor:")
    print(decoder_input)
