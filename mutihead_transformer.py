# -*- coding: utf-8 -*-
"""LearnTransformer
## Learning Transformers by Doing

Based on my Colab file at 
    https://colab.research.google.com/drive/1qvaWLJCenxxTcKjHksHGicxdbZDdsm7i

Author: Alex Punnen and ChatGPT,CoPilot

Lets see how simple self attention works by writing a single headed attention and then training them on our small dataset.
"""

# Use bpe to tokenise the sence

"""It all starts with a Tokenizer that breaks words to a smaller set and creates a fixed set of vocabulary. Why fixed set vocabulary, because that is finally what is used for prediction. The model is trained to output the probability of the occurance of just the next token in say a 2000 set vocabulary. The highest probability item in that set gets selected as the next. Hence the need for a constant and fixes set vocabulary

In the LLAMA Paper they are using SentencePeiece tokenize

*We tokenize the data with the bytepair encoding (BPE) algorithm (Sennrich et al.,2015), using the implementation from SentencePiece 
Notably, we split all numbers into individual digits, and fallback to bytes to decompose unknown UTF-8 characters.*
"""

# !pip install datasets
# !pip install --upgrade sentencepiece


# configure logging
import torch.nn as nn
import torch
import sentencepiece as spm
from datasets import load_dataset
import math
import logging as log
import os
import datatime as date

outfile=f"/logs/{date.datetime.now()}simple_transformer.log"
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler()
                ],
                )

# Load the small dataset for training our tiny language model
ds = load_dataset("roneneldan/TinyStories")
train_size =100000
# use the dataset as text for training
log.info(f"Length of trainig data is  {len(ds['train']['text'])}")
# use half of this training data text
trainingdata = ds['train']['text'][:train_size]
log.info(f"Limiting training legth to {len(trainingdata)}")

# 1) Write the list to a file.
with open("./data/train.txt", "w", encoding="utf-8") as f:
    for line in trainingdata:
        # replace newline with space to keep each original text chunk on a single line
        #replace special characters
        line = line.replace("â€", "")
        f.write(line.replace("\n", " ") + "\n")

test_sentence = "The Cat sat on the Fence"
# We use a small vocab_size just for demo. LLaMA uses a much larger vocabulary (32k tokens).
vocab_size = 2000

# if file is not there
# this creates a vocab file and a model file
log.info("Training Non contextual tokeniser")
spm.SentencePieceTrainer.Train(
    input="train.txt",   # our training data
    model_prefix='llama_like',
    vocab_size=vocab_size,
    model_type='bpe',
    character_coverage=1.0,
    max_sentence_length=2048,
    treat_whitespace_as_suffix=True,
    split_digits=True               # This forces splitting "123" -> "1", "2", "3"
)

sp = spm.SentencePieceProcessor()
sp.load("./data/llama_like.model")

tokens = sp.encode(test_sentence, out_type=str)
token_ids = sp.encode(test_sentence, out_type=int)

log.info(f"Sentence: {test_sentence}")
log.info(f"Tokens:  {tokens}")
log.info(f"Token IDs: {token_ids}")

# get the vocabulary dictionary mapping
# print(sp.id_to_piece(60))

# Part 2

# Step 1: Prepare the data for training the Attention layer

# Now lets tokenise the entire text and generate a map of input_ids
all_token_ids = []

if not os.path.isfile("./data/token_ids.txt"):
    log.info("Tokenizing text...")
    with open("./data/train.txt", "r", encoding="utf-8") as f:
        for line in f:
            # Encode each line to token IDs
            line_ids = sp.encode(line, out_type=int)
            # Append them, maybe add a special token like <eol> if desired
            all_token_ids.extend(line_ids)
            # all_token_ids.append(eol_id)  # If you have a special EOL token
    # Write token IDs to file
    with open("./data/token_ids.txt", "w", encoding="utf-8") as f:
        for token_id in all_token_ids:
            f.write(f"{token_id}\n")
else:
    log.info("Token ids already present in file")
    #read token ids from file
    all_token_ids = []
    with open("./data/token_ids.txt", "r", encoding="utf-8") as f:
        for line in f:
            all_token_ids.append(int(line))
        
log.info(f"Total tokens:  {len(all_token_ids)}")

# Lets resize the input ids for training

log.info("Resizing input_ids...")

# convert these to torch tensor
input_ids = torch.tensor(all_token_ids, dtype=torch.long).unsqueeze(0)
log.info(f"input_ids.shape={input_ids.shape}")
# shape these (torch.Size([1, 380627])) chunk to batchsize of 1 and length of 50
seq_length = 1000
input_ids = input_ids.squeeze(0)  # Remove batch dim, now shape = (380627,)
# How many 50-token chunks we can make
num_chunks = input_ids.shape[0] // seq_length

# Truncate to nearest multiple of 50
input_ids = input_ids[:num_chunks * seq_length]
# Reshape to (num_chunks, 50), each row is a sequence of 50 tokens
input_ids = input_ids.view(num_chunks, seq_length)

log.info(f"New shape:= {input_ids.shape}")  # Should be (num_chunks, 50)

# this will be same as labels
labels = input_ids.clone()
vocab_size = 2000
d_model = 512  # embediding size
d_k = 64  # attention size

# we need to add positional encoding to the input_ids
# Positional encoding is a way to provide the model with information about the position of each token in the sequence.
# This is important because the model has no inherent sense of order in the tokens, since it only sees them as embeddings.
# generated by LLM


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        # Create a long enough 'pe' matrix of shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        # Even indices (2i) -> sine
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices (2i+1) -> cosine
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer so it's moved to GPU automatically if needed
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        We add positional encoding up to seq_len from the precomputed 'pe'.
        """
        seq_len = x.size(1)
        # pe[:seq_len] -> shape [seq_len, d_model]
        # We unsqueeze(0) so that shape becomes [1, seq_len, d_model],
        # allowing addition to x which is [batch_size, seq_len, d_model].
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


"""### Step: Adding in a Simple Attention Class"""


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model):
        """
        d_model: dimension for Q, K, V
        use_output_proj: if True, applies a final linear W_O
        """
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
          Forward lyer of SingleHeadedAttention
        """
        B, seq_len, d_model = x.shape  # B is batch size , seq_len is the length of the sequence , and d_model is the embedding size (512) # torch.Size([1, 999, 512])

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        # Apply the mask to the attention scores
        # why is this needed; basically it allows the model from attending to only tokens in the past, that is in the left side of 
        # the current token, when you mulitply by V
        # the left side becomes the lower triangular matrix; and right side the future tokens are  the upper triangular matrix
        # We build an upper-triangular mask (set to -inf) that zeros out attention (the next softmmax layer will set it to zero)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device), diagonal=1
        ).bool()
        attention = attention.masked_fill(causal_mask, float('-inf'))
        attention = torch.softmax(attention, dim=-1)
        score = torch.matmul(attention, V)
        # ----- [1] Add residual connection ----- ttodo take this out
        out = x + score # without this the model output is not good
        return out, attention


log.info(f"vocab_size={vocab_size} embedding_dim/d_model={d_model}")

# Intialise all the layers

# add in the embdeiing part from previous layer
token_embedding = nn.Embedding(
    num_embeddings=vocab_size, embedding_dim=d_model)
pos_encoding = PositionalEncoding(d_model, max_len=seq_length)
# add in the attention layer

# Add a linear layer for prediction
num_heads=2
multihead_attention = nn.ModuleList()
for _ in range(num_heads):
    attention_mod = SingleHeadSelfAttention(d_model)
    multihead_attention.append(attention_mod)
    
prediction_layer1 = nn.Linear(d_model*num_heads, vocab_size*2) # as we are concatenating the heads output
layer_norm1 = nn.LayerNorm(vocab_size*2) 
prediction_layer2 = nn.Linear(vocab_size*2, vocab_size)
layer_norm2 = nn.LayerNorm(vocab_size) # last dimension is the vocab size


# Define the loss function
loss_function = nn.CrossEntropyLoss()
log.info(f"Length of input ids ={len(input_ids)}")

# We'll combine these into a simple pipeline
model = nn.ModuleList([token_embedding, pos_encoding,
                      multihead_attention,layer_norm1,layer_norm2,prediction_layer1,prediction_layer2])

# The most important part is the Stochastic Gradient Descent part
# Using model.parameters() in optimizer.step() ensures all layers, including token_embedding, attention_mod, and prediction_layer, are updated
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # SGD is unstable and hence we use this

# with higher learning loss is Nan

assert False == torch.isnan(input_ids).any()
assert False == torch.isinf(input_ids).any()

# Place all in GPU
token_embedding.to('cuda')
pos_encoding.to('cuda')
attention_mod.to('cuda')
prediction_layer1.to('cuda')
prediction_layer2.to('cuda')
model.to('cuda')

log.info("Training model...")

model.train()
batch_size = 25
N, seq_length = input_ids.shape
log.info(f"N= {N} seq_length= {seq_length}")
num_batches = N // batch_size

for epoch in range(10):
    for start_idx in range(0, N, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > N:
            break  # in case N not multiple of batch_size

        # Slice out a batch
        batch_input = input_ids[start_idx:end_idx, :]   # (B, seq_length)
        batch_labels = labels[start_idx:end_idx, :]     # (B, seq_length)
        if epoch == 0 and start_idx == 0:
            log.info(f"batch_input.shape={batch_input.shape}")
            log.info(f"batch_labels.shape={batch_labels.shape}")

        # Move to GPU
        batch_input = batch_input.to('cuda')
        batch_labels = batch_labels.to('cuda')

        # 1) Shift input & labels so model predicts next token
        #    shape -> (B, seq_length-1)
        trimmed_input = batch_input[:, :-1]
        target_labels = batch_labels[:, 1:]
        if epoch == 0 and start_idx == 0:
            # take 10 tokens
            log.info("Example input: %s", sp.decode(trimmed_input[0].tolist()[:10]))
            log.info("Example labels: %s",sp.decode(target_labels[0].tolist()[:10]))

        embedded_tokens = token_embedding(trimmed_input)
        # shape remains (batch_size, seq_len, d_model)
        pos_embedded_tokens = pos_encoding(embedded_tokens)
        # Initialise the scores
        # Initialize an empty list to store scores
        head_outputs = []
        # get attention and score from multihead attention
        for attention_mod in multihead_attention:
            score,_ = attention_mod(pos_embedded_tokens)
            head_outputs.append(score)
        #Convert list of scores into a single tensor (concatenation or summation)
        score = torch.cat(head_outputs, dim=-1)  # Concatenate along the last dimension
        #print(score.shape) # torch.Size([50, 999, 1024]) #  #last dim is dmodel*2 (num_heads)
        # Predict the next word
        hidden1 = prediction_layer1(score)  # Project to vocabulary size
        hidden1 = layer_norm1(hidden1)         # add layer norm
        logits = prediction_layer2(hidden1)  # through few linear layers
        logits = layer_norm2(logits)      # add layer norm
        # the last dimension of the output tensor represents the vocabulary size or the number of classes.
        # Therefore, applying softmax along the last dimension (dim=-1)
        predicted_probs = torch.softmax(logits, dim=-1)  # Get probabilities
        # Get the predicted word (token ID)
        predicted_token_id = torch.argmax(predicted_probs, dim=-1)
        # Calculate the loss # crossentropy already does softmax inside
        # If your input has 49 tokens, you predict 49 next tokens.
        loss = loss_function(
            logits.reshape(-1, vocab_size),
            target_labels.reshape(-1)
        )
        loss.backward()
        # We are not discarding the loss or ignoring it; rather, we’re enforcing a limit on the size of the update to avoid erratic jumps.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # print training progress occasionally
        if (start_idx // batch_size) % 50 == 0:
            log.info("[Epoch=%d | Batch=%d] loss=%.4f", epoch+1, start_idx//batch_size, loss.item())
        if loss.item() < 0.5:
            break
        # free gpu memory
        del batch_input
        del batch_labels
        torch.cuda.empty_cache()

    log.info(f"---------Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")

"""# Use the trained model to predict"""

# save the model weights
torch.save(model.state_dict(), f"./weights/{date.datetime.now()}_model_weights_mh.pth")
log.info("Model weights saved")
model.eval()  # Set to evaluation mode

# Test the generation function
prompt = "Bloom lived in a big garden"

generated_tokens = sp.encode(prompt, out_type=int)  # Tokenize input text

# Convert to tensor
input_tensor = torch.tensor(
    generated_tokens, dtype=torch.long).unsqueeze(0)  # (1, seq_length)
max_length = 100
for _ in range(max_length):
    # Get embedding
    embedded_tokens = token_embedding(input_tensor.to('cuda'))
    pos_embedded_tokens = pos_encoding(embedded_tokens)
    # Initialize an empty list to store scores
    head_outputs = []
    # Get attention and score
    # get attention and score from multihead attention
    for attention_mod in multihead_attention:
        score,_ = attention_mod(pos_embedded_tokens)
        head_outputs.append(score)
    #Convert list of scores into a single tensor (concatenation or summation)
    score = torch.cat(head_outputs, dim=-1)  # Concatenate along the last dimension
    #print(score.shape) # torch.Size([50, 999, 1024]) #  #last dim is dmodel*2 (num_heads)
    # Predict the next word
    hidden1 = prediction_layer1(score)  # Project to vocabulary size
    hidden1 = layer_norm1(hidden1)         # add layer norm
    logits = prediction_layer2(hidden1)  # through few linear layers
    logits = layer_norm2(logits)      # add layer norm
    logits = layer_norm2(logits)
    # Get the last token's logits (for autoregressive prediction)
    next_token_logits = logits[:, -1, :]  # Shape: (1, vocab_size)
    # Convert logits to token probabilities
    next_token_id = torch.argmax(next_token_logits, dim=-1)  # (1,)
    # Append new token
    generated_tokens.append(next_token_id.item())
    # Stop if we generate an EOS token (optional)
    if next_token_id.item() == sp.eos_id():  # Ensure your tokenizer has an EOS token
        break

    # Update input tensor with new token for next iteration
    input_tensor = torch.tensor(
        generated_tokens, dtype=torch.long).unsqueeze(0)

# Decode generated token IDs back to text
generated_text = sp.decode(generated_tokens)
log.info(f"Generated Text={generated_text}")
