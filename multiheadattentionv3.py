"""
Using torch.MultiHeadAttention instead of custom implementation
"""

# !pip install datasets
# !pip install --upgrade sentencepiece

# configure logging
import torch.nn.functional as F
import torch.nn as nn
import torch
import sentencepiece as spm
from datasets import load_dataset
import math
import logging as log
import os
import gc
import numpy as np
import glob
import random
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup

datetimesatmp = datetime.now().strftime("%Y%m%d%H%M%S")

outfile = f"multihead_{datetimesatmp}_.log"
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler()
                ],
                force=True,
                )

loss_log_file_base = f"loss_log_{datetimesatmp}_.npy"
loss_log_file = f"loss_log_{datetimesatmp}_.npy.npz"

# Initialize loss log
if not os.path.exists(loss_log_file):
    np.savez_compressed(loss_log_file, loss=np.array([]))
    log.info(f"Created path loss file {loss_log_file}")

# Load the small dataset for training our tiny language model
ds = load_dataset("roneneldan/TinyStories")
# load the full setis about 2 million 2119719,
train_size = len(ds['train']['text'])

# use the dataset as text for training
log.info(f"Length of trainig data is  {len(ds['train']['text'])}")
# use half of this training data text
trainingdata = ds['train']['text'][:train_size]

TRAIN_FILE = "train.txt"
if not os.path.exists(TRAIN_FILE):
    # 1) Write the list to a file.
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for line in trainingdata:
            # replace newline with space to keep each original text chunk on a single line
            # replace special characters
            line = line.replace("â€", "")
            f.write(line.replace("\n", " ") + "\n")


# We use a small vocab_size just for demo. LLaMA uses a much larger vocabulary (32k tokens).
vocab_size = 2000
vocab_trainingdata = ds['train']['text'][:vocab_size]
log.info(f"Vocaab training legth to {len(trainingdata)}")
# 1) Write the list to a file.
with open("vocab_train.txt", "w", encoding="utf-8") as f:
    for line in vocab_trainingdata:
        # replace newline with space to keep each original text chunk on a single line
        # replace special characters
        line = line.replace("â€", "")
        f.write(line.replace("\n", " ") + "\n")

test_sentence = "The Cat sat on the Fence"


# if file is not there
# this creates a vocab file and a model file
log.info("Training Non contextual tokeniser")
spm.SentencePieceTrainer.Train(
    input="vocab_train.txt",   # our training data
    model_prefix='llama_like',
    vocab_size=vocab_size,
    model_type='bpe',
    character_coverage=1.0,
    max_sentence_length=2048,
    treat_whitespace_as_suffix=True,
    split_digits=True               # This forces splitting "123" -> "1", "2", "3"
)
log.info("Training of Vocabulary complete")

sp = spm.SentencePieceProcessor()
sp.load("llama_like.model")

tokens = sp.encode(test_sentence, out_type=str)
token_ids = sp.encode(test_sentence, out_type=int)

log.info(f"Sentence: {test_sentence}")
log.info(f"Tokens:  {tokens}")
log.info(f"Token IDs: {token_ids}")


# this takes a long time about an hour
def tokenize_and_split(input_file, seq_length, out_dir):
    """
    Tokenizes the entire dataset and splits it into multiple files of fixed sequence length.
    """
    if os.path.exists(out_dir):
        log.error(f"Token folder already exisitng: {out_dir} Not overwriting")
        return
    os.makedirs(out_dir, exist_ok=False)

    token_list = []
    file_count = 0
    CHUNK_SIZE = 1024 * 1024 * 100  # 100MB chunk for readin
    with open(input_file, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)  # Read large chunk
            if not chunk:
                break  # Stop at EOF

            token_ids = sp.encode(chunk, out_type=int)  # Tokenize chunk
            token_list.extend(token_ids)

            # Save when we have enough tokens for multiple seq_length chunks
            while len(token_list) >= seq_length:
                split_tokens = token_list[:seq_length]
                token_list = token_list[seq_length:]  # Remove saved tokens

                # Save to a file
                filename = os.path.join(out_dir, f"tokens_{file_count}.npy")
                np.save(filename, np.array(split_tokens, dtype=np.int32))
                file_count += 1

    log.info(
        f"Tokenization complete. Saved {file_count} files in '{OUTPUT_DIR}'")

# read the tokenised batches in random order


def tokenized_batch_loader(output_dir, batch_size, seq_length):
    """
    Generator function to load tokenized files in batches while maintaining exact sequence length.
    """
    step = 0
    token_files = sorted(glob.glob(os.path.join(output_dir, "tokens_*.npy")))
    random.shuffle(token_files)  # Shuffle for better training dynamics

    token_buffer = []  # Buffer to store tokens across files
    batch_buffer = []  # Buffer to store batches

    for token_file in token_files:
        tokens = np.load(token_file).tolist()
        token_buffer.extend(tokens)  # Append to buffer

        # Process the buffer into seq_length chunks
        while len(token_buffer) >= seq_length: # the files have token lengths as  FILE_SEQ_LENGTH = 1000000
            batch_buffer.append(token_buffer[:seq_length])
            token_buffer = token_buffer[seq_length:]  # Remove used tokens

            # Yield when we have enough sequences for a batch
            if len(batch_buffer) == batch_size:
                step += 1
                batch_tensor = torch.tensor(batch_buffer, dtype=torch.long)
                yield batch_tensor[:, :-1].cuda(), batch_tensor[:, 1:].cuda(), step
                batch_buffer = []  # Reset buffer

    # Process remaining sequences if they form a batch
    if batch_buffer:
        step += 1
        batch_tensor = torch.tensor(batch_buffer, dtype=torch.long)
        yield batch_tensor[:, :-1].cuda(), batch_tensor[:, 1:].cuda(), step


OUTPUT_DIR = "./train_tokens"
FILE_SEQ_LENGTH = 1000000
# Run the tokenizer
tokenize_and_split(TRAIN_FILE, seq_length=FILE_SEQ_LENGTH, out_dir=OUTPUT_DIR)


# we need to add positional encoding to the input_ids
# Positional encoding is a way to provide the model with information about the position of each token in the sequence.
# This is important because the model has no inherent sense of order in the tokens, since it only sees them as embeddings.
# generated by LLM
class PositionalEncoding(nn.Module):  # generated by ChatGPT
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

# Instead of doing Multi head sequentially like previous, lets do it in parallel


# Intialise all the layers
vocab_size = 2000
d_model = 512  # embediding size
d_k = 64  # attention size
read_seq_length = 1000
log.info(f"vocab_size={vocab_size} embedding_dim/d_model={d_model}")

# add in the embdeiing part from previous layer
token_embedding = nn.Embedding(
    num_embeddings=vocab_size, embedding_dim=d_model)
pos_encoding = PositionalEncoding(d_model, max_len=read_seq_length)
# add in the attention layer

vocab_size = 2000
d_model = 512  # embediding size
d_k = 64  # attention size
# Add a linear layer for prediction
num_heads = 8  # work on A100 GPU

# multihead_attention = MultiHeadSelfAttention(d_model, num_heads)
multihead_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1,batch_first=True)
layer_norm1 = nn.LayerNorm(d_model)
prediction_layer2 = nn.Linear(d_model, vocab_size)
layer_norm2 = nn.LayerNorm(vocab_size)  # last dimension is the vocab size

# Define the loss function
loss_function = nn.CrossEntropyLoss()
# We'll combine these into a simple pipeline
model = nn.ModuleList([token_embedding, pos_encoding,
                      multihead_attention, layer_norm1, layer_norm2, prediction_layer2])
# SGD is unstable and hence we use this
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# with higher learning loss is Nan

# Place all in GPU
token_embedding.to('cuda')
pos_encoding.to('cuda')
multihead_attention.to('cuda')
prediction_layer2.to('cuda')
model.to('cuda')
# NO NEED TO EXECUTE THIS AGAIN ( this need A100, )
log.info("Training model...")
model_path = "./weights/model_weights20250217094947.pth" # epoch 1
model_path = "./weights/model_weights20250217130252.pth"  # epoch 2 # loss: 2.4915
model.load_state_dict(torch.load(model_path))
# load the previous weights

BATCH_SIZE = 20  # 1 GB for Batch size 10
model.train()
loss_value_list = []

# Count number of token files
token_files = glob.glob(os.path.join(OUTPUT_DIR, "tokens_*.npy"))
num_files = len(token_files)

# Compute total steps
steps_per_file = (FILE_SEQ_LENGTH // read_seq_length) // BATCH_SIZE
total_steps = num_files * steps_per_file
log.info(f"Total Training Steps: {total_steps}")

# Add a scheduler to adjust the learning rate
total_training_steps = total_steps
warmup_steps = total_training_steps // 10  # 10% warmup


for epoch in range(1):
    for batch_input, batch_labels, step in tokenized_batch_loader(OUTPUT_DIR, batch_size=BATCH_SIZE, seq_length=read_seq_length):
        if epoch == 0 and step == 1:
            log.info(f"batch_input.shape={batch_input.shape}")
            log.info(f"batch_labels.shape={batch_labels.shape}")

        # Move to GPU
        batch_input = batch_input.to('cuda')
        batch_labels = batch_labels.to('cuda')

        # 1) Shift input & labels so model predicts next token
        #    shape -> (B, seq_length-1)
        trimmed_input = batch_input
        target_labels = batch_labels
        embedded_tokens = token_embedding(trimmed_input)
        # shape remains (batch_size, seq_len, d_model)
        pos_embedded_tokens = pos_encoding(embedded_tokens)

        # **Create Causal Mask**: Prevents attention to future tokens
        seq_len =  pos_embedded_tokens.shape[1] 
         # Create causal mask (Upper triangular matrix with -inf masking)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(pos_embedded_tokens.device)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))  # Convert to -inf for softmax masking
        if epoch == 0 and step == 1:
            # take 10 tokens
            log.info("Example input: %s", sp.decode(
                trimmed_input[0].tolist()[:10]))
            log.info("Example labels: %s", sp.decode(
                target_labels[0].tolist()[:10]))
            log.info(
                f"pos_embedded_tokens.shape={pos_embedded_tokens.shape}, causal_mask.shape={causal_mask.shape}"+
                f"batch_size={BATCH_SIZE}, seq_length={read_seq_length}")
        # Ensure `pos_embedded_tokens` has correct shape for MultiheadAttention
        #pos_embedded_tokens = pos_embedded_tokens.permute(1, 0, 2)  # (seq_len, batch, d_model)

        # **Apply MultiHeadAttention with Mask**
        score, _ = multihead_attention(
            pos_embedded_tokens, pos_embedded_tokens, pos_embedded_tokens, attn_mask=causal_mask
        )
        # Convert back to (batch, seq_len, d_model)
        #score = score.permute(1, 0, 2)
        hidden1 = score + pos_embedded_tokens  # add a residual layer for score
        hidden1 = layer_norm1(hidden1)         # add layer norm
        logits = prediction_layer2(hidden1)  # through few linear layers
        logits = layer_norm2(logits)      # add layer norm
        # the last dimension of the output tensor represents the vocabulary size or the number of classes.
        # Therefore, applying softmax along the last dimension (dim=-1)
        # predicted_probs = torch.softmax(logits, dim=-1)  # Get probabilities
        # # Get the predicted word (token ID)
        # predicted_token_id = torch.argmax(predicted_probs, dim=-1)
        # # Calculate the loss # crossentropy already does softmax inside
        # If your input has 49 tokens, you predict 49 next tokens.
        loss = loss_function(
            logits.reshape(-1, vocab_size),
            target_labels.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        # We are not discarding the loss or ignoring it; rather, we’re enforcing a limit on the size of the update to avoid erratic jumps.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        # print training progress occasionally
        loss_value_list.append((epoch, step, loss.item()))
        if step % 100 ==0:
            log.info("[Epoch=%d | Step=%d/%d] loss=%.4f",
                     epoch+1, step, total_steps,loss.item())
        if step % 100 == 0:
            data = np.load(loss_log_file,allow_pickle=True)
            loss_history = []
            if "loss" in data:
                # Convert to list for appending
                loss_history = data["loss"].tolist()
            loss_list = loss_history + loss_value_list
            np.savez_compressed(
                loss_log_file, loss=np.array(loss_list, dtype=object))
            loss_value_list = []
        if loss.item() < 0.5:
            break
        # free gpu memory
        del batch_input, batch_labels, trimmed_input, target_labels, logits
        gc.collect()
        torch.cuda.empty_cache()

    log.info(f"---------Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")

"""# Use the trained model to predict"""

# save the model weights
# add data to the model

torch.save(model.state_dict(), f"model_weights{datetimesatmp}.pth")
log.info(f"Model weights saved at  model_weights{datetimesatmp}.pth")


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
    # Get attention and score
    # shape remains (batch_size, seq_len, d_model)
    pos_embedded_tokens = pos_encoding(embedded_tokens)
    # parallize multihead attention via matrix dimensions
    score, _ = multihead_attention(
        pos_embedded_tokens, pos_embedded_tokens, pos_embedded_tokens)
    hidden1 = score + pos_embedded_tokens  # add a residual layer for score
    hidden1 = layer_norm1(hidden1)         # add layer norm
    logits = prediction_layer2(hidden1)  # through few linear layers
    logits = layer_norm2(logits)      # add layer norm
    predicted_probs = torch.softmax(logits, dim=-1)  # Get probabilities
    # Get the last token's logits (for autoregressive prediction)
    next_token_logits = predicted_probs[:, -1, :]  # Shape: (1, vocab_size)
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
