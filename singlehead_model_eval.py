
# configure logging
import torch.nn as nn
import torch
import sentencepiece as spm
import math
import logging as log

vocab_size = 2000
d_model = 512  # embediding size
d_k = 64  # attention size
seq_length = 1000

model_path = "model_weights_sh.pth"
outfile='eval_transformer.log'
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler()
                ],
                )

#prompt = "Bloom lived in a big garden"
prompt = "THe Cat chase the "


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
        # A LayerNorm to normalize across the last dimension (d_model)
      

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
        # why is this needed; basically it allows the model from attending to only tokens in the past, that is in the left side of the current token, when you mulitply by V
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
        # Generated Text=Bloom lived in a big garden forestforestâ€œHelloœHelloœHello

        return out, attention

# Intialise all the layers

# add in the embdeiing part from previous layer
token_embedding = nn.Embedding(
    num_embeddings=vocab_size, embedding_dim=d_model)
pos_encoding = PositionalEncoding(d_model, max_len=seq_length)
# add in the attention layer
attention_mod = SingleHeadSelfAttention(d_model)
# Add a linear layer for prediction
prediction_layer1 = nn.Linear(d_model, vocab_size*2)
prediction_layer2 = nn.Linear(vocab_size*2, vocab_size)
layer_norm = nn.LayerNorm(d_model)


# We'll combine these into a simple pipeline
model = nn.ModuleList([token_embedding, pos_encoding,
                      attention_mod,layer_norm,prediction_layer1,prediction_layer2])

# The most important part is the Stochastic Gradient Descent part
# Using model.parameters() in optimizer.step() ensures all layers, including token_embedding, attention_mod, and prediction_layer, are updated
# gradient descent since this is a toy example
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

sp = spm.SentencePieceProcessor()
sp.load("llama_like.model")

# Place all in GPU
token_embedding.to('cuda')
pos_encoding.to('cuda')
attention_mod.to('cuda')
prediction_layer1.to('cuda')
prediction_layer2.to('cuda')
model.to('cuda')

log.info("Loading the  model...")


# load the model for evaluation
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode


generated_tokens = sp.encode(prompt, out_type=int)  # Tokenize input text
tokens = sp.encode(prompt, out_type=str)
token_ids = sp.encode(prompt, out_type=int)

log.info(f"Sentence: {prompt}")
log.info(f"Tokens:  {tokens}")
log.info(f"Token IDs: {token_ids}")

# Convert to tensor
input_tensor = torch.tensor(
    generated_tokens, dtype=torch.long).unsqueeze(0)  # (1, seq_length)
max_length = 10
for _ in range(max_length):
    # Get embedding
    embedded_tokens = token_embedding(input_tensor.to('cuda'))
    # Get attention and score
    score, attention = attention_mod(embedded_tokens)
    # Predict the next word
    hidden1 = prediction_layer1(score)  # (1, seq_length, vocab_size)
    logits = prediction_layer2(hidden1)  # (1, seq_length, vocab_size)
    
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

# Test the generation function

# Decode generated token IDs back to text
generated_text = sp.decode(generated_tokens)
log.info(f"Generated Text={generated_text}")
