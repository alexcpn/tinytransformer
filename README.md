# LearnTransformer: A set of simple Transformer Models to learn Self Attention

This repository provides an illustrative example of how to build a very simple single-headed transformer-based language model using **PyTorch** and **SentencePiece**. It follows a step-by-step approach:

1. **Tokenizing the data** using BPE (Byte Pair Encoding).  
2. **Embedding** the tokens.  
3. **Applying a single-head self-attention** mechanism.  
4. **Adding positional encoding** and a small feed-forward network.  
5. **Training** on a tiny dataset from **TinyStories**.  
6. **Generating** new text from the trained model.  

Start from the simplest 
[simple_transformer.py](simple_transformer.py) which has a `SingleHeadSelfAttention` to

the next [multihead_model_eval.py](multihead_model_eval.py) which demonstrates using the Multihead concept in a sequential manner using the same `SingleHeadSelfAttention` block in above

```
multihead_attention = nn.ModuleList()
for _ in range(num_heads):
    attention_mod = SingleHeadSelfAttention(d_model)
    multihead_attention.append(attention_mod)
```

and the next [multiheadattentionv2.py](multiheadattentionv2.py) which vectorises the above via `MultiHeadSelfAttention` class

and finally [multiheadattentionv3.py](multiheadattentionv3.py) which uses  the proper `torch.nn.MultiheadAttention` class

```
multihead_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1,batch_first=True)
```

Dataset used is  [Tiny Stories](https://huggingface.co/datasets/roneneldan/TinyStories) hosted in HuggingFace.
 This is  an interesting dataset in itself. This is from the paper *TinyStories: How Small Can Language Models Be and Still Speak Coherent English? (Ronen, Yuanzhi 2023)* . It is specially designed for very small models like ours.


This is the [result](logs/multihead_20250217154759_.log) of the last one here trained on three epochs on a 6 GB GPU RAM  NVIDIA GeForce RTX 3060 laptop GPU. Not bad for a single attention block using just the simplest generation (argmax - max probablity of the token in the vocabulary)


```
prompt = "Bloom lived in a big garden"
```

```
17-Feb-25 18:21:33 - Generated Text=Bloom lived in a big garden with many other animals. One day, Billy was playing with his friends, but he didn't want to share. He wanted to play with his friends, but he didn't want to share. He tried to take a nap, but he was too shy to share. He tried to take a nap, but he was too shy to take a nap. He was so sad and he didn't know what to do. He decided to take a nap. 
```

Not bad I shoud say; More details are here in this Medium [article](https://medium.com/towards-artificial-intelligence/explaining-transformers-as-simple-as-possible-through-a-small-language-model-6e6038941ca7) but the code and especially the colab notebooks are self-explantory

---
