# LearnTransformer: A Simple Single-Headed Self-Attention Language Model

This repository provides an illustrative example of how to build a very simple single-headed transformer-based language model using **PyTorch** and **SentencePiece**. It follows a step-by-step approach:

1. **Tokenizing the data** using BPE (Byte Pair Encoding).  
2. **Embedding** the tokens.  
3. **Applying a single-head self-attention** mechanism.  
4. **Adding positional encoding** and a small feed-forward network.  
5. **Training** on a tiny dataset from **TinyStories**.  
6. **Generating** new text from the trained model.  

Below is a walk-through of the code, with code snippets and brief explanations of each step.

---
