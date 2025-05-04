
#MIT License

#Copyright (c) 2025 Amirprx3, GameDevRichtofen-G

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#-> Importing necessary libraries from PyTorch and math. PyTorch is for building the neural network, and math is needed for the positional encoding formula.<-#
import torch
import torch.nn as nn
import torch.nn.functional as F #-> Not actually used in this specific code, but often useful for things like softmax, relu, etc.<-#
import math #-> Needed for calculating the positional encoding values using sine and cosine.<-#

#-> This class is supposed to add positional information to the input embeddings. Transformers don't inherently know the order of words, so we need to tell them!<-#
class PositionalEncoding(nn.Module):

    #-> Constructor for the positional encoding layer.<-#
    def __init__(self, d_model, max_len=5000):
        #-> Call the parent class constructor. Always do this!<-#
        super().__init__()

        #-> Create a tensor to hold the positional encodings. It's going to have a shape of (max_len, d_model).<-#
        pe = torch.zeros(max_len, d_model)
        #-> Create a tensor representing the positions (0, 1, 2, ... max_len-1). Unsqueeze adds a dimension so it's a column vector.<-#
        position = torch.arange(0, max_len,dtype=torch.float()).unsqueeze()
        #-> Calculate the division term for the sine and cosine formulas. This term decreases as the dimension index increases.<-#
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #-> Apply the sine function to the even indices of the positional encoding tensor.<-#
        pe[:, 0::2] = torch.sin(position * div_term)
        #-> Apply the cosine function to the odd indices.<-#
        pe[:, 1::2] = torch.cos(position * div_term)
        #-> Reshape the tensor to (max_len, 1, d_model). The '1' dimension is for broadcasting across batches later. Transpose is confusing, but the end shape is key.<-#
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        #-> Register this tensor as a buffer. Buffers are tensors that are part of the module's state but aren't parameters (not updated by gradients).<-#
        self.register_buffer('pe', pe)

    #-> This is how the positional encoding is applied during the forward pass.<-#
    def forward(self, x):
        #-> Add the positional encoding to the input tensor 'x'. We only take the positional encodings up to the current sequence length of 'x'.<-#
        x = x + self.pe[:x.size(0), :]
        #-> Return the input with positional information added.<-#
        return x

#-> This looks like the definition of a single Decoder Block, which is a key component of the Transformer.<-#
class DecoderBlock(nn.Module):

    #-> Constructor for the decoder block. d_model is the embedding dimension, nhead is the number of attention heads.<-#
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        #-> Call the parent constructor.<-#
        super().__init__()

        #-> First, a self-attention layer. This is where the model looks at the input sequence itself to figure out relationships. batch_first=False means input is (seq_len, batch_size, d_model).<-#
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        #-> Layer normalization is applied after attention and after the feed-forward network. It helps stabilize training.<-#
        self.norm1 = nn.LayerNorm(d_model)
        #-> Dropout is used to prevent overfitting by randomly setting some activations to zero.<-#
        self.dropout1 = nn.Dropout(dropout)

        #-> Then, a two-layer feed-forward network. It processes each position separately.<-#
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        #-> ReLU is a common activation function.<-#
        self.relu = nn.ReLU()
        #-> Dropout within the feed-forward network.<-#
        self.dropout = nn.Dropout(dropout)
        #-> The second linear layer projects back to the original d_model dimension.<-#
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        #-> Second layer norm.<-#
        self.norm2 = nn.LayerNorm(d_model)
        #-> Second dropout.<-#
        self.dropout2 = nn.Dropout(dropout)

        #-> Hmm, another activation defined here? GELU is also an activation function, often used in modern transformers like BERT/GPT.<-#
        self.activation = nn.GELU() #-> It seems the later definition of DecoderBlock uses this one.<-#

    #-> This is the forward pass through one decoder block. It takes input 'x', an attention 'mask', and a 'key_padding_mask'.<-#
    def forward(self, x, mask=None, key_padding_mask=None):

        #-> Store the input for the residual connection. This helps gradients flow back during training.<-#
        residual = x

        #-> Perform self-attention. Query, Key, and Value are all the input 'x' for self-attention. The mask prevents attending to future tokens (for language modeling) and the padding mask prevents attending to padding tokens.<-#
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask
        )
        #-> Add the attention output to the original input (residual connection) and apply dropout.<-#
        x = residual + self.dropout1(attn_output)
        #-> Apply layer normalization.<-#
        x = self.norm1(x)

        #-> Start the feed-forward part. Store the current 'x' for the next residual connection.<-#
        # Feed Forward
        residual = x
        #-> Pass through the first linear layer, activation (ReLU was defined earlier, but the later definition uses GELU), dropout, and the second linear layer.<-#
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        #-> Add the feed-forward output to the residual and apply dropout.<-#
        x = residual + self.dropout2(ff_output)
        #-> Apply the second layer normalization.<-#
        x = self.norm2(x)

        #-> Return the output of the decoder block.<-#
        return x

#-> Wait, this is another DecoderBlock definition? It seems to overwrite the one above. Let's look at the differences.<-#
class DecoderBlock(nn.Module): #-> This definition replaces the previous one.<-#
    #-> Constructor seems very similar...<-#
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        #-> Ah, here's a difference! This one uses batch_first=True in MultiheadAttention. This means the input/output shape will be (batch_size, seq_len, d_model), which is often more convenient.<-#
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        #-> Norms and dropouts are the same.<-#
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        #-> Feed-forward layers are the same.<-#
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        #-> Okay, this definition uses GELU as the activation here, unlike the previous one which had ReLU. This is probably the intended activation.<-#
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    #-> The forward pass looks structurally the same as the previous definition.<-#
    def forward(self, x, mask=None, key_padding_mask=None):

        #-> Residual connection before attention.<-#
        residual = x

        #-> Self-attention step. Note that because batch_first=True, x is (batch_size, seq_len, d_model).<-#
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask, key_padding_mask=key_padding_mask)
        #-> Add attention output (with residual and dropout) and normalize.<-#
        x = residual + self.dropout1(attn_output)
        x = self.norm1(x)

        #-> Residual connection before feed-forward.<-#
        residual = x
        #-> Feed-forward network steps with activation and dropout.<-#
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        #-> Add feed-forward output (with residual and dropout) and normalize.<-#
        x = residual + self.dropout2(ff_output)
        x = self.norm2(x)

        #-> Return the final output of the block.<-#
        return x


#-> This is the main Transformer model class. It seems to be a decoder-only model, often used for language generation.<-#
class TRANSFORMER(nn.Module):
    #-> Constructor for the whole model. Takes vocabulary size, hidden dimension, number of layers, heads, etc.<-#
    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, max_len=1024, dropout=0.1):
        #-> Call the parent constructor.<-#
        super().__init__()
        #-> An embedding layer to convert input token IDs (integers) into dense vectors of size hidden_size.<-#
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        #-> This is interesting! It uses nn.Embedding for positional encoding instead of the sine/cosine PositionalEncoding class defined earlier. This means the positional embeddings are *learned* during training.<-#
        self.pos_embedding = nn.Embedding(max_len, hidden_size)


        #-> A list of decoder blocks. The model is essentially a stack of these identical blocks.<-#
        self.decoder_blocks = nn.ModuleList([
            #-> Create num_layers instances of the DecoderBlock. The feedforward dimension is set to 4 times the hidden size, which is common.<-#
            DecoderBlock(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=dropout)
            for _ in range(num_layers)
        ])

        #-> A final layer normalization after the stack of decoder blocks.<-#
        self.norm = nn.LayerNorm(hidden_size)
        #-> The final linear layer (the "language model head") projects the output of the decoder stack back to the vocabulary size. The output logits represent the probability distribution over the next token.<-#
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        #-> Store some parameters for later use.<-#
        self.max_len = max_len
        self.hidden_size = hidden_size

        #-> Apply custom weight initialization to the model's layers.<-#
        self.apply(self._init_weights)

    #-> A helper function to initialize the weights of different layer types. This can be important for stable training.<-#
    def _init_weights(self, module):
        #-> If the module is a linear layer...<-#
        if isinstance(module, nn.Linear):
            #-> Initialize weights from a normal distribution with a small standard deviation.<-#
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #-> If there's a bias, initialize it to zeros.<-#
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        #-> If the module is an embedding layer...<-#
        elif isinstance(module, nn.Embedding):
             #-> Initialize weights from a normal distribution.<-#
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #-> If the module is a layer norm...<-#
        elif isinstance(module, nn.LayerNorm):
            #-> Initialize bias to zeros and weight to ones.<-#
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    #-> The forward pass for the entire Transformer model. Takes input token IDs and an optional padding mask.<-#
    def forward(self, input_ids, padding_mask=None):
        #-> Get batch size and sequence length from the input shape.<-#
        batch_size, seq_len = input_ids.shape

        #-> Check if the input sequence length exceeds the maximum allowed length.<-#
        if seq_len > self.max_len:
             raise ValueError(f"Sequence length ({seq_len}) exceeds max_len ({self.max_len})")


        #-> Get token embeddings for the input IDs. Shape: (batch_size, seq_len, hidden_size).<-#
        token_embed = self.token_embedding(input_ids)
        #-> Create a tensor of positions from 0 to seq_len-1.<-#
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        #-> Get learned positional embeddings for these positions. Unsqueeze adds a batch dimension so it can be broadcast. Shape: (1, seq_len, hidden_size).<-#
        pos_embed = self.pos_embedding(positions).unsqueeze(0)
        #-> Combine token and positional embeddings by simple addition. This is the input to the first decoder block. Shape: (batch_size, seq_len, hidden_size).<-#
        x = token_embed + pos_embed


        #-> Create a causal mask. This is crucial for decoder-only models doing language modeling. It prevents the model from seeing future tokens when predicting the current token.<-#
        causal_mask = torch.full(
            (seq_len, seq_len), float('-inf'), device=input_ids.device, dtype=torch.float
        )
        #-> Fill the upper triangle of the matrix (excluding the diagonal) with -inf. When softmax is applied, exp(-inf) becomes 0, effectively masking these positions.<-#
        causal_mask = torch.triu(causal_mask, diagonal=1) # Mask upper triangle

        #-> This comment block doesn't actually do anything, but it seems intended to handle the padding mask logic.<-#
        if padding_mask is None:

             pass #-> If no padding mask is provided, just continue.<-#
        #-> Note: The padding mask is passed directly to the decoder block's self-attention. MultiheadAttention handles combining the causal mask and padding mask internally when both are provided.<-#


        #-> Loop through each decoder block in the stack.<-#
        for block in self.decoder_blocks:
             #-> Pass the output of the previous block (or the initial embeddings) through the current block, applying the causal mask and padding mask.<-#
             x = block(x, mask=causal_mask, key_padding_mask=padding_mask)


        #-> Apply the final layer normalization after the last decoder block.<-#
        x = self.norm(x)


        #-> Pass the normalized output through the language model head to get the logits (raw scores) for each token in the vocabulary for each position.<-#
        logits = self.lm_head(x)

        #-> Return the logits. These can then be used with a loss function (like CrossEntropyLoss) or softmax to get probabilities.<-#
        return logits