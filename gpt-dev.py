# Notes from Let's Build GPT video

import torch
import torch.nn as nn
from torch.nn import functional as F


# Read txt doc to inspect it
with open('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#print("length of dataset in characters: ", len(text))

# Inspect the first 1000
#print(text[:1000])

# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# #print(''.join(chars))
# #print(vocab_size)

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #encoder: take a string and outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: takes a list of integers and outputs a string

# #print(encode("hello world"))
# #print(decode(encode("hello world")))

data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000]) # Prints first 1000 chars in the tensor

# Split up data into train and validation sets
n = int(0.9*len(data)) # first 90% will be use to train, rest for validation
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

#print(train_data)

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    #print(f"when input is {context} the target : {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
#print('inputs:')
#print(xb.shape)
#print(xb)
#print('targets:')
#print(yb.shape)
#print(yb)

#print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        #print(f"when input is {context.tolist()} the target: {target}")


torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
#print(logits.shape)
#print(loss)

#print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# Create PyTorch Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):

    # sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the Loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

# Consider the following example
torch.manual_seed(1337)
B,T,C = 4,8,2 #Batch, Time, Channels
x = torch.randn(B,T,C)
x.shape

print(x.shape)

# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C)) # bag of words = xbow
for b in range(B): # iterate over batches
    for t in range(T): # iterate over time
        xprev = x[b, :t+1] # (t, C)
        xbow[b,t] = torch.mean(xprev, 0)

# Version 2
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (T, T) @ (B, T, C) ----> (B, T, C)
# now xbow and xbow2 dims are the same
torch.allclose(xbow, xbow2)

# Version 3 - Uses Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T)) # think of this like an interaction strength or affinity
wei = wei.masked_fill(tril == 0, float('-inf')) # makes all zeros negative infinity
wei = F.softmax(wei, dim=1) # take softmax of every row
xbow3 = wei @ x # normalizes and sum
torch.allclose(xbow, xbow3)

# Another Example
torch.manual_seed(42)
a = torch.ones(3, 3)
a = a / torch.sum(a, 1, keepdim=True) # now gives you average of elements in each row
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('b=')
print(b)
print('c=')
print(c)

# Version 4: self-attention
torch.manual_seed(1337)
B,T,C = 4,8,32 # Batch, Time, Channels
x = torch.randn(B,T,C)

# Lets see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B, T, 16) (B, T, head_size)
q = query(x) # (B, T, 16) (B, T, head_size)
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v # v is the vector we aggragated instead of x
#out = wei @ x

print(out.shape)
print(tril)
print(wei)

# Self attention
k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5
print(k.var())
print(q.var())
print(wei.var())
print(torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])*8, dim=-1))

# From MakeMore Series
# Layer Normalization
class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        # self.momentum = momentum
        # self.training = True
        # Parameters (trained with backpropagation)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # # Buffers (trained with a running "momentum update")
        # self.running_mean = torch.zeros(dim)
        # self.running_var = torch.ones(dim)

    def __call__(self, x):
        # calculate the forward pass
        # if self.training:
        #     xmean = x.mean(1, keepdim=True) # Batch mean
        #     xvar = x.var(1, keepdim=True) # Batch variance
        # else:
        #     xmean = self.running_mean
        #     xvar = self.running_var
        # xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        # self.out = self.gamma * xhat + self.beta
        #update the buffers
        # if self.training:
        #     with torch.no_grad():
        #         self.running_mean = (1- self.momentum) * self.running_mean + self.momentum * xmean
        #         self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        #     return self.out

        # calculate the forward pass
        xmean = x.mean(1, keepdim=True)  # Batch mean
        xvar = x.var(1, keepdim=True)  # Batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        return self.out


    def parameters(self):
        return [self.gamma, self.beta]

torch.manual_seed(1337)
module= BatchNorm1d(100)
x = torch.randn(32, 100) # batch size 32 of 100 dimensional vectors
x = module(x)
x.shape
print(torch.Size([32, 100]))

# mean, std of one feature across all batch inputs
print(x[:, 0].mean(), x[:, 0].std())

# mean, std of a single input from the batch
print(x[0, :].mean(), x[0, :].std())