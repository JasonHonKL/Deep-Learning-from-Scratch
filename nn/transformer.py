import numpy as np 

class Embedding:
    def __init__(self , vocab_size , embedding_dim):
        # this would create a dimension of weight matrix (dot) vocab_size 
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.normal(0 , 0.01 , (vocab_size , embedding_dim))

    def forward(self ,x):
        """
         here n is a 2 dimension array
         consider the embedding matrix as a look up table

         The dimension would be N x sequence length --> N x sequence length x embedding dim

        """
        return self.embedding_matrix[x]

    def backward(self , x , d_out):
        """
            
        """
        d_embedding_matrix = np.zeros_like(self.embedding_matrix)
        np.add.at(d_embedding_matrix, x.flatten(), d_out.reshape(-1, self.embedding_dim))
        return d_embedding_matrix
    

class PositionalEncoding:
    """
        Note that positional encoding should follows after the embedding 
        the embeddiing should return [x , embedding_dim] vector
    """
    def __init__(self,  embedding_dim , max_length):
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.pe = self._compute_positional_encoding()
    
    def _compute_positional_encoding(self):
        positions = np.arange(self.max_length).reshape(-1, 1)
        div_term = np.exp(np.arange(0 , self.embedding_dim , 2) * (-np.log(10000.0)/self.embedding_dim))
        pe = np.zeros((self.max_length , self.embedding_dim))
        pe[: , 0::2] = np.sin(positions * div_term)
        pe[: , 1::2] = np.cos(positions * div_term)

        return pe

    def forward(self , x):
        batch_size , seq_len , _ = x.shape
        return x + self.pe[:seq_len , :]

class MultiHeadSelfAttention:
    def __init__(self , embedding_dim , number_head):
        # number of head need to be divisible
        assert embedding_dim % number_head == 0 
        self.embedding_dim = embedding_dim
        self.num_head = number_head
        # head dimension = total // number of head 
        self.head_dim = embedding_dim // number_head
        
        self.W_q = np.random.normal(0 , 0.01 , (embedding_dim , embedding_dim))
        self.W_k = np.random.normal(0 , 0.01 , (embedding_dim , embedding_dim))
        self.W_v = np.random.normal(0 , 0.01 , (embedding_dim , embedding_dim))
        self.W_o = np.random.normal(0 , 0.01 , (embedding_dim , embedding_dim))

        self.b_q = np.zeros(embedding_dim)
        self.b_k = np.zeros(embedding_dim)
        self.b_v = np.zeros(embedding_dim)
        self.b_o = np.zeros(embedding_dim)

    def forward(self , x , mask=None):
        batch_size , seq_len , _ = x.shape
        
        Q = np.dot(x , self.W_q) + self.b_q
        K = np.dot(x , self.W_k) + self.b_k
        V = np.dot(x , self.W_v) + self.b_v

        Q = Q.reshape(batch_size , seq_len , self.num_head , self.head_dim).transpose(0,2,1,3)
        K = K.reshape(batch_size , seq_len , self.num_head , self.head_dim).transpose(0,2,1,3)
        V = V.reshape(batch_size , seq_len , self.num_head , self.head_dim).transpose(0,2,1,3)

        scores = np.matmul(Q , K.transpose(0,1,3,2))/np.sqrt(self.head_dim)
        #softmax
        if mask is not None:
            scores = scores + mask 

        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)  # (batch_size, num_heads, seq_len, seq_len)
        attention = np.matmul(weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        attention = attention.transpose(0,2,1,3).reshape(batch_size , seq_len , self.embedding_dim)

        output = np.dot(attention , self.W_o) + self.b_o

        self.cache = (x , Q , K ,V , weights , attention)

        return output
    
    def backward(self, d_out):
        batch_size, seq_len, _ = d_out.shape
        x, Q, K, V, weights, attention = self.cache
        
        # Backward through output projection
        d_attention = np.dot(d_out, self.W_o.T)
        d_W_o = np.einsum('bsi,bsj->ij', attention, d_out)
        d_b_o = np.sum(d_out, axis=(0, 1))
        
        # Reshape for heads
        d_attention = d_attention.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Backward through attention
        d_V = np.matmul(weights.transpose(0, 1, 3, 2), d_attention)  # (batch_size, num_heads, seq_len, head_dim)
        d_weights = np.matmul(d_attention, V.transpose(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        
        # Backward through softmax
        weights_grad = weights * (d_weights - np.sum(d_weights * weights, axis=-1, keepdims=True))
        d_scores = weights_grad / np.sqrt(self.head_dim)
        
        # Backward through Q, K
        d_Q = np.matmul(d_scores, K)  # (batch_size, num_heads, seq_len, head_dim)
        d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back
        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embedding_dim)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embedding_dim)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embedding_dim)
        
        # Backward through Q, K, V projections
        d_x = np.dot(d_Q, self.W_q.T) + np.dot(d_K, self.W_k.T) + np.dot(d_V, self.W_v.T)
        d_W_q = np.einsum('bsi,bsj->ij', x, d_Q)
        d_W_k = np.einsum('bsi,bsj->ij', x, d_K)
        d_W_v = np.einsum('bsi,bsj->ij', x, d_V)
        d_b_q = np.sum(d_Q, axis=(0, 1))
        d_b_k = np.sum(d_K, axis=(0, 1))
        d_b_v = np.sum(d_V, axis=(0, 1))
        
        return d_x, (d_W_q, d_W_k, d_W_v, d_W_o, d_b_q, d_b_k, d_b_v, d_b_o)
    
def create_causal_mask(seq_len):
    """Create a causal mask for decoder self-attention.
    
    Args:
        seq_len (int): Length of the sequence.
    
    Returns:
        np.ndarray: Mask of shape (1, 1, seq_len, seq_len).
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
    return mask[None, None, :, :]  

