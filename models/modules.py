import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
import copy
import dill

# ==================================================================
# LTE module
class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        """
        # ==============================
        # **Log time encoding function**:
        timestamps = torch.abs(timestamps)
        timestamps = torch.log(1 + timestamps)
        # ==============================

        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output

# ==================================================================
# TRC module

class HistEmbAggregatorWeightedSum(nn.Module):
    """
    Equation 4 in the paper: update historical edge embedding r^t_uv.
    """

    def __init__(self, gamma=0.9):
        super().__init__()

        self.aggregator_name = 'weighted_sum'
        self.gamma = gamma

        print(f'[system] hyperparameter gamma value: {self.gamma}.')
    
    def forward(self, current_emb, hist_emb):
        """
        update historical edge embedding r^t_uv.
        :param: current_emb: shape (batch_size, dim)
        :param: hist_emb: shape (batch_size, dim)
        """
        weighted_sum = self.gamma * current_emb + (1 - self.gamma) * hist_emb

        return weighted_sum


class HistoricalDecoder(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int, device='cpu', gamma=0.9):
        """
        HistoricalDecoder to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim*2, output_dim)
        self.act = nn.ReLU()

        self.device = device

        # =================================================
        # function to update TRC historical edge embedding:
        self.aggregate_hist_emb = HistEmbAggregatorWeightedSum(gamma=gamma)
        self.hist_emb_aggregator = self.aggregate_hist_emb.aggregator_name

        # TRC memory:
        self.historical_interaction_memory = TRCMemory(dim=input_dim1, device=device)

    def forward(self, src_ids, dst_ids, src_emb: torch.Tensor, dst_emb: torch.Tensor, update_memories=False):
        """
        merge and project the inputs
        :param src_ids: Tensor, shape (batch_size, 1)
        :param dst_ids: Tensor, shape (batch_size, 1)
        :param src_emb: Tensor, shape (batch_size, hidden_dim)
        :param dst_emb: Tensor, shape (batch_size, hidden_dim)
        """
        # ===========================================
        # 0. combine the embeddings of source node and destination node:
        x = torch.cat([src_emb, dst_emb], dim=1)    # Tensor, shape (*, input_dim1 + input_dim2)
        h_current = self.act(self.fc1(x))           # Tensor, shape (*, hidden_dim)
        # ===========================================

        # obtain (u, v) links:
        keys = list(zip(src_ids, dst_ids))          # shape: (batch_size, 1), each key is a tuple, denoting an edge.
        # get dedicated historical edge embeddings from TRC memory:
        hist_emb_most_recent_one = self.historical_interaction_memory.get_memories(keys)       
        
        # =============================================
        # 1. compute updated TRC historical edge embedding using equation 4.
        h_most_recent_emb = torch.stack(hist_emb_most_recent_one, dim=0).to(self.device)    # shape: (batch_size, hidden_dim) 

        h_historical = self.aggregate_hist_emb(current_emb=h_current, hist_emb=h_most_recent_emb)

        if self.hist_emb_aggregator == 'weighted_sum':
            updated_weighted_hist_emb = h_historical
            h_historical = h_most_recent_emb
        else:
            raise Exception('Invalid TRC aggregator')

        # 2. combine temporal node embeddings and the most recent TRC historical edge embedding:
        h = self.fc2(torch.cat((h_current, h_historical), dim=1))

        # =============================================
        # 3. update memory:
        if update_memories:
            if self.hist_emb_aggregator == 'weighted_sum':
                self.historical_interaction_memory.update_memories(keys, updated_weighted_hist_emb.detach().cpu())
            else:
                raise Exception('Invalid TRC aggregator when updating TRC memory')
        # =============================================
        return h
    
class TRCMemory(nn.Module):
    def __init__(self, dim=None, device=None):
        """
        TRC Memory, store historical edge embeddings.
        :param dim: int, embedding of stored historical edge embedding.
        """
        super().__init__()

        self.most_recent_hist_emb = dict()

        self.device = device
        self.PAD_ZERO = torch.zeros(dim, device=device, requires_grad=False)
    
    def update_memories(self, keys, hist_emb):
        """
        update memory
        :param keys: shape (batch_size, 1), each key is an edge, for instance, the first key might be an edge (a,b)
        :param hist_emb: shape (batch_size, hidden_dim)
        """
        for i, k in enumerate(keys):
            self.most_recent_hist_emb[k] = hist_emb[i]

    def get_memories(self, keys):
        """
        get TRC historical edge embeddings of required links
        :param keys: shape (batch_size, 1) each key is an edge, for instance, the first key might be an edge (a,b)
        """
        most_recent_hist_emb = [self.PAD_ZERO if k not in self.most_recent_hist_emb.keys() else self.most_recent_hist_emb[k].to(self.device) for k in keys]       # (batch_size, dim)

        return most_recent_hist_emb

    def reset_memory(self):
        """
        Reset memory. 
        This should be called at the beginning of each training epoch
        """
        self.most_recent_hist_emb.clear()


    def save_memory(self, path):
        """
        Save TRC memory to the given path . 
        """
        save_path = path
        
        with open(save_path + '_save_most_recent_emb.pkl', 'wb') as f:
            dill.dump(self.most_recent_hist_emb, f)
        
    
    def load_memory(self, path):
        """
        Load TRC memory to the given path . 
        """
        save_path = path
        
        with open(save_path + '_save_most_recent_emb.pkl', 'rb') as f:
            self.most_recent_hist_emb = dill.load(f)
    

    def backup_memory_bank(self):
        """
        Backup TRC memory during the training stage
        """
        backup = {
            'most_recent_emb': copy.deepcopy(self.most_recent_hist_emb),
        }

        return backup
    
    def load_memory_bank(self, bank):
        """
        Load TRC memory during the training stage.
        """
        self.reset_memory()

        self.most_recent_hist_emb = bank['most_recent_emb']

# TRC module ends here
## --------------------------------------------------------------------------------------

class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Multi-Layer Perceptron Classifier.
        :param input_dim: int, dimension of input
        :param dropout: float, dropout rate
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        multi-layer perceptron classifier forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        # Tensor, shape (*, 80)
        x = self.dropout(self.act(self.fc1(x)))
        # Tensor, shape (*, 10)
        x = self.dropout(self.act(self.fc2(x)))
        # Tensor, shape (*, 1)
        return self.fc3(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)

        self.scaling_factor = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor, neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor, neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs_query: torch.Tensor, inputs_key: torch.Tensor = None, inputs_value: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        encode the inputs by Transformer encoder
        :param inputs_query: Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        :param inputs_key: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param inputs_value: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param neighbor_masks: ndarray, shape (batch_size, source_seq_length), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # transposed_inputs_query, Tensor, shape (target_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_key, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_value, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        transposed_inputs_query, transposed_inputs_key, transposed_inputs_value = inputs_query.transpose(0, 1), inputs_key.transpose(0, 1), inputs_value.transpose(0, 1)

        if neighbor_masks is not None:
            # Tensor, shape (batch_size, source_seq_length)
            neighbor_masks = torch.from_numpy(neighbor_masks).to(inputs_query.device) == 0

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs_query, key=transposed_inputs_key,
                                                  value=transposed_inputs_value, key_padding_mask=neighbor_masks)[0].transpose(0, 1)
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs
