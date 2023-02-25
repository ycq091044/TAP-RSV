import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TensorGRU(nn.Module):
    def __init__(self, hidden=8, test_length=4):
        super(TensorGRU, self).__init__()
        # static features
        self.county_linear = nn.Linear(14, hidden)
        self.covid_linear = nn.Linear(91, hidden)
        self.mobility_linear = nn.Linear(2334, hidden)
        self.distance_linear = nn.Linear(2334, hidden)
        self.vac_time_linear = nn.Linear(91, hidden)
        self.vac_type_linear = nn.Linear(34, 1)
        self.hos_time_linear = nn.Linear(91, hidden)
        self.hos_type_linear = nn.Linear(4, 1)
        self.claim_time_linear = nn.Linear(91, hidden)
        self.claim_type_linear = nn.Linear(20, 1)
        
        self.static = nn.Linear(7*hidden, hidden)
        
        self.county_tensor_emb_linear = nn.Linear(3, hidden)
        self.month_table = nn.Embedding(13, hidden)
        
        # for disease similarity attention
        self.Kw = nn.Linear(8, hidden)
        self.Qw = nn.Linear(8, hidden)
        
        self.rnn = nn.GRU(19*2, hidden, 2, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, test_length),
        )
        self.init()
        
    @staticmethod
    def get_last_visit(hidden_states, mask):
        last_visit = torch.sum(mask,1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state

    def forward(self, county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X, month, disease_emb, mask):
        """ get static features """
        county_feature = self.county_linear(county)
        covid_feature = self.covid_linear(covid)
        mob_feature = self.mobility_linear(mob)
        dist_feature = self.distance_linear(distance)
        vac_feature = self.vac_type_linear(vac).squeeze(-1)
        vac_feature = self.vac_time_linear(vac_feature)
        hos_feature = self.hos_type_linear(hos).squeeze(-1)
        hos_feature = self.hos_time_linear(hos_feature)
        claim_feature = self.claim_type_linear(claim).squeeze(-1)
        claim_feature = self.claim_time_linear(claim_feature)
        
        static_features = torch.concat([county_feature, covid_feature, mob_feature, dist_feature, \
                                  vac_feature, hos_feature, claim_feature], 1)
        static_emb = self.static(static_features)
        
        """ get location and month features """
        county_tensor_feature = self.county_tensor_emb_linear(county_tensor_emb)
        month_feature = self.month_table(month)
        
        """ dynamic features """
        K = self.Kw(disease_emb)
        Q = self.Qw(disease_emb)
        adj = torch.softmax(K @ Q.T, dim=-1)
        gru_output, _ = self.rnn(torch.concat([torch.einsum("ikj,jr->ikr", X, adj), X], -1))
        dynamic_emb = self.get_last_visit(gru_output, mask)
        
        """ final """
        final = self.fc(torch.concat([dynamic_emb, static_emb, county_tensor_feature, month_feature], 1))
        # final = torch.maximum(final, torch.ones_like(final) * 0)
        # final = torch.minimum(final, torch.ones_like(final) * 200)
        return torch.exp(final)
    
    def init(self):
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                m.bias.data.uniform_(-0.1, 0.1)
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-0.1, 0.1)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)


class LSTM(nn.Module):
    def __init__(self, hidden=8, test_length=4):
        super(LSTM, self).__init__()
        # static features
        self.county_linear = nn.Linear(14, hidden)
        self.covid_linear = nn.Linear(91, hidden)
        self.mobility_linear = nn.Linear(2334, hidden)
        self.distance_linear = nn.Linear(2334, hidden)
        self.vac_time_linear = nn.Linear(91, hidden)
        self.vac_type_linear = nn.Linear(34, 1)
        self.hos_time_linear = nn.Linear(91, hidden)
        self.hos_type_linear = nn.Linear(4, 1)
        self.claim_time_linear = nn.Linear(91, hidden)
        self.claim_type_linear = nn.Linear(20, 1)
        
        self.static = nn.Linear(7*hidden, hidden)
        
        self.rnn = nn.LSTM(19, hidden, 2, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(2*hidden, test_length),
        )
        self.init()
        
    @staticmethod
    def get_last_visit(hidden_states, mask):
        last_visit = torch.sum(mask,1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state

    def forward(self, county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X, month, disease_emb, mask):
        """ get static features """
        county_feature = self.county_linear(county)
        covid_feature = self.covid_linear(covid)
        mob_feature = self.mobility_linear(mob)
        dist_feature = self.distance_linear(distance)
        vac_feature = self.vac_type_linear(vac).squeeze(-1)
        vac_feature = self.vac_time_linear(vac_feature)
        hos_feature = self.hos_type_linear(hos).squeeze(-1)
        hos_feature = self.hos_time_linear(hos_feature)
        claim_feature = self.claim_type_linear(claim).squeeze(-1)
        claim_feature = self.claim_time_linear(claim_feature)
        
        static_features = torch.concat([county_feature, covid_feature, mob_feature, dist_feature, \
                                  vac_feature, hos_feature, claim_feature], 1)
        static_emb = self.static(static_features)
        
        gru_output, _ = self.rnn(X)
        lstm_emb = self.get_last_visit(gru_output, mask)
        
        """ final """
        final = self.fc(torch.concat([lstm_emb, static_emb], 1))
        return final
    
    def init(self):
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                m.bias.data.uniform_(-0.1, 0.1)
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-0.1, 0.1)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)


class Transformer(nn.Module):
    def __init__(self, hidden=16, test_length=4):
        super(Transformer, self).__init__()
        # static features
        self.county_linear = nn.Linear(14, hidden)
        self.covid_linear = nn.Linear(91, hidden)
        self.mobility_linear = nn.Linear(2334, hidden)
        self.distance_linear = nn.Linear(2334, hidden)
        self.vac_time_linear = nn.Linear(91, hidden)
        self.vac_type_linear = nn.Linear(34, 1)
        self.hos_time_linear = nn.Linear(91, hidden)
        self.hos_type_linear = nn.Linear(4, 1)
        self.claim_time_linear = nn.Linear(91, hidden)
        self.claim_type_linear = nn.Linear(20, 1)
        
        self.static = nn.Linear(7*hidden, hidden)
        
        # transformer requires d_model to be divisible by nhead
        self.feature_transform = nn.Linear(19, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=16, batch_first=True)
        self.rnn = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)
        
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(2*hidden, test_length),
        )
        self.init()
        
    @staticmethod
    def get_last_visit(hidden_states, mask):
        last_visit = torch.sum(mask,1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state

    def forward(self, county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X, month, disease_emb, mask):
        """ get static features """
        county_feature = self.county_linear(county)
        covid_feature = self.covid_linear(covid)
        mob_feature = self.mobility_linear(mob)
        dist_feature = self.distance_linear(distance)
        vac_feature = self.vac_type_linear(vac).squeeze(-1)
        vac_feature = self.vac_time_linear(vac_feature)
        hos_feature = self.hos_type_linear(hos).squeeze(-1)
        hos_feature = self.hos_time_linear(hos_feature)
        claim_feature = self.claim_type_linear(claim).squeeze(-1)
        claim_feature = self.claim_time_linear(claim_feature)
        
        static_features = torch.concat([county_feature, covid_feature, mob_feature, dist_feature, \
                                  vac_feature, hos_feature, claim_feature], 1)
        static_emb = self.static(static_features)
        
        gru_output = self.rnn(self.feature_transform(X))
        lstm_emb = self.get_last_visit(gru_output, mask)
        
        """ final """
        final = self.fc(torch.concat([lstm_emb, static_emb], 1))
        return final
    
    def init(self):
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                m.bias.data.uniform_(-0.1, 0.1)
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-0.1, 0.1)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)



class HOIST(nn.Module):
    def __init__(self, dynamic_dims, static_dims = None, distance_dims = None, rnn_dim=128, hidden=16, signs=None, test_length=4):
        """The HOIST Model
        Args:
            dynamic_dims: List of integers (Number of features in each dynamic feature category, e.g., vaccination, hospitalization, etc.).
            static_dims (Optional): List of integers (Number of features in each static feature category, e.g., demographic, economic, etc.). If None, no static features are used.
            distance_dims (Optional): Interger (Number of distance types, e.g., geographical, mobility, etc.). If None, no distance features are used.
            rnn_dim: Integer (Number of hidden units in the RNN layer).
            signs: List of 1 or -1 (Field direction of each dynamic feature category, e.g., -1 for vaccination, +1 for hospitalization, etc.). If None, all signs are positive.
            
        Inputs:
            dynamic: List of FloatTensor with shape (N, T, D_k) (Dynamic features). D_k is the number of features in the k-th category and it should be the same as the k-th dimension in dynamic_dims.
            static (Optional): List of FloatTensor with shape (N, D_k) (Static features). D_k is the number of features in the k-th category and it should be the same as the k-th dimension in static_dims.
            distance (Optional): FloatTensor with shape (N, N, D_k) (Distance features). D_k is the number of distance types and it should be the same as the dimension in distance_dims.
            *** if both static and distance is None, the spatial relationships won't be used. ***
            h0 (Optional): FloatTensor with shape (1, N, rnn_dim) (Initial hidden state of the RNN layer). If None, it will be initialized as a random tensor.
        """
        
        super(HOIST, self).__init__()
        self.dynamic_dims = dynamic_dims
        self.dynamic_feats = len(dynamic_dims)
        self.static_dims = static_dims
        self.distance_dims = distance_dims
        self.rnn_dim = rnn_dim
        self.signs = signs
        if self.signs != None:
            try:
                assert len(self.signs) == self.dynamic_feats
                assert all([s == 1 or s == -1 for s in self.signs])
            except:
                raise ValueError('The signs should be a list of 1 or -1 with the same length as dynamic_dims.')
        
        self.dynamic_weights = nn.ModuleList([nn.Sequential(nn.Linear(self.dynamic_dims[i], rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, self.dynamic_dims[i]), nn.Sigmoid()) for i in range(self.dynamic_feats)])
        
        self.total_feats = np.sum(self.dynamic_dims)       
        self.rnn = nn.LSTM(self.total_feats, rnn_dim, batch_first=True)
        
        
        self.county_linear = nn.Linear(14, hidden)
        self.covid_linear = nn.Linear(91, hidden)
        self.vac_time_linear = nn.Linear(91, hidden)
        self.vac_type_linear = nn.Linear(34, 1)
        self.hos_time_linear = nn.Linear(91, hidden)
        self.hos_type_linear = nn.Linear(4, 1)
        self.claim_time_linear = nn.Linear(91, hidden)
        self.claim_type_linear = nn.Linear(20, 1)        
        self.county_tensor_emb_linear = nn.Linear(3, hidden)
        self.month_table = nn.Embedding(13, hidden)
        self.static = nn.Sequential(
            nn.GELU(),
            nn.Linear(7 * hidden, hidden),
        )
        
        self.linear = nn.Linear(rnn_dim+hidden, rnn_dim)
        self.linear_2 = nn.Linear(rnn_dim, test_length)
        
        self.static_dims = static_dims
        if self.static_dims != None:
            self.static_feats = len(static_dims)
    
            self.w_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.static_dims[i], self.static_dims[i]).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True) for i in range(self.static_feats)])
            self.a_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2*self.static_dims[i], 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True) for i in range(self.static_feats)])

        if self.distance_dims != None:
            self.W_dis = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(distance_dims, distance_dims).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
            self.a_dis = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(distance_dims, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
    
    @staticmethod
    def get_last_visit(hidden_states, mask):
        last_visit = torch.sum(mask,1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state
    
    def forward(self, dynamic, mask, static = None, distance = None, h0 = None):
        county, covid, vac, hos, claim, county_tensor_emb, month = static
                
        county_feature = self.county_linear(county)
        covid_feature = self.covid_linear(covid)
        vac_feature = self.vac_type_linear(vac).squeeze(-1)
        vac_feature = self.vac_time_linear(vac_feature)
        hos_feature = self.hos_type_linear(hos).squeeze(-1)
        hos_feature = self.hos_time_linear(hos_feature)
        claim_feature = self.claim_type_linear(claim).squeeze(-1)
        claim_feature = self.claim_time_linear(claim_feature)
        
        county_tensor_feature = self.county_tensor_emb_linear(county_tensor_emb)
        month_feature = self.month_table(month)
        static_features = torch.concat([county_feature, covid_feature, \
                                  vac_feature, hos_feature, claim_feature, county_tensor_feature, month_feature], 1)
        static_emb = self.static(static_features)
        
        static = [county_feature, covid_feature, vac_feature, hos_feature, claim_feature, county_tensor_feature, month_feature]
        
        static_dis = []
        N = dynamic[0].shape[0]
        T = dynamic[0].shape[1]
        if self.static_dims != None:
            for i in range(self.static_feats):
                h_i = torch.mm(static[i], self.w_list[i])
                h_i = torch.cat([h_i.unsqueeze(1).repeat(1, N, 1), h_i.unsqueeze(0).repeat(N, 1, 1)], dim=2)
                d_i = torch.sigmoid(h_i @ self.a_list[i]).reshape(N, N)
                static_dis.append(d_i)

        if self.distance_dims != None:
            print(distance.shape)
            h_i = distance @ self.W_dis
            print(h_i.shape)
            h_i = torch.sigmoid(h_i @ self.a_dis).reshape(N, N)
            static_dis.append(h_i)
            
        if self.static_dims != None or self.distance_dims != None:
            static_dis = torch.stack(static_dis, dim=0)
            static_dis = static_dis.sum(0)
            static_dis = torch.softmax(static_dis, dim=-1)
        
 
        cur_weight = self.dynamic_weights[0](dynamic[0].reshape(N*T, -1)).reshape(N, T, -1)
        if self.signs != None:
            cur_weight = cur_weight * self.signs[0]
        dynamic_weights = cur_weight

        if h0 is None:
            h0 = torch.randn(1, N, self.rnn_dim).to(cur_weight.device)
        dynamic = torch.cat(dynamic, dim=-1)
        h, hn = self.rnn(dynamic_weights*dynamic)
        
        if self.static_dims != None or self.distance_dims != None:
            h_att = h.reshape(N,1,T,self.rnn_dim).repeat(1,N,1,1)
            h = h + (h_att * static_dis.reshape(N,N,1,1)).sum(1)
        h = self.get_last_visit(h, mask)
        h = torch.concat([h, static_emb], 1)
        y = self.linear(h)
        y = self.linear_2(F.leaky_relu(y))
        return y, [static_dis, dynamic_weights, hn]
    
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'     


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input, adj):
        h = torch.mm(input, self.W)

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(0,1))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
     
class STAN(nn.Module):
    def __init__(self, gat_hidden, hidden, static_dim, test_length=4):
        super(STAN, self).__init__()
 
        self.layer1 = GATLayer(19, gat_hidden)
        self.layer2 = GATLayer(gat_hidden, gat_hidden)
        
        self.county_linear = nn.Linear(14, hidden)
        self.covid_linear = nn.Linear(91, hidden)
        self.mobility_linear = nn.Linear(2334, hidden)
        self.distance_linear = nn.Linear(2334, hidden)
        self.vac_time_linear = nn.Linear(91, hidden)
        self.vac_type_linear = nn.Linear(34, 1)
        self.hos_time_linear = nn.Linear(91, hidden)
        self.hos_type_linear = nn.Linear(4, 1)
        self.claim_time_linear = nn.Linear(91, hidden)
        self.claim_type_linear = nn.Linear(20, 1)
        self.static = nn.Sequential(
            nn.GELU(),
            nn.Linear(7 * hidden, static_dim),
        )
        
        self.county_tensor_emb_linear = nn.Linear(3, hidden)
        self.month_table = nn.Embedding(13, hidden)

                
        self.rnn = nn.GRU(gat_hidden, gat_hidden, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(gat_hidden+static_dim, test_length),
        )
    
    @staticmethod
    def get_last_visit(hidden_states, mask):
        last_visit = torch.sum(mask,1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state

    def forward(self, county, covid, vac, hos, claim, county_tensor_emb, X, month, mask, adj):
        """ get static features """
        county_feature = self.county_linear(county)
        covid_feature = self.covid_linear(covid)
        vac_feature = self.vac_type_linear(vac).squeeze(-1)
        vac_feature = self.vac_time_linear(vac_feature)
        hos_feature = self.hos_type_linear(hos).squeeze(-1)
        hos_feature = self.hos_time_linear(hos_feature)
        claim_feature = self.claim_type_linear(claim).squeeze(-1)
        claim_feature = self.claim_time_linear(claim_feature)
        
        county_tensor_feature = self.county_tensor_emb_linear(county_tensor_emb)
        month_feature = self.month_table(month)
        static_features = torch.concat([county_feature, covid_feature,  \
                                  vac_feature, hos_feature, claim_feature, county_tensor_feature, month_feature], 1)
        
        static_emb = self.static(static_features) #batch_size, hidden

        rep = []
        for i in range(X.shape[1]):
            cur_rep = self.layer1(X[:, i], adj)
            rep.append(cur_rep)
        rep = torch.stack(rep, 1)
        rep = self.rnn(rep)[0]
        rep = self.get_last_visit(rep, mask)
        rep = torch.concat([rep, static_emb], -1)
        rep = self.fc(rep)
        return rep
    