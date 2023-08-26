import math

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import pipeline


class ImageEncoder(nn.Module):
    def __init__(self, fine_tune=False):
        super().__init__()
        # Img feature extraction
        resnet = models.resnet50(pretrained=True)  ##TODO: optional with resnet18, resnet101
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False

        # # Fine tune resnet
        if fine_tune:
            for c in list(self.resnet.children())[6:]:
                for p in c.parameterss():
                    p.requires_grad = True

    def forward(self, images):
        img_embeddings = self.resnet(images)
        size = img_embeddings.size()
        out = img_embeddings.view(*size[:2], -1)
        ##TODO: Return img_embeddings imediately
        return out.view(*size).contiguous()  # batch_size, 2048, img_size/32, img_size/32


class TemporalEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.temporal_fusion = nn.Linear(embedding_dim*4, embedding_dim)
        self.dropout= nn.Dropout(0.2)

    def forward(self, temporal_features):
        # Temporal dummy variables (day, week, month, year)
        d, w, m, y = temporal_features[:, 0].unsqueeze(1), temporal_features[:, 1].unsqueeze(1), \
            temporal_features[:, 2].unsqueeze(1), temporal_features[:, 3].unsqueeze(1)
        d_emb, w_emb, m_emb, y_emb = self.day_embedding(d), self.week_embedding(w), self.month_embedding(m), self.year_embedding(y)
        temporal_embeddings = self.temporal_fusion(torch.cat([d_emb, w_emb, m_emb, y_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings


# class AttributeEncoder(nn.Module):
#     def __init__(self, embedding_dim, cat_dict, col_dict, fab_dict, gpu_num):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.cat_dict = {v: k for k, v in cat_dict.items()}
#         self.col_dict = {v: k for k, v in col_dict.items()}
#         self.fab_dict = {v: k for k, v in fab_dict.items()}
#         self.word_embedder = pipeline("feature-extraction", model="bert-base-uncased")
#         self.fc = nn.Linear(768, embedding_dim)
#         self.dropout = nn.Dropout(0.1)
#         self.gpu_num = gpu_num
        
#     def forward(self, categories, colors, fabrics):
#         textual_description = [self.col_dict[colors.detach().cpu().numpy().tolist()[i]] + ' ' \
#                             + self.fab_dict[fabrics.detach().cpu().numpy().tolist()[i]] + ' ' \
#                             + self.cat_dict[categories.detach().cpu().numpy().tolist()[i]] for i in range(len(categories))]

#         # Use BERT to extract features
#         word_embeddings = self.word_embedder(textual_description)

#         # BERT gives us embedding for [CLS] .. [EOS], which is why we inly average the embeddings in the range [1:-1]
#         # We're not fine tuning BERT and we dont wnat the noise coming from [CLS] or [EOS]
#         word_embeddings = [torch.FloatTensor(x[0][1:-1]).mean(axis=0) for x in word_embeddings]
#         word_embeddings = torch.stack(word_embeddings).to('cuda:'+str(self.gpu_num))

#         # Embed to our embedding space
#         word_embeddings = self.dropout(self.fc(word_embeddings))

#         return word_embeddings


class AttributeEncoder(nn.Module):
    def __init__(self, num_cat, num_col, num_fab, embedding_dim):
        super(AttributeEncoder, self).__init__()
        self.cat_embedder = nn.Embedding(num_cat, embedding_dim)
        self.col_embedder = nn.Embedding(num_col, embedding_dim)
        self.fab_embedder = nn.Embedding(num_fab, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, cat, col, fab):
        cat_emb = self.dropout(self.cat_embedder(cat))
        col_emb = self.dropout(self.col_embedder(col))
        fab_emb = self.dropout(self.fab_embedder(fab))
        attribute_embeddings = cat_emb + col_emb + fab_emb

        return attribute_embeddings


class TSEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(TSEncoder, self).__init__()
        self.ts_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.ts_encoder(x)[0])
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=52):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module  # Can be any layer we wish to apply like Linear, Conv, etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timestamps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class GTrendEncoder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends, gpu_num):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_mask = use_mask
        self.gpu_num = gpu_num

    def _generate_encoder_mask(self, size, forecast_horizon):
        mask = torch.zeros((size, size))
        split = math.gcd(size, forecast_horizon)  ##TODO: ??
        for i in range(0, size, split):
            mask[i:i+split, i:i+split] = 1
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(self, gtrends):
        gtrend_emb = self.input_linear(gtrends.permute(0, 2, 1))
        gtrend_emb = self.pos_embedding(gtrend_emb.permute(1, 0, 2))
        input_mask = self._generate_encoder_mask(gtrend_emb.shape[0], self.forecast_horizon) ##TODO: Compare change to normal mask, use with discount
        
        if self.use_mask == 1:
            gtrend_emb = self.encoder(gtrend_emb, input_mask)
        else:
            gtrend_emb = self.encoder(gtrend_emb)
        return gtrend_emb


class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_img, use_attribute, dropout=0.2):
        super(FusionNetwork, self).__init__()

        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        self.use_img = use_img
        self.use_attribute = use_attribute
        input_dim = embedding_dim + (embedding_dim*use_img) + (embedding_dim*use_attribute)
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, img_encoding, attribute_encoding, temporal_encoding):
        # Fuse static features together
        pooled_img = self.img_pool(img_encoding)
        condensed_img = self.img_linear(pooled_img.flatten(1))

        # Build input
        decoder_inputs = []
        if self.use_img == 1:
            decoder_inputs.append(condensed_img)
        if self.use_attribute == 1:
            decoder_inputs.append(attribute_encoding)
        decoder_inputs.append(temporal_encoding)
        concat_features = torch.cat(decoder_inputs, dim=1)
        
        final = self.feature_fusion(concat_features)

        return final


class AdditiveAttention(nn.Module):  # Bahdanau encoder-decoder attention (Additive attention)
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(AdditiveAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.encoder_linear = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_linear = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.attn_linear = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # h_j and s_i refer to the variable names from the original formula of Bahdanau et al.
        h_j = self.encoder_linear(encoder_out)  # (batch_size, len, attention_dim)
        s_i = self.decoder_linear(decoder_hidden).squeeze(0)  # (batch_size, attention_dim)
        energy = self.attn_linear(self.tanh(h_j + s_i.unsqueeze(1))).squeeze(2)  # (batch_size, len)
        alpha = self.softmax(energy)  # (batch_size, len)
        attention_weighted_encoding = (alpha.unsqueeze(2) * h_j)  # (batch_size, len, encoder_dim)

        return attention_weighted_encoding, alpha