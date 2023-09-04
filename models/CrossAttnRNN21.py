from thop import profile

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fairseq.optim.adafactor import Adafactor

from .modules import ImageEncoder, TemporalEncoder, AttributeEncoder, \
                    GTrendEncoder, FusionNetwork, PositionalEncoding, \
                    TSEncoder, AdditiveAttention


class CrossAttnRNN(pl.LightningModule):
    def __init__(self,
                attention_dim,
                embedding_dim,
                hidden_dim,
                use_img,
                use_trends,
                use_attribute,
                apply_concatenate,
                out_len,
                cat_dict,
                col_dict,
                fab_dict,
                trend_len,
                num_trends,):
        super().__init__()
        self.out_len = out_len 
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.use_img = use_img
        self.use_trends = use_trends
        self.use_attribute = use_attribute
        self.apply_concatenate = apply_concatenate

        # Encoder(s)
        self.trend_encoder = TSEncoder(num_trends, embedding_dim)
        self.image_encoder = ImageEncoder(fine_tune=True)
        self.temporal_encoder = TemporalEncoder(embedding_dim)
        self.attribute_encoder = AttributeEncoder(
            len(cat_dict) + 1,
            len(col_dict) + 1,
            len(fab_dict) + 1,
            embedding_dim
        )
        self.static_feature_encoder = FusionNetwork(embedding_dim, hidden_dim, use_img, use_attribute)
        ##TODO: Replace this by lstm
        self.ts_embedder = nn.GRU(1, embedding_dim, batch_first=True)

        # Attention module
        # self.img_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.trend_self_attention = nn.MultiheadAttention(embedding_dim, num_heads=4, dropout=0.1)
        self.trend_linear = nn.Linear(trend_len * embedding_dim, embedding_dim)
        
        if apply_concatenate:
            input_mul_to_decoder = 2
            if self.use_trends:
                input_mul_to_decoder = 3
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(input_mul_to_decoder*hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim//2, 1)
            )
        else:
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim//2, 1)
            )

        # self.save_hyperparamters()  ##TODO: What this?

    def forward(self, X, y, 
                categories, colors, fabrics, stores,
                temporal_features, gtrends,
                images):
        bs, num_ts_splits, timesteps = X.shape[0], X.shape[1], X.shape[2]
        
        # Encode statuc input data
        img_encoding = self.image_encoder(images)
        gtrend_encoding = self.trend_encoder(gtrends.permute(0, 2, 1))
        temporal_encoding = self.temporal_encoder(temporal_features)
        attribute_encoding = self.attribute_encoder(categories, colors, fabrics)

        if self.use_trends:
            # Self-attention over the temporal features (this filters initial noise from the time series features)
            gtrend_encoding, trend_self_attn_weights = self.trend_self_attention(
                gtrend_encoding.permute(1, 0, 2),
                gtrend_encoding.permute(1, 0, 2),
                gtrend_encoding.permute(1, 0, 2),
            )
            gtrend_encoding = gtrend_encoding.permute(1, 0, 2).reshape(bs, -1)
            gtrend_encoding = gtrend_encoding.repeat_interleave(num_ts_splits, dim=0)
            gtrend_encoding = self.trend_linear(gtrend_encoding)
        
        # Fuse static features together
        static_feature_fusion = self.static_feature_encoder(img_encoding, attribute_encoding, temporal_encoding)

        ts_input = X.reshape((bs*num_ts_splits, timesteps)).unsqueeze(-1)  # Collapse values to make 2-1 (single) predictions for each 2 step embedding
        ts_features = self.ts_embedder(ts_input)[1]

        x = ts_features.squeeze(0)
        # Add static features
        static_feature_fusion = static_feature_fusion.repeat_interleave(num_ts_splits, dim=0)
        if self.apply_concatenate:
            x = torch.cat((x, static_feature_fusion), dim=1)
        else:
            x = x + static_feature_fusion

        # Add gtrends
        if self.use_trends:
            if self.apply_concatenate:
                x = torch.cat((x, gtrend_encoding), dim=1)
            else:
                x = x + gtrend_encoding

        outputs = self.decoder(x)
        outputs = outputs.reshape(bs, num_ts_splits, -1)

        return outputs

    def configure_optimizers(self):
        ##TODO: Add `lr_scheduler`
        ##TODO: Understand this `Adafactor`
        optimizer = Adafactor(
            self.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
        )

        return [optimizer]

    def get_flops_and_params(self):
        X = torch.randn(1, 10, 2)
        y = torch.randn(1, 10, 1)
        images = torch.randn(1, 3, 299, 299)
        macs, params = profile(self, inputs=(X, y, images))
        return macs * 2 / 10**9, params / 10**6

    def training_step(self, train_batch, batch_idx):
        (X, y, categories, colors, fabrics, stores, temporal_features, gtrends), images = train_batch
        forecasted_sales = self.forward(X, y, 
                                        categories, colors, fabrics, stores,
                                        temporal_features, gtrends,
                                        images)

        y = y.squeeze()
        forecasted_sales = forecasted_sales.squeeze()
        loss = F.mse_loss(y, forecasted_sales)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, test_batch, batch_idx):
        (X, y, categories, colors, fabrics, stores, temporal_features, gtrends), images = test_batch
        forecasted_sales = self.forward(X, y, 
                                        categories, colors, fabrics, stores,
                                        temporal_features, gtrends,
                                        images)

        y = y.squeeze()
        forecasted_sales = forecasted_sales.squeeze()
        return y, forecasted_sales

    def validation_epoch_end(self, val_step_outputs):
        item_sales, forecasted_sales = (
            [x[0] for x in val_step_outputs],
            [x[1] for x in val_step_outputs]
        )
        item_sales, forecasted_sales = (
            torch.vstack(item_sales),
            torch.vstack(forecasted_sales),
        )
        rescaled_item_sales, rescaled_forecasted_sales = (
            item_sales * 53,  ##TODO: Dynamic this hyperparams
            forecasted_sales * 53,
        )  # 53 is the normalization factor (max of the sales of the training set == stfore_sales_norm_scalar.npy)
        loss = F.mse_loss(item_sales, forecasted_sales)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        wape = 100 * torch.sum(torch.abs(rescaled_item_sales - rescaled_forecasted_sales)) / torch.sum(rescaled_item_sales)
        ts = 100 * torch.sum(rescaled_item_sales - rescaled_forecasted_sales) / mae

        self.log("val_mae", mae)
        self.log("val_wWAPE", wape)
        self.log("val_wTS", ts)
        self.log("val_loss", loss)

        print(
            "Validation MAE:", mae.detach().cpu().numpy(),
            "Validation WAPE:", wape.detach().cpu().numpy(),
            "Validation Tracking Signal:", ts.detach().cpu().numpy(),
            "LR:", self.optimizers().param_groups[0]["lr"],
        )