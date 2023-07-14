import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ZeroShotDataset():
    def __init__(self, data_df, img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len, train=True):
        self.data_df = data_df
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = trend_len
        self.img_root = img_root

        print("Starting dataset creation process...")
        local_savepath = f"visuelle_{'train' if train else 'test'}_processed.pt"

        print("Loading dataset")
        if os.path.isfile(local_savepath):
            self.dataset = torch.load(local_savepath)  # load dataset directly from saved files
        else:
            self.dataset = self.preprocess_data()  # If file doesn't exist or u wish to re-process/create from scratch
            torch.save(self.dataset, local_savepath)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # Define image transformation
        ##TODO: Dynamic these params
        img_transforms = Compose([Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        ##TODO: Read by lmdb
        # Read image
        img_path = self.data_df.loc[idx, "image_path"]
        img = Image.open(os.path.join(self.img_root, img_path)).convert("RGB")  # Make sure to use RGB
        pt_img = img_transforms(img)

        return self.dataset[idx], pt_img

    def preprocess_data(self):
        data = self.data_df

        # Get the Gtrends time series associated with each product
        # Read the images (extracted image features) as well
        gtrends = []
        for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
            cat, col, fab, fiq_attr, start_date = row["category"], row["color"], row["fabric"], row["extra"], \
                                                    row["release_date"]
            
            # Get the gtrend signal up to the previous year (52 weeks) of the release date
            gtrend_start = start_date - pd.DateOffset(weeks=52)
            cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[:self.trend_len]
            col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[:self.trend_len]
            fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[:self.trend_len]

            cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1,1)).flatten()
            col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1,1)).flatten()
            fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1,1)).flatten()
            multitrends =  np.vstack([cat_gtrend, col_gtrend, fab_gtrend])

            # Append them to the lists
            gtrends.append(multitrends)

        # Convert to numpy arrays
        gtrends = np.array(gtrends)

        # Remove non-numerical information
        ##TODO: One hot these??
        data.drop(["external_code", "season", "release_date", "image_path"], axis=1, inplace=True)

        # Create tensors for each part of the input/output
        item_sales, temporal_features = torch.FloatTensor(data.iloc[:, :12].values), torch.FloatTensor(data.iloc[:, 13:17].values)
        categories, colors, fabrics = [self.cat_dict[val] for val in data.iloc[:].category.values], \
                                       [self.col_dict[val] for val in data.iloc[:].color.values], \
                                       [self.fab_dict[val] for val in data.iloc[:].fabric.values]

        categories, colors, fabrics = torch.LongTensor(categories), torch.LongTensor(colors), torch.LongTensor(fabrics)
        gtrends = torch.FloatTensor(gtrends)

        return TensorDataset(item_sales, categories, colors, fabrics, temporal_features, gtrends)