# -*- coding: utf-8 -*-
# port of transformer4rec


import torch
from torch import nn
import numpy as np
from transformers4rec import torch as tr

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss, RegLoss
from recbole.utils import InputType

from merlin.schema import Schema, ColumnSchema, Tags
from merlin.models.utils.schema_utils import create_categorical_column

class T4Rec(SequentialRecommender):
  

    input_type = InputType.PAIRWISE
  

    def __init__(self, config, dataset):
        super(T4Rec, self).__init__(config, dataset)

        d_model = 320
        max_sequence_length = 10

        # Create a Schema
        schema = Schema([
            # create_categorical_column('item_id', max_sequence_length)
            ColumnSchema(name='item_id', tags=[Tags.CATEGORICAL], dtype=np.int32, is_list=True, properties={
               "domain": {"name": 'item_id', "min": 0, "max": 10},
               "value_count": { "min": 0, "max": 100}

            })
        ])
        print("shema is ", schema)

    
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            embedding_dim_default=128,
            max_sequence_length=max_sequence_length,
            # continuous_projection=64,
            aggregation="concat",
            d_output=d_model,
            # masking="mlm",            
        )
      
        # Define Next item prediction-task 
        prediction_task = tr.NextItemPredictionTask(weight_tying=True)

        transformer_config = tr.XLNetConfig.build(
            d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
        )

        # Get the end-to-end model 
        self.model = transformer_config.to_torch_model(input_module, prediction_task)
       


    def calculate_loss(self, interaction):
        ret = self.model.forward(item_seq)
        item_seq = interaction[self.ITEM_SEQ]
        ret = self.model.forward(item_seq)
        loss = ret['loss']
        return loss
        

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]  # [B Len]
        test_item = interaction[self.ITEM_ID]

        seq_output = self.forward(item_seq)  # [B H]
        test_item_emb = self.item_embedding(test_item)  # [B H]

        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    
