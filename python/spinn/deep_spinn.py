import copy

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from spinn.util.blocks import MLP
from spinn.util.blocks import the_gpu, to_gpu
from spinn.util.catalan import interpolate

from spinn.spinn_core_model import BaseModel as _BaseModel
from spinn.spinn_core_model import SPINN


def build_model(data_manager, initial_embeddings, vocab_size,
                num_classes, FLAGS, context_args, composition_args):
    model_cls = BaseModel
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA

    return model_cls(model_dim=FLAGS.model_dim,
                     word_embedding_dim=FLAGS.word_embedding_dim,
                     vocab_size=vocab_size,
                     initial_embeddings=initial_embeddings,
                     num_classes=num_classes,
                     embedding_keep_rate=FLAGS.embedding_keep_rate,
                     tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
                     transition_weight=FLAGS.transition_weight,
                     use_sentence_pair=use_sentence_pair,
                     lateral_tracking=FLAGS.lateral_tracking,
                     tracking_ln=FLAGS.tracking_ln,
                     use_tracking_in_composition=FLAGS.use_tracking_in_composition,
                     predict_use_cell=FLAGS.predict_use_cell,
                     use_difference_feature=FLAGS.use_difference_feature,
                     use_product_feature=FLAGS.use_product_feature,
                     classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
                     mlp_dim=FLAGS.mlp_dim,
                     num_mlp_layers=FLAGS.num_mlp_layers,
                     mlp_ln=FLAGS.mlp_ln,
                     context_args=context_args,
                     composition_args=composition_args,
                     )


class DeepSPINN(SPINN):
    pass


class BaseModel(nn.Module):

    optimize_transition_loss = True

    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()

        # Necessary properties.

        self.use_sentence_pair = kwargs.get('use_sentence_pair')


        # Multilayer Init

        n_layers = 1

        self.layers = []
        for i_layer in range(n_layers):
            layer_name = "layer_{}".format(i_layer)
            setattr(self, layer_name, _BaseModel(*args, **kwargs))
            self.layers.append(getattr(self, layer_name))

    @property
    def transition_loss(self):
        return self.layers[0].transition_loss

    @property
    def transition_acc(self):
        return self.layers[0].transition_acc

    def get_transitions_per_example(self, style="preds"):
        return self.layers[0].get_transitions_per_example(style)

    def forward(self, sentences, transitions, y_batch=None,
                use_internal_parser=False, validate_transitions=True):

        return self.layers[0](sentences, transitions, y_batch,
            use_internal_parser, validate_transitions)
