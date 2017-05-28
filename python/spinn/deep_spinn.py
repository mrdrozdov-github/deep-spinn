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
    """
    This model is inspired by `Irsoy and Cardie. Deep Recursive Neural 
    Networks for Compositionality in Language. 2014`:
    http://www.cs.cornell.edu/~oirsoy/drsv.htm

    A Deep SPINN has multiple layers of SPINN, where each successive layer
    incorporates the top element of the stack of the previous layer during
    the REDUCE operation. Each layer has an independent projection layer,
    although it should be optional to treat each layer's input as different
    hidden states in an MLP or RNN. Each SPINN layer also has an independent
    tracking LSTM, any of which has hidden states that can be used as input
    to predict the next transition.
    """

    optimize_transition_loss = True

    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()

        # Local Properties
        model_dim = kwargs.get('model_dim')

        # Necessary properties.
        self.use_sentence_pair = kwargs.get('use_sentence_pair')

        # Composition Unit
        composition_args = kwargs.get('composition_args')
        composition_args.composition = None

        # Multilayer Init
        n_layers = 1

        self.layers = []
        for i_layer in range(n_layers):
            _kwargs = copy.deepcopy(kwargs)

            _composition_args = _kwargs.get('composition_args')

            if i_layer == 0:
                _composition_args.external_size = None
            else:
                _composition_args.external_size = _composition_args.size

            _kwargs['composition_args'].composition = _composition_args.fn(
                size=_composition_args.size,
                tracker_size=_composition_args.tracker_size,
                external_size=_composition_args.external_size,
                use_tracking_in_composition=_composition_args.use_tracking_in_composition,
                composition_ln=_composition_args.composition_ln)

            # SPINN Layer
            layer_name = "layer_{}".format(i_layer)
            setattr(self, layer_name, _BaseModel(*args, **_kwargs))
            self.layers.append(getattr(self, layer_name))

            # In Between Layer
            # layer_name = "layer_inbetween_{}".format(i_layer)
            # setattr(self, layer_name, nn.Linear(model_dim, model_dim))

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
        """

        There are L x N "inputs", which is the number of layers by the number of tokens.
        Each of these will need an affine transformation using one of the L linear layers.

        Each step/reduce will need this input. Can be done by modifying the buffer/stack.

        """

        assert use_internal_parser == False, "Deep SPINN does not support predicted transitions yet."

        return self.layers[0](sentences, transitions, y_batch,
            use_internal_parser, validate_transitions)
