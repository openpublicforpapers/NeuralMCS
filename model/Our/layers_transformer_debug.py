from config import FLAGS
from transformer_debug.Models import Transformer
import transformer_debug.Constants as Constants
from layers import get_prev_layer, NodeEmbedding, NodeEmbeddingCombinator
import torch
import torch.nn as nn


class OurTransformerPrototype(nn.Module):
    def __init__(
            self,
            in_dim,
            d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,
            beam_size, n_outputs, n_init_embedding_dim=None):
        super(OurTransformerPrototype, self).__init__()
        self.in_dim = in_dim
        self.transformer = Transformer(
            d_model, d_inner, n_layers, n_head,
            d_k, d_v, dropout, n_outputs)
        self.n_outputs = n_outputs
        self.beam_size = beam_size

    def _create_trainable_special_emb(self, in_dim):
        rtn = nn.Parameter(torch.Tensor(1, in_dim))
        nn.init.xavier_normal_(rtn)
        return rtn

    def forward(self, ins, batch_data, model):
        if self.training:
            return self._train(ins, batch_data, model)
        else:
            with torch.no_grad():
                return self._inference(ins, batch_data, model)

    ################################ train ################################

    def _train(self, ins, batch_data, model):
        # get the inputs
        g1_embs, g1_nids, g2_embs, g2_nids, pair_list = self._prepare_input(
            ins, batch_data, model)
        # call the transformer
        trans_out = self._call_transformer(g1_embs, g1_nids, g2_embs, g2_nids)
        # use loss function in a class
        y_preds_all = self._post_process(trans_out, pair_list)
        return y_preds_all

    def _prepare_input(self, ins, batch_data, model):
        # verify inputs are node embeddings
        assert type(get_prev_layer(self, model)) is NodeEmbedding or \
               type(get_prev_layer(self, model)) is NodeEmbeddingCombinator
        assert FLAGS.positional_encoding == False
        # get pair_list
        pair_list = batch_data.split_into_pair_list(ins, 'x')
        # get input to transformer
        transformer_in = self._prepare_input_for_pair_list(pair_list)
        return transformer_in

    def _prepare_input_for_pair_list(self, pair_list):

        # Get the lengths of input/output
        g1_lengths, g2_lengths = [], []
        for pair in pair_list:
            g1, g2 = pair.g1.get_nxgraph(), pair.g2.get_nxgraph()
            m, n = g1.number_of_nodes(), g2.number_of_nodes()
            g1_lengths.append(m)
            g2_lengths.append(n)

        max_g1_length, max_g2_length = max(g1_lengths), max(g2_lengths)

        # Initialize src and tgt
        # NOTE:
        #   Cannot have all zeros in a batch.
        #   In other words, if use FLAGS.batch_size and
        #   len(pair_list) < FLAGS.batch_size, mysterious nan error will happen
        #   caused by the extra 0s in the batch.
        # NOTE:
        #   pid = (length_G2) * (nid1) + (nid2) + (count_special)
        assert Constants.PAD == 0
        g1_embs = torch.zeros(
            len(pair_list), max_g1_length, self.in_dim,
            device=FLAGS.device)
        g1_nids = torch.zeros(
            len(pair_list), max_g1_length,
            device=FLAGS.device)
        g2_embs = torch.zeros(
            len(pair_list), max_g2_length, self.in_dim,
            device=FLAGS.device)
        g2_nids = torch.zeros(
            len(pair_list), max_g2_length,
            device=FLAGS.device)

        # Construct src and tgt
        for i, pair in enumerate(pair_list):
            g1, g2, x1, x2, m, n = self._unpack_pair(pair)

            for j, edge in enumerate(x1):
                g1_embs[i, j] = edge
                g1_nids[i, j] = j + Constants.SPECIAL_TOKEN_COUNT
            for j, edge in enumerate(x2):
                g2_embs[i, j] = edge
                g2_nids[i, j] = j + Constants.SPECIAL_TOKEN_COUNT

            if FLAGS.debug:
                pass
            # Two reasons why we normalize (I think):
            # 1) Positional Encoding is \in [0,1]... (though we do not need it for this transformer)
            # 2) Normalization helps with gradients (like batch-normalization)
            # 3) Do we really want to normalize though? TODO
            # NOTE: Tensorflow's official Transformer normalizes as well!
            '''
            g1_embs = F.normalize(g1_embs, p=2, dim=1)
            g2_embs = F.normalize(g2_embs, p=2, dim=1)
            '''

        return g1_embs, g1_nids, g2_embs, g2_nids, pair_list

    def _unpack_pair(self, pair):
        g1, g2 = pair.g1.get_nxgraph(), pair.g2.get_nxgraph()
        x1, x2 = pair.g1.x, pair.g2.x
        assert x1.shape[1] == self.in_dim and x2.shape[1] == self.in_dim, '{} {} {}' \
            .format(x1.shape[1], x2.shape[1], self.in_dim)
        m, n = x1.shape[0], x2.shape[0]
        assert m == g1.number_of_nodes() and n == g2.number_of_nodes()
        return g1, g2, x1, x2, m, n

    def _call_transformer(self, g1_embs, g1_nids, g2_embs, g2_nids):
        trans_out = self.transformer(g1_embs, g1_nids, g2_embs, g2_nids)
        # debug_tensor(trans_out)
        return trans_out

    def _post_process(self, trans_out, pair_list):
        y_preds_all = trans_out[0]
        _, n, m = y_preds_all.shape
        y_preds_all = y_preds_all.reshape((self.n_outputs, -1, n, m)).permute(1, 0, 3,
                                                                              2)  # shape: (batch_size, n_outputs, n, m)
        for i, pair in enumerate(pair_list):
            g1, g2 = pair.g1.nxgraph, pair.g2.nxgraph
            N, M = g1.number_of_nodes(), g2.number_of_nodes()
            pair.assign_y_pred_list([y_pred for y_pred in y_preds_all[i, :, :N, :M]],
                                    format='torch_{}'.format(FLAGS.device))
        return y_preds_all

    ################################ inference ################################

    def _inference(self, ins, batch_data, model):
        # get the inputs
        g1_embs, g1_nids, g2_embs, g2_nids, pair_list = self._prepare_input(ins, batch_data, model)
        # call the transformer
        trans_out = self._call_transformer(g1_embs, g1_nids, g2_embs, g2_nids)
        # call loss function
        y_preds_all = self._post_process(trans_out, pair_list)
        return y_preds_all
