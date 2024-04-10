import ego
from ego.modules import DenseTower
from ego.layers import Normalization, LeakyRelu
import ego.layers as L
from tensorflow.python.keras.layers import Layer
import tensorflow as tf
from ego.variable import Variable
from tensorflow.python.ops import variables as tf_variables
import numpy as np

# conda install gast=0.2.2 to solve the unable to train warning
"""
based on cap custom sfmx
1. Add L1 L2 global cat
"""
hop1_num = 10
hop2_num = 10
sparse_dim = 16
embedding_dim = 16
# neg_num = 300
# logits_expand_rate = 20
output_tracking_stats = True

SLOT_IDS = [2001, 3001, 4001, 2002, 3002, 4002]
VA_ID_SLOT_IDS = SLOT_IDS[:1]
VA_HOP1_SLOT_IDS = SLOT_IDS[1:2]
VA_HOP2_SLOT_IDS = SLOT_IDS[2:3]
VB_ID_SLOT_IDS = SLOT_IDS[3:4]
VB_HOP1_SLOT_IDS = SLOT_IDS[4:5]
VB_HOP2_SLOT_IDS = SLOT_IDS[5:]

slot_embs = ego.get_slots(SLOT_IDS,
                          [sparse_dim] * len(VA_ID_SLOT_IDS) + [(hop1_num, sparse_dim)] * len(VA_HOP1_SLOT_IDS) + [
                              (hop1_num * hop2_num, sparse_dim)] * len(VA_HOP2_SLOT_IDS)
                          + [sparse_dim] * len(VB_ID_SLOT_IDS) + [(hop1_num, sparse_dim)] * len(VB_HOP1_SLOT_IDS) + [
                              (hop1_num * hop2_num, sparse_dim)] * len(VB_HOP2_SLOT_IDS),
                          ['sum'] * len(VA_ID_SLOT_IDS) + ['tile'] * len(VA_HOP1_SLOT_IDS) + ['tile'] * len(
                              VA_HOP2_SLOT_IDS)
                          + ['sum'] * len(VB_ID_SLOT_IDS) + ['tile'] * len(VB_HOP1_SLOT_IDS) + ['tile'] * len(
                              VB_HOP2_SLOT_IDS)
                          )


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])
    return shape


def process_seq(embs, name):
    """_summary_

    Args:
        embs (List): [1 , emb_dim] * fea_num

    Returns:
        seq_len: [bs, 1]
        emb: [bs, max_seq_len, emb_dim * fea_num]
    """
    seq_len = tf.concat([seq_len for _, seq_len in embs], axis=1)  # [bs, fea_num]
    seq_len = tf.reduce_max(seq_len, axis=1, keepdims=True)  # [bs, 1]
    emb = tf.convert_to_tensor([slot_emb for slot_emb, _ in embs])  # fea_num, batch_size, seq_len, sparse_dim
    emb = tf.transpose(emb, [1, 2, 0, 3])
    emb_shape = get_shape(emb)
    emb = tf.reshape(emb, [-1, emb_shape[1], emb_shape[2] * emb_shape[3]])  # batch_size, max_seq_len, emb_dim * fea_num
    emb = Normalization("item_list_norm_{}".format(name))(emb)
    return seq_len, emb


va_embs = slot_embs[: 1]
va_hop1_embs = slot_embs[1: 2]
va_hop2_embs = slot_embs[2: 3]

vb_embs = slot_embs[3: 4]
vb_hop1_embs = slot_embs[4: 5]
vb_hop2_embs = slot_embs[5: 6]

va_emb = tf.concat(va_embs, axis=1)  # batch_size, user_fea_num * sparse_dim
va_emb = Normalization('va_embs_norm')(va_emb)

vb_emb = tf.concat(vb_embs, axis=1)  # batch_size, user_fea_num * sparse_dim
vb_emb = Normalization('vb_embs_norm')(vb_emb)

va_hop1_len, va_hop1_emb = process_seq(va_hop1_embs, "va_hop1")
va_hop2_len, va_hop2_emb = process_seq(va_hop2_embs, "va_hop2")

vb_hop1_len, vb_hop1_emb = process_seq(vb_hop1_embs, "vb_hop1")
vb_hop2_len, vb_hop2_emb = process_seq(vb_hop2_embs, "vb_hop2")

batch_size = get_shape(va_hop1_len)[0]


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)
    return initial


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, idx, hidden_dim=32, neigh_input_dim=None,
                 dropout=0., bias=False, act=LeakyRelu(alpha=0.02, name="agg_act"), name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.hidden_dim = hidden_dim

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        param_initializer = tf.truncated_normal_initializer(stddev=0.001, dtype=tf.float32)
        self.mlp = DenseTower(name='mlp{}'.format(idx),
                              output_dims=[hidden_dim],
                              kernel_initializers=[param_initializer],
                              bias_initializers=[param_initializer],
                              activations=[LeakyRelu(alpha=0.02)],
                              norms=[False])
        # glorot([neigh_input_dim, output_dim],name='neigh_weights')

        self.va_weight = DenseTower(name='weight_va{}'.format(idx),
                                    output_dims=[output_dim],
                                    kernel_initializers=[glorot([hidden_dim, output_dim], name='va_init')],
                                    # TODO: modify same to paper
                                    bias_initializers=[None],
                                    activations=[None],
                                    norms=[False])

        self.neigh_weight = DenseTower(name='weight_neigh{}'.format(idx),
                                       output_dims=[output_dim],
                                       kernel_initializers=[glorot([hidden_dim, output_dim], name='neigh_init')],
                                       # TODO: modify same to paper
                                       bias_initializers=[None],
                                       activations=[None],
                                       norms=[False])

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def __call__(self, inputs):
        """_summary_

        Args:
            inputs (self_vecs, neigh_vecs): self_vecs, neigh_vecs : [B * hop1_num, hop2_num, dim]
        Returns:
            _type_: [B * hop1_num, dim]
        """
        print("debug0")
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        # for l in self.mlp_layers:
        #     h_reshaped = l(h_reshaped)
        h_reshaped = self.mlp(h_reshaped)

        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        self_vecs = self.va_weight(self_vecs)
        neigh_h = self.neigh_weight(neigh_h)

        output = tf.add_n([self_vecs, neigh_h])
        if self.act != None:
            output = self.act(output)
        # output = tf.concat([self_vecs, neigh_h], axis=1) #B, dim * 2
        # print("debug1:",output)
        # output = self.weight(output)
        return output


# 25,10
# b*25,16   *  b*25, 10, 16 -> b*25,16
# b*1,16 * b*1,25, 16 -> b*1,16

# 25,10,8
# b*250, 16   *  b*250, 8, 16 -> b*250,16
# b*25, 16   *  b*25, 10, 16 -> b*25,16
# b*1, 16, * b*1,25, 16 -> b*1,16

def agg_hops(emb, hop1_emb, hop2_emb, hop1_layer, hop2_layer):
    hop1_emb = tf.reshape(hop1_emb, [batch_size * hop1_num, embedding_dim])  # [bs *  max_seq_len1, emb_dim * fea_num]
    hop2_emb = tf.reshape(hop2_emb, [batch_size * hop1_num, hop2_num,
                                     embedding_dim])  # [bs *  max_seq_len1,max_seq_len2, emb_dim * fea_num]
    print("test")
    hop2_out = hop2_layer([hop1_emb, hop2_emb])  # [bs *  max_seq_len1, emb_dim * fea_num]
    print("debug2:", hop2_out)
    hop2_out = tf.reshape(hop2_out, [batch_size, hop1_num, embedding_dim])
    hop1_out = hop1_layer([emb, hop2_out])  # [bs, emb_dim * fea_num]
    hop1_out = tf.nn.l2_normalize(hop1_out, 1)

    return hop1_out


hop2_layer = MaxPoolingAggregator(input_dim=16, output_dim=16, idx="hop2_layer", hidden_dim=32)
hop1_layer = MaxPoolingAggregator(input_dim=16, output_dim=16, idx="hop1_layer", hidden_dim=32, act=None)

va_res = agg_hops(va_emb, va_hop1_emb, va_hop2_emb, hop1_layer, hop2_layer)
vb_res = agg_hops(vb_emb, vb_hop1_emb, vb_hop2_emb, hop1_layer, hop2_layer)


def softmax_loss(va_res, vb_res, shift=0, scale=1):
    origin_scores = tf.matmul(va_res, vb_res, transpose_b=True)
    pos_scores = tf.linalg.tensor_diag_part(origin_scores)  # B
    scores = tf.linalg.set_diag(origin_scores, pos_scores - shift)
    probs = tf.nn.softmax(scale * scores, axis=-1)

    diag = tf.linalg.tensor_diag_part(probs)
    click = tf.expand_dims(diag, axis=1, name='click')

    neg_mask = tf.cast(tf.linalg.set_diag(tf.ones_like(origin_scores), tf.zeros_like(pos_scores)), dtype=tf.bool)
    pos_cos = tf.reduce_mean(pos_scores, name='pos_cos')
    neg_cos = tf.reduce_mean(tf.boolean_mask(origin_scores, neg_mask), name='neg_cos')

    return click, origin_scores, pos_cos, neg_cos


click0, origin_scores, pos_cos, neg_cos = softmax_loss(va_res, vb_res, shift=0, scale=1)
label0 = ego.get_label(name='label0', label_idx=0)
loss0 = tf.multiply(label0, -tf.log(click0 + 1e-8))
loss_weight0 = ego.get_loss_weight(name='loss_weight0')


def diag_metrics(score_tensor):
    '''
    :param score_tensor: shape [batch_size, batch_size]
    :return: ranking of scores
    '''
    pred_idx = tf.cast(tf.argmax(score_tensor, 1), tf.int32)  # B
    batch_size = get_shape(pred_idx)[0]
    true_idx = tf.range(batch_size)
    p = tf.reduce_mean(tf.cast(tf.equal(pred_idx, true_idx), tf.float32), name='p_at_1')

    ranks = tf.argsort(score_tensor, axis=-1, direction='DESCENDING')  # B, B
    true_idx_2d = tf.tile(tf.expand_dims(true_idx, 1), [1, batch_size])
    true_rank = tf.cast(tf.where(tf.equal(ranks, true_idx_2d))[:, 1:], tf.float32)
    mrr = tf.reduce_mean(1.0 / (true_rank + 1.0), name='mrr')
    return p, mrr


# for eval HR@N
# ego.add_print_vars_to_hdfs([user_eb])
round0 = ego.Round(name='eval',
                   losses=None,
                   loss_weights=None,
                   targets=click0,
                   target_names='click',
                   labels=label0,
                   sparse_train=False,
                   is_layer_res_output=False)

# TODO: didn't finish
if output_tracking_stats:
    # output tracking stats
    # negative/positive avg cosine sim in log
    # cos_similarity_true = tf.reduce_sum(tf.reduce_sum(pos_score, axis=1), axis=0)
    # cos_similarity_false= tf.reduce_sum(tf.reduce_sum(neg_score, axis=1), axis=0)
    true_num = tf.cast(batch_size, tf.float32)
    false_num = tf.cast(batch_size * 512, tf.float32)
    # mrr
    p, mrr = diag_metrics(origin_scores)

    cos_to_print = tf.stack([pos_cos, neg_cos, true_num, false_num, mrr, p])
    loss0 += 0 * tf.reduce_sum(cos_to_print)
    ego.add_print_vars_to_log([pos_cos, neg_cos, true_num, false_num, mrr, p])
    # 1,1,511.963135,153588.938,0.0133173335,0.00195416063

round1 = ego.Round(name='click_train',
                   losses=loss0,
                   loss_weights=loss_weight0,
                   targets=click0,
                   target_names='click',
                   labels=label0,
                   sparse_train=True,
                   is_join_data_output=output_tracking_stats)

rounds = [round0, round1]

ego.compile(rounds=rounds)