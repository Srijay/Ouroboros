import torch
import torch.nn as nn
from layers import build_mlp


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self, input_dim, output_dim=None, hidden_dim=16,
                 pooling='avg', mlp_normalization='none', noise_dim=2):
        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling

        net1_layers = [2*input_dim, hidden_dim, 2 * hidden_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
        self.net1.cuda()
        self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
        self.net2.cuda()
        self.net2.apply(_init_weights)

    def forward(self, object_embeddings, edge_weights, edges):

        dtype, device = object_embeddings.dtype, object_embeddings.device
        O, T = object_embeddings.size(0), edge_weights.size(0)

        #layout_noise = torch.randn((1, self.noise_dim), dtype=object_embeddings.dtype, device=object_embeddings.device).repeat((O, 1)).view(O, self.noise_dim)
        layout_noise = torch.randn((O, self.noise_dim), dtype=object_embeddings.dtype, device=object_embeddings.device)
        object_embeddings = torch.cat([object_embeddings, layout_noise], dim=1)

        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        # Get current vectors for subjects and objects; these have shape (T, Din)
        cur_s_vecs = object_embeddings[s_idx]
        cur_o_vecs = object_embeddings[o_idx]

        # Get current vectors for triples; shape is (T, 3 * Din)
        # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
        cur_t_vecs = torch.cat([cur_s_vecs, cur_o_vecs], dim=1)
        cur_t_vecs = torch.mul(cur_t_vecs,edge_weights) #in case of gconv version 1

        new_t_vecs = self.net1(cur_t_vecs)
        # new_t_vecs = torch.mul(new_t_vecs, edge_weights) #in case of gconv version 2

        # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
        # p vecs have shape (T, Dout)
        new_s_vecs = new_t_vecs[:, :H]
        new_o_vecs = new_t_vecs[:, H:]

        # Allocate space for pooled object vectors of shape (O, H)
        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        # we first need to expand the indices to have shape (T, D)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'avg':
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            # Divide the new object vectors by the number of times they
            # appeared, but first clamp at 1 to avoid dividing by zero;
            # objects that appear in no triples will have output vector 0
            # so this will not affect them.
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (O, Dout)

        new_obj_vecs = self.net2(pooled_obj_vecs)
        return new_obj_vecs


class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
                 mlp_normalization='none',noise_dim=0):
        super(GraphTripleConvNet, self).__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'pooling': pooling,
            'mlp_normalization': mlp_normalization,
            'noise_dim': noise_dim
        }
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, object_embeddings, edge_weights, edges):

        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs = gconv(object_embeddings, edge_weights, edges)
        return obj_vecs


