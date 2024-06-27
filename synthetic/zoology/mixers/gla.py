import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from typing import Optional, Tuple, Union

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from fla.ops.gla import fused_chunk_gla


class GatedLinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.n_head
        
        self.gate_fn = nn.functional.silu
        assert config.use_gk and not config.use_gv, "Only use_gk is supported for simplicity."

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim//2, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim//2, bias=False)
        self.k_gate =  nn.Sequential(nn.Linear(self.embed_dim, 16, bias=False), nn.Linear(16, self.embed_dim // 2))

        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.g_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.head_dim = self.embed_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)

        self.post_init()

    def post_init(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        if isinstance(self.k_gate, nn.Sequential):
            nn.init.xavier_uniform_(self.k_gate[0].weight, gain=2 ** -2.5)
            nn.init.xavier_uniform_(self.k_gate[1].weight, gain=2 ** -2.5)
        else:
            nn.init.xavier_uniform_(self.k_gate.weight, gain=2 ** -2.5)

    def forward(self, x, hidden_states=None):
        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        k_gate = self.k_gate(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        output, new_hidden_states = self.gated_linear_attention(q, k, v, k_gate, hidden_states=hidden_states)
        output = self.gate_fn(g) * output
        output = self.out_proj(output)
        return output, new_hidden_states

    def gated_linear_attention(self, q, k, v, gk, normalizer=16, hidden_states=None):
        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        k = rearrange(k, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        gk = rearrange(gk, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        gk = F.logsigmoid(gk) / normalizer

        o, new_hidden_states = fused_chunk_gla(q, k, v, gk, initial_state=hidden_states, output_final_state=True)

        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        return o, new_hidden_states


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class GLAConfig(PretrainedConfig):
    model_type = "gla"
    alternate_layers = []
    alternate_layers_2 = []
    alternate_layer_type = "conv"
    alternate_layer_type_2 = "swa" 
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        d_model=2048,
        n_head=8,
        n_layer=24,
        use_gk=True,
        use_gv=False,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        context_length=2048,
        vocab_size=50432,
        tie_word_embeddings=False,
        load_from_llama=False,
        **kwargs,
    ):
    
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.context_length = context_length
        self.use_cache = use_cache

        self.use_gk = use_gk
        self.use_gv = use_gv
        self.load_from_llama = load_from_llama

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class GLABlock(nn.Module):
    def __init__(self, config, i):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model, eps=1e-5)
        self.attn = GatedLinearAttention(config)
        self.ln_2 = RMSNorm(config.d_model, eps=1e-5)

        mlp_ratio = config.mlp_ratio
        multiple_of = config.multiple_of
        mlp_hidden = int(config.d_model * mlp_ratio * 2 / 3)
        mlp_hidden = multiple_of * ((mlp_hidden + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(config.d_model, mlp_hidden, bias=False)
        self.w2 = nn.Linear(mlp_hidden, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, mlp_hidden, bias=False)
        self.mlp = lambda x: self.w2(F.silu(self.w1(x)) * self.w3(x))


    def forward(self, x, hidden_states=None):
        # attention/rnn
        result = self.attn(self.ln_1(x), hidden_states)
        if len(result) == 2: x_att, new_hidden_states = result
        else: x_att, new_hidden_states = result, result
        x = x + x_att
        # ffn
        x_mlp = self.mlp(self.ln_2(x))
        x = x + x_mlp
        return x, new_hidden_states


class GLAPreTrainedModel(PreTrainedModel):
    config_class = GLAConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["GLABlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class GLAModel(GLAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        layers = []
        for i in range(config.n_layer):
           layers.append(GLABlock(config, i))
        self.h = nn.ModuleList(layers)
        self.ln_f = RMSNorm(config.d_model, eps=1e-5)

    def forward( self,
        input_ids: torch.LongTensor = None,
        hidden_states: torch.Tensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if hidden_states is None:
            hidden_states = [None] * self.config.n_layer

        new_hidden_states = []
        x = self.wte(input_ids)
        for block, hidden_state in zip(self.h, hidden_states):
            x, last_context_state = block(x, hidden_state)
            new_hidden_states.append(last_context_state)

        x = self.ln_f(x)            
        
        # the hidden states now means the recurrent hidden states
        return BaseModelOutputWithPast(
            last_hidden_state=x,
            hidden_states=new_hidden_states,
        )


class GLAForCausalLM(GLAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.transformer = GLAModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.n_layer = config.n_layer

        self.apply(self._init_weights)
        self._post_init()

    def _init_weights(self, module):
        """general init strategy"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
            if hasattr(module, "bias") and  module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)
    
    def _post_init(self):
        """custom init strategy"""
        for name, module in self.named_modules():
            if hasattr(module, "post_init"):
                module.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        assert return_dict is True
        
        xmr_output = self.transformer(input_ids, hidden_states)
        logits = self.lm_head(xmr_output.last_hidden_state)
        new_hidden_states = xmr_output.hidden_states
        
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-1)
        else:
            loss = None

        # output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=new_hidden_states,
        )
    