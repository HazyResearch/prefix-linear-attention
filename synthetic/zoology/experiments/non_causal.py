import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.non_autoreg import DisjointSetsConfig, DisjointSetsConfig_Multi

causal=False
VOCAB_SIZE = 2_048
output_type = 'intersect_token' 
model_name = 'based'
layers = 4
heads = 2
cache_dir="/scratch/simran/" # REPLACE WITH YOUR PATH

sweep_id = uuid.uuid4().hex[:6]
sweep_name = f"causal{causal}-{model_name}-{layers}L-head{heads}" + sweep_id

sizes = [
    (4, 16), (16, 4),
    (8, 32), (32, 8), 
    (64, 16), (16, 64), 
    (4, 128), (128, 4),
    (16, 256), (256, 16),
    (4, 256), (256, 4),
]
num_examples_lst = [int(20_000)] * len(sizes)
train_configs = []
for size, num_ex in zip(sizes, num_examples_lst):
    train_configs.append(
        DisjointSetsConfig(vocab_size=VOCAB_SIZE, num_examples=num_ex, short_length=size[0], long_length=size[1]),
    )

test_sizes = [
    (1, 32), (32, 1),
    (4, 32), (32, 4),
    (4, 128), (128, 4),
    (16, 256), (256, 16),
    (4, 256), (256, 4),
    (16, 512), (512, 16),
    (4, 512), (512, 4), 
    (8, 768), (768, 8),
    (16, 768), (768, 16),
    (4, 768), (768, 4),
]
num_examples_lst_test = [1_000] * len(test_sizes)
test_configs = []
for size, num_ex in zip(test_sizes, num_examples_lst_test):
    test_configs.append(
        DisjointSetsConfig(vocab_size=VOCAB_SIZE, num_examples=num_ex, short_length=size[0], long_length=size[1]),
    )

input_seq_len=max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 128
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    # can pass a tuple if you want a different batch size for train and test
    batch_size=(batch_size, batch_size / 8),
    cache_dir=f"{cache_dir}/zoology/data/",
    mode=output_type
)

# 2. Next, we are going to collect all the different model configs we want to sweep
models = []
model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}), "vocab_size": VOCAB_SIZE,
}

# define this conv outside of if/else block because it is used in multiple models
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
        "causal": causal,
    }
)


# based
for d_model in [
    36, 
    48,
    64, 
    96,
    128, 
]:
    for ftr_dim in [
        4,
        8, 
        16, 
        24,
    ]:
        lin_attn = dict(
            name="zoology.mixers.based.Based",
            kwargs={
                "l_max": input_seq_len,
                "feature_dim": ftr_dim,
                "feature_name": "taylor_exp",
                "num_key_value_heads": heads,
                "num_heads": heads,
                "train_view": "quadratic",
                "apply_rotary": "false",
                "causal": causal,
            }
        )
        mixer = dict(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, lin_attn]}
        )
        name = f"based"
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name=name,
            causal=causal,
            **model_factory_kwargs
        )
        models.append(model)


included = [model_name] 
models = [m for m in models if any([i in m.name for i in included])]
configs = []
for model in models:
    for lr in [0.0001, 0.0005, 0.0008]: 
            run_id = f"{model.name}-lr{lr:.1e}"
            config = TrainConfig(
                model=model,
                data=data,
                learning_rate=lr,
                max_epochs=48,
                logger=LoggerConfig(
                    project_name="zoology",
                    entity="hazy-research"
                ),
                slice_keys=['short_length_long_length'],
                sweep_id=sweep_name,
                run_id=run_id,
                predictions_path=f"{cache_dir}/zoology/predictions/{run_id}",
                collect_predictions=True,
            )
            configs.append(config)

