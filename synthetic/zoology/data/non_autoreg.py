import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment


class MQARMiniConfig(DataSegmentConfig):
    name: str="mini_multiquery_ar"
    power_a: float=0.01
    num_kv_pairs: int=8
    num_queries: int=2
    random_non_queries: bool=True
    include_slices: bool=True
    causal: bool=True

    def build(self, seed: int) -> DataSegment:
        return mini_multiquery_ar(**self.model_dump(), seed=seed)

def mini_multiquery_ar(
    vocab_size: int,
    num_examples: int,
    num_queries: int,
    input_seq_len: int,
    seed: int,
    num_kv_pairs: int=8,
    random_non_queries: bool=False,
    include_slices: bool=True,
    causal: bool=True,
    **kwargs
) -> DataSegment:
    
    random_non_queries = False
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 8 <= input_seq_len
    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(2, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    # create sequences, putting keys and values at the beginning.
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # get a subset of k keys randomly without replacement
    key_choices= np.apply_along_axis(np.random.choice, 1, keys, replace=False, size=num_queries)
    value_choices = np.zeros((num_examples, num_queries), dtype=np.int64)
    for i in range(num_examples):
        for j in range(num_queries):
            key = key_choices[i, j]
            value_choices[i, j] = values[i, np.where(keys[i] == key)[0][0]]
    
    # add special token 
    sep = np.zeros((num_examples, 1), dtype=np.int64)
    sep.fill(1)

    # queries and answers
    filler = np.zeros((num_examples, num_queries*2), dtype=np.int64)
    examples = np.concatenate([
        kvs, 
        sep,
        key_choices,
        sep,
        filler,
    ], axis=1)

    labels = np.full((num_examples, examples.shape[1]), -100, dtype=np.int64)
    labels[:, -num_queries*2::2] = key_choices
    labels[:, -num_queries*2+1::2] = value_choices

    if causal:
        inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    else:
        inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, :-1])
        breakpoint()
    
    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size-2, size=inputs.shape)[inputs == 0]
    return DataSegment(
        inputs, 
        labels, 
        slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len}
    )


class DisjointSetsConfig(DataSegmentConfig):
    name: str="disjoint_sets"
    short_length: int=4
    long_length: int=8
    include_slices: bool=True

    def build(self, seed: int) -> DataSegment:
        return disjoint_sets(**self.model_dump(), seed=seed)

def disjoint_sets(
    vocab_size: int,
    num_examples: int,
    short_length: int,
    long_length: int,
    seed: int,
    **kwargs
) -> DataSegment:    
    np.random.seed(seed)

    # special toks
    prefix_tok = 1          # binary
    sep_lists_tok = 2      # binary
    mask_tok = 0                # binary
    sep_ans_tok = 3           # separator between short and long lists
    num_special_tokens = 4

    inputs = []
    labels = []
    for i in range(num_examples):

        # get a short and long list of tokens.
        half_vocab = vocab_size // 2
        all_idx = np.arange(num_special_tokens, vocab_size)
        all_idx_shuffled = np.random.permutation(all_idx)
        all_short = all_idx_shuffled[:half_vocab]
        all_long = all_idx_shuffled[half_vocab:]
        short_tokens = np.random.choice(all_short, short_length, replace=False)
        long_tokens = np.random.choice(all_long, long_length, replace=False)

        # make sure a token in short occurs in long
        overlap_token = short_tokens[np.random.randint(short_length)]
        long_tokens[np.random.randint(long_length)] = overlap_token
        answer_tok = overlap_token  
        
        # Inputs and outputs
        input_seq = np.concatenate([[prefix_tok], short_tokens, [sep_lists_tok], long_tokens, [sep_ans_tok], [answer_tok]])
        input = torch.tensor(input_seq)

        label = torch.full_like(input[:-1], -100) 
        label[-1] = input[-1]
        input = input[:-1]

        # Save
        inputs.append(input)
        labels.append(label)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    short_length_long_length = f"{short_length}_{long_length}"
    return DataSegment(
        inputs, 
        labels, 
        slices={"short_length_long_length": short_length_long_length}
    )

# main 
if __name__ == "__main__":

    test_data = 'disjoint'

    if test_data == 'mqar_mini':
        config = MQARMiniConfig(
            vocab_size=8_192,
            num_examples=12,
            num_queries=16,
            input_seq_len=256,
            power_a=0.01,
            num_kv_pairs=16,
            random_non_queries=False,
            include_slices=True
        )
    else:
        num_examples = 12
        config = DisjointSetsConfig(
            vocab_size=24,
            num_examples = num_examples,
            short_length=4,
            long_length=8,
            include_slices=True
        )
    data = config.build(seed=0)
    for i in range(num_examples):
        print(data.inputs[i])
        print(data.labels[i])
        print("\n----------------------\n")

