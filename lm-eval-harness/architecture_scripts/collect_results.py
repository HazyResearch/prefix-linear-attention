import os
import json
import pandas as pd
from collections import defaultdict

# replace this with your path to the log files
prefix_path = '/home/simarora/code/clean/prefix-linear-attention/lm-eval-harness/'
assert(os.path.exists(prefix_path)), print(f"Ths path {prefix_path} does not exist. Please update the ``prefix_path'' variable.")

chosen_metric='contains,none'             
tasks = {
        "based_swde": chosen_metric,
        'based_fda':chosen_metric, 
        'based_triviaqa': chosen_metric,
        "based_squad": chosen_metric,
        "based_drop": chosen_metric,
        "based_nq_512": chosen_metric,
        "based_nq_1024": chosen_metric,
        "based_nq_2048": chosen_metric,
}

output_dir="run_jrt_rnn_sweep"
extra = ""
full_sweep = False

def collect_results():

    task2model2scores = defaultdict(dict)
    task2model2preds = defaultdict(dict)

    timestamps = os.listdir(f"{prefix_path}/{output_dir}/{extra}")
    timestamps = [t for t in timestamps if 'json' not in t]
    # print(timestamps)
    for timestamp in timestamps:
        for task, metric in tasks.items():
            time_dir = f"{prefix_path}/{output_dir}/{extra}/{timestamp}/"
            entities = os.listdir(time_dir)

            data_paths = []
            if any(['https' in e for e in entities]): continue
            if any('hazy' in e for e in entities):
                try:
                    wandb_path = f"{prefix_path}/{output_dir}/{extra}/{timestamp}/hazy-research/"
                    wandb_items = os.listdir(wandb_path)
                    for entity in wandb_items:
                        data_path = f"{prefix_path}/{output_dir}/{extra}/{timestamp}/hazy-research/{entity}/"
                        models = os.listdir(data_path)
                        data_paths.append((entity, data_path, models))
                except:
                    data_path = f"{prefix_path}/{output_dir}/{extra}/{timestamp}/hazyresearch/"
                    entity = "hazyresearch"
                    models = os.listdir(data_path)
                    data_paths.append((entity, data_path, models))
            else:
                entity = entities[0]
                data_path = f"{prefix_path}/{output_dir}/{extra}/{timestamp}/{entity}/"
                models = os.listdir(data_path)
                data_paths.append((entity, data_path, models))

            # print(f"Found {len(models)} models for {task} in {timestamp}")
            for entity, data_path, models in data_paths:
                for model in models:
                    path = f"{data_path}/{model}"
                    task_path = f"{path}/{task}/"

                    # For the scores
                    try:
                        result_file = f'results.json'
                        with open(f"{task_path}/{result_file}", 'r') as f:
                            results = json.load(f)
                            if metric not in results['results'][task]:
                                breakpoint()
                            score = results['results'][task][metric]
                            length = results['context_length']

                            # Run tags
                            if "triviaqa" in task:
                                task_name = f"{task}".replace("based_", "").replace("2000", "")  
                            else:
                                task_name = f"{task}{length}".replace("based_", "").replace("2000", "")                        
                            model_name = f"{model}_{length}".lower()

                            if full_sweep:
                                assert results['config']['limit'] is None, print("Need to run on full dataset")
                            if len(model) > 30:
                                model_title = model.lower() #[:-15] # + model[-15:]
                            else: model_title = model.lower()

                            if 'f1' not in metric: 
                                score = score * 100
                            task2model2scores[task_name][model_title] = f"& %.1f" % (score)
                    except Exception as e:
                        # print(f"{task} not found for {model}")
                        # print(e)
                        continue

                    # For the predictions
                    try:
                        result_file = f' checkpoint_name__hazy-research__cylon__{model}_{task}.jsonl'
                        if not os.path.exists(f"{task_path}/{result_file}"):
                            result_file = f' checkpoint_name__hazy-research__attention__{model}_{task}.jsonl'
                        with open(f"{task_path}/{result_file}", 'r') as f:
                            results = json.load(f)
                            task2model2preds[task_name][model_name] = results
                    except:
                        result_file = f'checkpoint_name__{entity}__{model}_{task}.jsonl'
                        with open(f"{task_path}/{result_file}", 'r') as f:
                            results = json.load(f)
                            task2model2preds[task_name][model_name] = results

    scores_df = pd.DataFrame(task2model2scores)
    scores_df = scores_df.reindex(sorted(scores_df.columns), axis=1)
    scores_df = scores_df.sort_index()

    # Print Results
    print(scores_df)

    print(f"Summary:")
    print(f"{output_dir=}")
    for i, (task, model2preds) in enumerate(task2model2preds.items()):
        lengths = [len(preds) for m, preds in model2preds.items()]
        print(f"{task}: {len(model2preds)} models found -- {lengths}")
        if i == 1:
            break

    # Save the task2model2preds for inspection
    with open(f"{prefix_path}/{output_dir}/{extra}/task2model2preds.json", 'w') as f:
        json.dump(task2model2preds, f)

if __name__ == '__main__':
    collect_results()

