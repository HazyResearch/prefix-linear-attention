import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs

def plot(
    df: pd.DataFrame,
    metric: str="valid/accuracy",
    tag: str="",
):  
    # put three plots in one figure
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    
    state_key = 'model.sequence_mixer.kwargs.configs.1.kwargs.feature_dim'
    print(f"Found {len(df)} finished runs")
    df['model.name'] = df['model.name'].str.upper()
    new_key = df['model.name'].astype(str) + " Causal: " + df['model.causal'].astype(str)
    df['Model'] = new_key
    df['log_state_size'] = np.log(df['state_size'])
    df = df[df['state'] == 'finished']
    df = df[df['model.d_model'] > 40]
    df = df[df[state_key] < 200000]

    # Group metrics by slices
    # metrics = [k for k in df.columns if 'valid/length_slice/' in k]
    metrics = [k for k in df.columns if 'valid/length_slice/' in k or 'short_length_long_length' in k]
    short_to_long_metrics = []
    long_to_short_metrics = []
    for m in metrics:
        acc_name = m.split("/")[-1].split("-")[1]
        a, b = acc_name.split("_")
        a, b = int(a), int(b)
        if a < b:
            print(f" {a=}, {b=} in short to long")
            short_to_long_metrics.append(m)
        elif b < a:
            print(f" {a=}, {b=} in long to short")
            long_to_short_metrics.append(m)

    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'axes.titlesize': 20})
    plt.rcParams.update({'axes.labelsize': 20})
    plt.rcParams.update({'xtick.labelsize': 20})
    for i, ax in enumerate(axs):
        if i == 0: short_to_long, diff = True, False
        elif i == 1: short_to_long, diff = False, False
        else: diff = True

        if not diff:
            if short_to_long:
                acc = df[short_to_long_metrics].mean(axis=1)
                df['subset_accuracy'] = acc
            else:
                acc = df[long_to_short_metrics].mean(axis=1)
                df['subset_accuracy'] = acc
        else:
            df['subset_accuracy'] = df[short_to_long_metrics].mean(axis=1) - df[long_to_short_metrics].mean(axis=1)

        idx = df.groupby(
            ["state_size", "Model"]
        )['subset_accuracy'].idxmax(skipna=True).dropna()
        plot_df = df.loc[idx]
        plot_df['size_label'] = plot_df[state_key].astype(str) + ", " + plot_df['model.d_model'].astype(str)
        print(f"Max state size: {plot_df['state_size'].max()}")
        
        sns.set_theme(style="whitegrid")
        g = sns.lmplot(
            data=plot_df, x="log_state_size", y="subset_accuracy", hue="Model",
            legend=True
        )

        # Set font size to 16
        for item in g.ax.get_xticklabels(): item.set_fontsize(16)
        for item in g.ax.get_yticklabels(): item.set_fontsize(16)

        # In the legend, replace the following labels
        handles, labels = g.ax.get_legend_handles_labels()
        replacements = {
            "BASED Causal: False": "Based Non-Casual",
            "BASED Causal: True": "Based Casual",
        }
        new_labels = [replacements.get(label, label) for label in labels]
        g.ax.legend(handles, new_labels, title="", fontsize=16)
        

        # Set labels
        if not diff: title=f"{'Short A to Long B' if short_to_long else 'Long A to Short B'}"
        else: title = "Difference (Short A - Long A)"
        g.set(ylabel="Accuracy", xlabel="Log (State Size)", title=title,)
        g.ax.set_title(title, fontsize=16)
        g.set_axis_labels("Log (State Size)", "Accuracy", fontsize=16)

        # Add a best fit line
        if not diff: ax.set_title(f"{'Short A to Long B' if short_to_long else 'Long A to Short B'}")
        else: ax.set_title("Difference (Short A - Long A)")
        ax.set_xlabel("State Size")
        ax.set_ylabel("Accuracy")
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)    

        # Save the plots
        out_path = f"{i}_synthetic_plot.png"
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved!")

        # Save the df
        out_path = f"{tag}{i}_synthetic_df.csv"
        plot_df.to_csv(out_path)


if __name__ == "__main__" :
    tag = ''
    launch_id=[
            # 0508: Original runs
            "default-2024-05-08-18-30-37",
            "default-2024-05-08-18-31-29",
            # "default-2024-05-09-01-26-53",
            # "default-2024-05-09-03-49-26",
        ]

    # tag = 'new_'
    # launch_id=[
    #         # 0610: Code release runs on new repo
    #         "default-2024-06-07-03-32-47",
    #         "default-2024-06-10-06-48-38",
    #     ]

    # More seeds on the old repo
    # tag = 'new2_'
    # launch_id=[
    #         "default-2024-06-13-00-53-34", # seed 123
    #         "default-2024-06-12-16-37-01", # seed 123
    #         # "default-2024-06-13-11-14-14", # seed 2
    #         # "default-2024-06-13-13-39-20", # seed 2
    #         # "default-2024-06-13-14-35-30", # seed 10 causal
    #         # "default-2024-06-13-20-57-48", # seed 10 non-causal
    #     ]
    
    # Runs at 2 layers
    # tag = 'new3_'
    # launch_id = [
    #     "default-2024-06-13-16-34-45",
    #     "default-2024-06-13-19-50-56"
    # ]

    # Runs at 6 layers
    # tag = 'new4_'
    # launch_id = [
    #     'default-2024-06-14-14-45-55',
    #     'default-2024-06-14-14-40-31',
    #     # 'default-2024-06-14-19-00-15',
    # ]

    launch_id = [
        "default-2024-06-15-11-16-51",  # smaller vocab (1600)
        "default-2024-06-15-11-17-32",

        # "default-2024-06-15-15-22-50",  # larger vocab (4096)
        # "default-2024-06-15-15-22-54"
    ]

    # launch_id = [
    #     # "default-2024-06-17-16-06-23", # (causal) improved precision
    #     # "default-2024-06-17-16-05-09", # (non-causal)

    #     # "default-2024-06-15-15-22-54",  # (causal)
    #     # "default-2024-06-15-15-22-50",  # (non-causal) larger vocab (4096)

    #     # "default-2024-06-17-20-21-25",
    #     # "default-2024-06-17-20-21-37",
    # ]

    launch_id = [
        "default-2024-06-18-00-22-15",
        "default-2024-06-18-15-14-28",

        # "default-2024-06-18-00-23-26",
        # "default-2024-06-15-15-22-54",  # (causal)
        # "default-2024-06-18-00-25-47",
    ]

    launch_id = [
        "default-2024-06-20-17-04-21",
        "default-2024-06-20-17-04-03",
    ]


    df = fetch_wandb_runs(
        launch_id=launch_id,
        project_name="zoology",
    )
    print(f"Found {len(df)} runs")
    plot(df, tag=tag)
