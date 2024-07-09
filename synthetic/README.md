The following instructions are for the set disjointness synthetic in our work. 

To get setup:
```bash
git submodule init
git submodule update
cd zoology/
pip install -e .
```

Run the following. Note that the ```-p``` leads the sweep to parallelize across your GPUs. If you're debugging / don't want to parallelize, remove the flag. 
```bash
# causal based
python zoology/launch.py zoology/experiments/causal.py -p
python zoology/launch.py zoology/experiments/causal_seed2.py -p

# non-causal based
python zoology/launch.py zoology/experiments/non_causal.py -p
python zoology/launch.py zoology/experiments/non_causal_seed2.py -p
```

Plotting the figure. We use WandB to help pull down metrics and plot. To use this method, find the ```launch_id``` values on WandB for the sweeps you've launched and add them to the list of launch_id's in the following python file. Then run:
```bash
python zoology/experiments/arxiv24_jrt_figure2/plot.py
```