#!/bin/bash

# Runs `run.py` on a GPU node through Slurm. Every argument is passed through to run.py:
#   ./run.sh kvcache --init_from=gpt2-large
#   ./run.sh speculative
#   ./run.sh sweep --rates 1,2,4 --batch_sizes 1,4
#   ./run.sh plot                      # doesn't need a GPU, so runs on this machine
# Set TIME (default 00:10:00) for longer runs, e.g. TIME=00:30:00 ./run.sh sweep

usage() {
    echo "Usage: $0 (kvcache | speculative | serve | sweep | plot) [run.py options]"
    echo "Run '$0 <command> --help' to see each command's options."
    exit 1
}

case "$1" in
    kvcache|speculative|serve|sweep|plot) ;;
    *) usage ;;
esac

# Make sure we are in the same directory as the script
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR" || { echo "Error: could not cd to $SCRIPT_DIR" >&2; exit 1; }

EXEC_CMD=(uv run run.py "$@")
echo "> ${EXEC_CMD[*]}"
unset LD_LIBRARY_PATH # changes made for project 3 interferes with Python virtual env searching for CUDA dependencies

# Plotting and --help don't need a GPU
if [[ "$1" == "plot" || " $* " == *" --help "* || " $* " == *" -h "* ]]; then
    exec "${EXEC_CMD[@]}"
fi

# Pick a GPU that fits the model (run.py's default model depends on the command)
MODEL="gpt2-large"
[[ "$1" == "serve" || "$1" == "sweep" ]] && MODEL="gpt2"
prev=""
for arg in "$@"; do
    [[ "$arg" == --init_from=* ]] && MODEL="${arg#*=}"
    [[ "$prev" == "--init_from" ]] && MODEL="$arg"
    prev="$arg"
done

GPU=(--gres=gpu:1)
if [[ "$MODEL" == "gpt2-large" ]]; then
    GPU=(--gres=gpu:geforce_gtx_2080_ti:1)
elif [[ "$MODEL" == "gpt2-xl" ]]; then
    GPU=(--gres=gpu:1 --constraint=rtx_a6000\|l40)
fi

srun -u --ntasks=1 --partition=gpus "${GPU[@]}" --time="${TIME:-00:10:00}" "${EXEC_CMD[@]}"
