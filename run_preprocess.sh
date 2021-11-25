#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=mega-prepro
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --nodes=1
#XSBATCH --ntasks=16
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/prepro.log

# DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
PROJECT="/ceph/hpc/home/eurobink/group_space/robin/workspace/Megatron-DeepSpeed"
LOGGING="$PROJECT/logs"
LOGFILE="${LOGGING}/%x_${DATETIME}.log"

deactivate
module load PyTorch/1.7.1-fosscuda-2020b
source $PROJECT/.env/bin/activate

INPUT_JSON_FILE="data/wiki.sv.json"
OUTPUT_PATH="data/wiki"
VOCAB_FILE="tokenizer_dir/vocab.json"
MERGE_FILE="tokenizer_dir/merges.txt"
NUM_CPUS=16

echo $SLURM_JOB_NAME
echo $SLURM_JOB_ID
echo $SLURM_JOB_NODELIST
echo $SLURM_JOB_NUM_NODES
echo $SLURM_LOCALID
echo $SLURM_NODEID
echo $SLURM_PROCID

cmd="python ./tools/preprocess_data.py \
                       --input $INPUT_JSON_FILE \
                       --output-prefix $OUTPUT_PATH \
                       --json-keys text \
                       --vocab-file $VOCAB_FILE \
                       --merge-file $MERGE_FILE \
                       --dataset-impl mmap \
                       --tokenizer-type GPT2BPETokenizer \
                       --workers $NUM_CPUS \
                       --append-eod \
                       "

srun -l -o $LOGFILE $cmd