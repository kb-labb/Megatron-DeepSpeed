#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=mega-pretrain
#SBATCH --mem=40G
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --cpus-per-gpu=4
#XSBATCH --ntasks=16
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/pretrain.log

deactivate
module purge

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
PROJECT="/ceph/hpc/home/eurobink/group_space/robin/workspace/Megatron-DeepSpeed"
LOGGING="$PROJECT/logs"
LOGFILE="${LOGGING}/%x_${DATETIME}.log"

export NPROC_PER_NODE=4
export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=6007
export NNODES=2
# export NODE_RANK=0
# WORLD_SIZE=16

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs


#DATASET_1="<PATH TO THE FIRST DATASET>"
#DATASET_2="<PATH TO THE SECOND DATASET>"
#DATASET_3="<PATH TO THE THIRD DATASET>"
#DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"

BASE_DATA_PATH=$PROJECT/data
DATASET=${BASE_DATA_PATH}/wiki_text_document
VOCAB_PATH=${BASE_DATA_PATH}/tokenizer_dir/vocab.json
MERGE_PATH=${BASE_DATA_PATH}/tokenizer_dir/merges.txt
CHECKPOINT_PATH=$PROJECT/ckpt


CONFIG_JSON="$BASE_DATA_PATH/ds_config.json"
echo $CONFIG_JSON

USE_DEEPSPEED=1
ZERO_STAGE=0


# Debug
TP=4
PP=2
LAYERS=8
HIDDEN=512
SEQ=1024
GLOBAL_BATCH=128
WORKER_STR="-i worker-0"


# 52B
#TP=4
#PP=16
#HIDDEN=8192
#LAYERS=64
#SEQ=1024
#GLOBAL_BATCH=1024
#WORKER_STR=""

MICRO_BATCH=4

options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --num-attention-heads 32 \
        --seq-length $SEQ \
        --loss-scale 12 \
        --max-position-embeddings $SEQ \
	--micro-batch-size $MICRO_BATCH \
	--global-batch-size $GLOBAL_BATCH \
	--train-iters 500 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --lr 6.0e-5 \
	--min-lr 6.0e-6 \
        --lr-decay-style cosine \
        --log-interval 100 \
        --eval-iters 40 \
        --eval-interval 200 \
	--data-path ${DATASET} \
	--vocab-file ${VOCAB_PATH} \
	--merge-file ${MERGE_PATH} \
	--save-interval 100 \
        --split 98,2,0 \
        --clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
        --fp16 \
	--checkpoint-activations \
        "


if [[ ${USE_DEEPSPEED} -eq 1 ]]; then
	echo "Using DeepSpeed"
	options="${options} \
		--deepspeed \
		--deepspeed_config=${CONFIG_JSON} \
		--zero-stage=${ZERO_STAGE} \
		--deepspeed-activation-checkpointing \
	"
fi


cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 100,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "gradient_clipping": 1.0,
  "prescale_gradients": true,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

#run_cmd="deepspeed -i worker-0:0,1,2,3 ${DIR}/pretrain_gpt.py $@ ${options}"
#run_cmd="deepspeed -i worker-0 ${DIR}/pretrain_gpt.py $@ ${options}"
# run_cmd="deepspeed $WORKER_STR ${DIR}/pretrain_gpt.py $@ ${options}"

DISTRIBUTED_ARGS="--nproc_per_node \$NPROC_PER_NODE --nnodes \$SLURM_JOB_NUM_NODES --node_rank \$SLURM_NODEID --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT"
run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
                ./pretrain_gpt.py \
                        ${options}"


cat <<EOT > distributed_runner.sh
#!/bin/bash
/bin/hostname -s
echo $DISTRIBUTED_ARGS
$run_cmd
exit 0
EOT

echo ${run_cmd}
# eval ${run_cmd}

run_cmd="bash distributed_runner.sh"

srun --mpi=pmix -l -o $LOGFILE singularity exec --nv -B $(pwd) /ceph/hpc/home/eurobink/group_space/containers/megatron-deepspeed.sif ${run_cmd}

set +x
exit 0