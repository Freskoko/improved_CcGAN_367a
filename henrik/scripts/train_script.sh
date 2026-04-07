mkdir data/my_dataset
mv my_dataset.h5 data/my_dataset/

export DATA_ROOT=$(pwd)/data
export SAVE_PATH=$(pwd)/checkpoints

python train.py \
    --dataset my_dataset \
    --data_root $DATA_ROOT \
    --epochs 200 \
    --batch_size 32 \
    --save_path $SAVE_PATH
