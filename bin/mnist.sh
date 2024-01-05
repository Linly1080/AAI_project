cuda=0
src_dataset=MNIST_02468_01234_5
tar_dataset=MNIST_13579_01234_5

python src/main.py \
    --cuda 2 \
    --src_dataset MNIST_02468_01234_5 \
    --tar_dataset MNIST_13579_01234_5 \
    --transfer_ebd \
    --lr 0.001 \
    --weight_decay 0.01 \
    --patience 5

python src/main.py --cuda 2 --src_dataset MNIST_02468_0123456789_10 --tar_dataset MNIST_13579_0123456789_10 --transfer_ebd --lr 0.001 --weight_decay 0.01 --patience 5
