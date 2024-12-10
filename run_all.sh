# echo "=====all=====" >> all.txt
# models=("GCN" "GAT" "Cheb")

# for task in 0; do
#     for model in "${models[@]}"; do
#         echo "===Baseline = ${model}, task = ${task}=====" >> all.txt
#         # python main.py --dataset='bitcoin_alpha' --model="$model" --task=$task --method="none" --batch_size=512 --n_hidden=64 --n_embed=64 --train_ratio=0.2 --val_ratio=0.4 --n_out=32 --mixup=0 >> all.txt
#         # python main.py --dataset='bitcoin_alpha' --model="$model" --task=$task --method="tq" --batch_size=512 --n_hidden=64 --n_embed=64 --train_ratio=0.2 --val_ratio=0.4 --n_out=32 --n=1  --T=2 --f=0.1 --mixup=0  >> all.txt
#         # python main.py --dataset='bitcoin_alpha' --model="$model" --task=$task --method="none" --batch_size=512 --n_hidden=64 --n_embed=64 --train_ratio=0.2 --val_ratio=0.4 --n_out=32 --n=2  --mixup=2 --h=0.5 --k=0.1 >> all.txt
#         python main.py --dataset='bitcoin_alpha' --model="$model" --task=$task --method="tq" --batch_size=512 --n_hidden=64 --n_embed=64 --train_ratio=0.2 --val_ratio=0.4 --n_out=32 --n=1  --T=2 --f=0.1 --mixup=2 --h=0.1 --k=0.1 --wd=9e-6 >> all.txt
#     done
# done 

# for task in 1; do
#     for model in "${models[@]}"; do
#         echo "===Baseline = ${model}, task = ${task}=====" >> all.txt
#         python main.py --dataset='intrusion' --model="$model" --task=$task --method="none" --batch_size=8192 --n_hidden=64 --n_embed=256 --train_ratio=0.2 --val_ratio=0.4 --n_out=128 --mixup=0 >> all.txt
#         python main.py --dataset='intrusion' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=64 --n_embed=256 --train_ratio=0.2 --val_ratio=0.4 --n_out=128 --n=1  --T=2 --f=0.3 --mixup=0  >> all.txt
#         python main.py --dataset='intrusion' --model="$model" --task=$task --method="none" --batch_size=8192 --n_hidden=64 --n_embed=256 --train_ratio=0.2 --val_ratio=0.4 --n_out=128 --n=1  --mixup=2 --h=0.1 --k=0.1 >> all.txt
#         python main.py --dataset='intrusion' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=64 --n_embed=256 --train_ratio=0.2 --val_ratio=0.4 --n_out=128 --n=1  --T=2 --f=0.3 --mixup=2 --h=0.3 --k=0.1 >> all.txt
#     done
# done 

# for task in 0; do
#     for model in "${models[@]}"; do
#         echo "===Baseline = ${model}, task = ${task}=====" >> all.txt
#         python main.py --dataset='ppi' --model="$model" --task=$task --method="none" --epoch=100 --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --label_ratio=1 --mixup=0 >> all.txt
#         python main.py --dataset='ppi' --model="$model" --task=$task --method="tq" --epoch=100 --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --label_ratio=1 --n=1  --T=0.5 --f=0.3  --mixup=0  >> all.txt
#         python main.py --dataset='ppi' --model="$model" --task=$task --method="none" --epoch=100 --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --label_ratio=1 --n=1  --mixup=2 --h=0.3 --k=0.1 >> all.txt
#         python main.py --dataset='ppi' --model="$model" --task=$task --method="tq" --epoch=100 --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --label_ratio=1 --n=1  --T=0.5 --f=0.3  --mixup=2 --h=0.001 --k=0.1 >> all.txt
#     done
# done 


# echo "=====reddit=====" >> reddit.txt
# models=("Cheb" )

# for task in 0 ; do
#     for model in "${models[@]}"; do
#         echo "===Baseline = ${model}, task = ${task}=====" >> reddit.txt
#         # python main.py --dataset='reddit' --model="$model" --task=$task --method="none" --lr=0.05 --batch_size=16384 --n_hidden=512 --n_embed=128 --n_out=32 --mixup=0 >> reddit.txt
#         # python main.py --dataset='reddit' --model="$model" --task=$task --method="none" --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --mixup=0 >> reddit.txt        
#         # python main.py --dataset='reddit' --model="$model" --task=$task --method="qw" --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --n=2  --T=1 --mixup=0 >> reddit.txt
#         # python main.py --dataset='reddit' --model="$model" --task=$task --method="tw" --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --n=1  --T=3  --T=0.2 --mixup=0  >> reddit.txt
#         # python main.py --dataset='reddit' --model="$model" --task=$task --method="tq" --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --n=1  --T=3 --f=0.00001 --mixup=0  >> reddit.txt
#         # python main.py --dataset='reddit' --model="$model" --task=$task --method="none" --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --mixup=1 --h=0.1 --k=0.1 >> reddit.txt
#         # python main.py --dataset='reddit' --model="$model" --task=$task --method="none" --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --mixup=3 --h=0.1 --k=0.1 >> reddit.txt
#         python main.py --dataset='reddit' --model="$model" --task=$task --method="tq" --lr=0.05 --batch_size=16384 --n_hidden=128 --n_embed=128 --n_out=64 --n=2  --T=1 --f=0.1  --mixup=2 --h=0.1 --k=0.1 --wd=5e-4 >> reddit.txt
#     done
# done 

# echo "=====epinions=====" >> epinions.txt
# models=("Cheb")
# # models=("GCN" "GAT" "Cheb")

# for task in 0 ; do
#     for model in "${models[@]}"; do
#         echo "===Baseline = ${model}, task = ${task}=====" >> epinions.txt
#         # python main.py --dataset='epinions' --model="$model" --task=$task --method="none" --batch_size=8192 --n_hidden=1024 --n_out=32 --n_embed=256 --lr=0.001 --mixup=0 >> epinions.txt
#         # python main.py --dataset='epinions' --model="$model" --task=$task --method="qw" --batch_size=8192 --n_hidden=1024 --n_embed=256 --lr=0.001 --n=1  --T=3 --mixup=0 >> epinions.txt
#         # python main.py --dataset='epinions' --model="$model" --task=$task --method="tw" --batch_size=8192 --n_hidden=1024 --n_embed=256 --lr=0.001 --n=1  --T=3 --f=0.1 --mixup=0  >> epinions.txt
#         # python main.py --dataset='epinions' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=1024 --n_embed=256 --lr=0.001 --n=1  --T=3 --f=0.1 --mixup=0  >> epinions.txt
#         # python main.py --dataset='epinions' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=1024 --n_embed=256 --lr=0.001 --n=1  --T=3 --f=0.07 --mixup=0  >> epinions.txt
#         # python main.py --dataset='epinions' --model="$model" --task=$task --method="none" --batch_size=8192 --n_hidden=1024 --n_embed=256 --lr=0.001 --mixup=2 --h=0.5 --k=0.1 >> epinions.txt
#         # python main.py --dataset='epinions' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=1024 --n_embed=256 --lr=0.001 --n=1  --T=3 --f=0.1 --mixup=2 --h=0.5 --k=0.1 --wd=5e-4 >> epinions.txt
#         # python main.py --dataset='epinions' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=1024 --n_embed=256 --lr=0.001 --n=1  --T=3 --f=0.1 --mixup=2 --h=0.5 --k=0.1 --wd=1e-4 >> epinions.txt
#         # python main.py --dataset='epinions' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=1024 --n_embed=256 --lr=0.001 --n=1  --T=3 --f=0.1 --mixup=2 --h=0.5 --k=0.1 --wd=5e-5 >> epinions.txt
#         # python main.py --dataset='epinions' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=1024 --n_embed=256 --lr=0.001 --n=1  --T=3 --f=0.1 --mixup=2 --h=0.5 --k=0.1 --wd=1e-5 >> epinions.txt
#         python main.py --dataset='epinions' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=512 --n_embed=512 --lr=0.001 --n=1  --T=3 --f=0.5 --mixup=2 --h=0.1 --k=0.5 --wd=5e-4 >> epinions.txt
#         python main.py --dataset='epinions' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=512 --n_embed=512 --lr=0.001 --n=1  --T=3 --f=0.5 --mixup=2 --h=0.1 --k=0.5 --wd=1e-4 >> epinions.txt
#         python main.py --dataset='epinions' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=512 --n_embed=512 --lr=0.001 --n=1  --T=3 --f=0.5 --mixup=2 --h=0.1 --k=0.5 --wd=5e-5 >> epinions.txt
#         python main.py --dataset='epinions' --model="$model" --task=$task --method="tq" --batch_size=8192 --n_hidden=512 --n_embed=512 --lr=0.001 --n=1  --T=3 --f=0.5 --mixup=2 --h=0.1 --k=0.5 --wd=1e-5 >> epinions.txt
#     done
# done 

# mag --batch_size=512 --n_hidden=128 --n_embed=128 --out=64 --lr=0.001 