cmd="python train_meta_na.py --dataset=sims --pretrained=bert-base-chinese 
    --input_dim_a=25 --input_dim_l=768 --input_dim_v=177
    --lr=0.0003 --alpha_lr=0.0003 --vision_weight=2 --max_shots_sup=1024 --q_sz=256"
echo -e "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
eval $cmd