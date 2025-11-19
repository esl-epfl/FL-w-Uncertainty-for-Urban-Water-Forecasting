if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

data_version=0

for target_node in STEP_Debit_entree_total #Hydrique_hydrique-Vuachere_Vallon_hauteur STEP_DВbitcanalEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09 STEP_DВbitcanalOUEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09
do
    if [ ! -d "./logs/Hydrique_Lausanne_"$data_version ]; then
        mkdir ./logs/Hydrique_Lausanne_$data_version
    fi
    
    if [ ! -d "./logs/Hydrique_Lausanne_"$data_version"/"$target_node ]; then
        mkdir ./logs/Hydrique_Lausanne_"$data_version"/$target_node
    fi
    seq_len=96
    model_name=AquaCast

    root_path_name=/home/abgo/Data/Hydrique_Lausanne/code/
    endo_path=Hydrique_WWTP_total.csv                       # e.g. 'ETTh1.csv' but with [date, Node1, Node2, ...]
    exo_measured_path=measure_timeseries_dt_measure.csv      # columns: [time, value]
    exo_forecast_path=measure_timeseries_dt_forecast.csv      # columns: [issued_at, forecast_time, value]
    data_name=custom_real

    features=M
    enc_in=2

    # pred_len=96

    random_seed=2025

    e_layers=3
    n_heads=2
    d_model=32
    d_ff=128
    for pred_len in 96 #192 480 720
    do
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $endo_path \
        --exo_measured_path $exo_measured_path \
        --exo_forecast_path $exo_forecast_path \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features $features \
        --target $target_node \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --exo \
        --exo_future \
        --e_layers $e_layers \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff $d_ff \
        --dropout 0.5\
        --stride 2\
        --des 'Exp_exo' \
        --lradj 'constant'\
        --learning_rate 0.0001\
        --train_epochs 100\
        --patience 10\
        --gpu 0\
        --itr 1 --batch_size 64 >logs/Hydrique_Lausanne_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_exo_exo_future_dm'$d_model.log
    done
done

# data_version=1_3_1

# for target_node in STEP_Debit_entree_total #Hydrique_hydrique-Vuachere_Vallon_hauteur STEP_DВbitcanalEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09 STEP_DВbitcanalOUEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09
# do
#     if [ ! -d "./logs/Hydrique_Lausanne_"$data_version ]; then
#         mkdir ./logs/Hydrique_Lausanne_$data_version
#     fi
    
#     if [ ! -d "./logs/Hydrique_Lausanne_"$data_version"/"$target_node ]; then
#         mkdir ./logs/Hydrique_Lausanne_"$data_version"/$target_node
#     fi
#     seq_len=96
#     model_name=AquaCast

#     root_path_name=/home/abgo/Data/LausanneCity/
#     data_path_name=Hydrique_Lausanne_v"$data_version".csv
#     model_id_name=Hydrique_Lausanne_v"$data_version"
#     data_name=custom_exo

#     features=M
#     enc_in=5

#     # pred_len=96

#     random_seed=2025

#     e_layers=3
#     n_heads=2
#     d_model=32
#     d_ff=128
#     for pred_len in 96 192 480 720
#     do
#         python -u run_longExp.py \
#         --random_seed $random_seed \
#         --is_training 1 \
#         --root_path $root_path_name \
#         --data_path $data_path_name \
#         --model_id $model_id_name'_'$seq_len'_'$pred_len \
#         --model $model_name \
#         --data $data_name \
#         --features $features \
#         --target $target_node \
#         --seq_len $seq_len \
#         --pred_len $pred_len \
#         --enc_in $enc_in \
#         --exo \
#         --exo_future \
#         --e_layers $e_layers \
#         --n_heads $n_heads \
#         --d_model $d_model \
#         --d_ff $d_ff \
#         --dropout 0.5\
#         --stride 2\
#         --des 'Exp_exo' \
#         --lradj 'constant'\
#         --learning_rate 0.0001\
#         --train_epochs 100\
#         --patience 10\
#         --gpu 0\
#         --itr 1 --batch_size 32 >logs/Hydrique_Lausanne_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_exo_exo_future_dm'$d_model.log
#     done
# done

# data_version=1_3_1

# for target_node in STEP_Debit_entree_total #Hydrique_hydrique-Vuachere_Vallon_hauteur STEP_DВbitcanalEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09 STEP_DВbitcanalOUEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09
# do
#     if [ ! -d "./logs/Hydrique_Lausanne_"$data_version ]; then
#         mkdir ./logs/Hydrique_Lausanne_$data_version
#     fi
    
#     if [ ! -d "./logs/Hydrique_Lausanne_"$data_version"/"$target_node ]; then
#         mkdir ./logs/Hydrique_Lausanne_"$data_version"/$target_node
#     fi
#     seq_len=96
#     model_name=AquaCast

#     root_path_name=/home/abgo/Data/LausanneCity/
#     data_path_name=Hydrique_Lausanne_v"$data_version".csv
#     model_id_name=Hydrique_Lausanne_v"$data_version"
#     data_name=custom

#     features=M
#     enc_in=5

#     # pred_len=96

#     random_seed=2025

#     e_layers=3
#     n_heads=2
#     d_model=32
#     d_ff=128
#     for pred_len in 96 192 480 720
#     do
#         python -u run_longExp.py \
#         --random_seed $random_seed \
#         --is_training 1 \
#         --root_path $root_path_name \
#         --data_path $data_path_name \
#         --model_id $model_id_name'_'$seq_len'_'$pred_len \
#         --model $model_name \
#         --data $data_name \
#         --features $features \
#         --target $target_node \
#         --seq_len $seq_len \
#         --pred_len $pred_len \
#         --enc_in $enc_in \
#         --exo \
#         --no-exo_future \
#         --e_layers $e_layers \
#         --n_heads $n_heads \
#         --d_model $d_model \
#         --d_ff $d_ff \
#         --dropout 0.5\
#         --stride 2\
#         --des 'Exp_exo' \
#         --lradj 'constant'\
#         --learning_rate 0.0001\
#         --train_epochs 100\
#         --patience 10\
#         --gpu 0\
#         --itr 1 --batch_size 32 >logs/Hydrique_Lausanne_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_exo_no-exo_future_dm'$d_model.log
#     done
# done


# data_version=1_2

# for target_node in STEP_Debit_entree_total #Hydrique_hydrique-Vuachere_Vallon_hauteur STEP_DВbitcanalEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09 STEP_DВbitcanalOUEST__A107_A_2FAB10_CF001_71-data-2024-12-0612_16_09
# do
#     if [ ! -d "./logs/Hydrique_Lausanne_"$data_version ]; then
#         mkdir ./logs/Hydrique_Lausanne_$data_version
#     fi
    
#     if [ ! -d "./logs/Hydrique_Lausanne_"$data_version"/"$target_node ]; then
#         mkdir ./logs/Hydrique_Lausanne_"$data_version"/$target_node
#     fi
#     seq_len=96
#     model_name=AquaCast

#     root_path_name=/home/abgo/Data/LausanneCity/
#     data_path_name=Hydrique_Lausanne_v"$data_version".csv
#     model_id_name=Hydrique_Lausanne_v"$data_version"
#     data_name=custom

#     features=M
#     enc_in=4

#     pred_len=96

#     random_seed=2025

#     e_layers=3
#     n_heads=2
#     d_model=32
#     d_ff=128
#     for pred_len in 96 192 480 720
#     do
#         python -u run_longExp.py \
#         --random_seed $random_seed \
#         --is_training 1 \
#         --root_path $root_path_name \
#         --data_path $data_path_name \
#         --model_id $model_id_name'_'$seq_len'_'$pred_len \
#         --model $model_name \
#         --data $data_name \
#         --features $features \
#         --target $target_node \
#         --seq_len $seq_len \
#         --pred_len $pred_len \
#         --enc_in $enc_in \
#         --no-exo \
#         --no-exo_future \
#         --e_layers $e_layers \
#         --n_heads $n_heads \
#         --d_model $d_model \
#         --d_ff $d_ff \
#         --dropout 0.5\
#         --stride 2\
#         --des 'Exp_exo' \
#         --lradj 'constant'\
#         --learning_rate 0.0001\
#         --train_epochs 100\
#         --patience 10\
#         --gpu 0\
#         --itr 1 --batch_size 32 >logs/Hydrique_Lausanne_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_no-exo_no-exo_future_dm'$d_model.log
#     done
# done
