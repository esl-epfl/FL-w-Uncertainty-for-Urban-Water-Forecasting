if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

target_node=WWTP_total

# for data_version in Lausanne #Crissier Mont-sur-Lausanne
# do
#     if [ ! -d "./logs/Hydrique_Lausanne_"$data_version ]; then
#         mkdir ./logs/Hydrique_Lausanne_$data_version
#     fi
    
#     if [ ! -d "./logs/Hydrique_Lausanne_"$data_version"/"$target_node ]; then
#         mkdir ./logs/Hydrique_Lausanne_"$data_version"/$target_node
#     fi
#     seq_len=144
#     model_name=AquaCast

#     root_path_name=/home/abgo/Data/Hydrique_Lausanne/data/processed/
#     endo_path=Hydrique_1_1_"$data_version".csv                      # e.g. 'ETTh1.csv' but with [date, Node1, Node2, ...]
#     # exo_measured_path=measure_timeseries_dt_measure.csv      # columns: [time, value]
#     exo_forecast_path=Hydrique_"$data_version"_forecast.csv      # columns: [issued_at, forecast_time, value]
#     model_id_name=Hydrique_real_forecast_"$data_version"
#     data_name=custom_real

#     features=M
#     enc_in=2 # one endo and one exo

#     # pred_len=96

#     random_seed=2025

#     e_layers=3
#     n_heads=2
#     d_model=32
#     d_ff=128
#     for pred_len in 6 12 18 24 30 36 42 48 54 60 66 72
#     do
#         python -u run_longExp.py \
#         --random_seed $random_seed \
#         --is_training 0 \
#         --root_path $root_path_name \
#         --data_path $endo_path $exo_forecast_path\
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
#         --learning_rate 0.00005\
#         --train_epochs 100\
#         --patience 10\
#         --gpu 0\
#         --itr 1 --batch_size 32 >logs/Hydrique_Lausanne_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_exo_exo_future_dm'$d_model.log
#     done
# done

# for data_version in Lausanne #Crissier Mont-sur-Lausanne
# do
#     if [ ! -d "./logs/Hydrique_Lausanne_"$data_version ]; then
#         mkdir ./logs/Hydrique_Lausanne_$data_version
#     fi
    
#     if [ ! -d "./logs/Hydrique_Lausanne_"$data_version"/"$target_node ]; then
#         mkdir ./logs/Hydrique_Lausanne_"$data_version"/$target_node
#     fi
#     seq_len=144
#     model_name=AquaCast

#     root_path_name=/home/abgo/Data/Hydrique_Lausanne/data/processed/
#     endo_path=Hydrique_1_1_"$data_version".csv                      # e.g. 'ETTh1.csv' but with [date, Node1, Node2, ...]
#     # exo_measured_path=measure_timeseries_dt_measure.csv      # columns: [time, value]
#     # exo_forecast_path=Hydrique_"$data_version"_forecast.csv      # columns: [issued_at, forecast_time, value]
#     model_id_name=Hydrique_perfect_forecast_"$data_version"
#     data_name=custom_exo

#     features=M
#     enc_in=2

#     # pred_len=96

#     random_seed=2025

#     e_layers=3
#     n_heads=2
#     d_model=32
#     d_ff=128
#     for pred_len in 6 12 18 24 30 36 42 48 54 60 66 72
#     do
#         python -u run_longExp.py \
#         --random_seed $random_seed \
#         --is_training 0 \
#         --root_path $root_path_name \
#         --data_path $endo_path \
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
#         --learning_rate 0.00005\
#         --train_epochs 100\
#         --patience 10\
#         --gpu 0\
#         --itr 1 --batch_size 32 >logs/Hydrique_Lausanne_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_exo_exo_future_dm'$d_model.log
#     done
# done

for data_version in 1_0 
do
    if [ ! -d "./logs/Hydrique_Lausanne_"$data_version ]; then
        mkdir ./logs/Hydrique_Lausanne_$data_version
    fi
    
    if [ ! -d "./logs/Hydrique_Lausanne_"$data_version"/"$target_node ]; then
        mkdir ./logs/Hydrique_Lausanne_"$data_version"/$target_node
    fi
    seq_len=144
    model_name=AquaCast

    root_path_name=/home/abgo/Data/Hydrique_Lausanne/data/processed/
    endo_path=Hydrique_"$data_version".csv                      # e.g. 'ETTh1.csv' but with [date, Node1, Node2, ...]
    # exo_measured_path=measure_timeseries_dt_measure.csv      # columns: [time, value]
    # exo_forecast_path=Hydrique_"$data_version"_forecast.csv      # columns: [issued_at, forecast_time, value]
    model_id_name=Hydrique_no_rain_"$data_version"
    data_name=custom

    features=M
    enc_in=1

    # pred_len=96

    random_seed=2025

    e_layers=3
    n_heads=2
    d_model=32
    d_ff=128
    for pred_len in 6 12 18 24 30 36 #42 48 54 60 66 ##72
    do
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 0 \
        --root_path $root_path_name \
        --data_path $endo_path \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features $features \
        --target $target_node \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --no-exo \
        --no-exo_future \
        --e_layers $e_layers \
        --n_heads $n_heads \
        --d_model $d_model \
        --d_ff $d_ff \
        --dropout 0.5\
        --stride 2\
        --des 'Exp_exo' \
        --lradj 'constant'\
        --learning_rate 0.00005\
        --train_epochs 100\
        --patience 10\
        --gpu 0\
        --itr 1 --batch_size 32 >logs/Hydrique_Lausanne_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_exo_exo_future_dm'$d_model.log
    done
done

# # Fine tune parameters for best models from above experiments in perfect forecast setting
# if [ ! -d "./logs_pt" ]; then
#     mkdir ./logs_pt
# fi

# for data_version in Lausanne #Crissier Mont-sur-Lausanne
# do
#     if [ ! -d "./logs_pt/Hydrique_Lausanne_"$data_version ]; then
#         mkdir ./logs_pt/Hydrique_Lausanne_$data_version
#     fi
    
#     if [ ! -d "./logs_pt/Hydrique_Lausanne_"$data_version"/"$target_node ]; then
#         mkdir ./logs_pt/Hydrique_Lausanne_"$data_version"/$target_node
#     fi
#     seq_len=144
#     model_name=AquaCast

#     root_path_name=/home/abgo/Data/Hydrique_Lausanne/data/processed/
#     endo_path=Hydrique_1_1_"$data_version".csv                      # e.g. 'ETTh1.csv' but with [date, Node1, Node2, ...]
#     # exo_measured_path=measure_timeseries_dt_measure.csv      # columns: [time, value]
#     # exo_forecast_path=Hydrique_"$data_version"_forecast.csv      # columns: [issued_at, forecast_time, value]
#     model_id_name=Hydrique_perfect_forecast_"$data_version"
#     data_name=custom_exo

#     features=M
#     enc_in=2

#     pred_len=30

#     random_seed=2025

#     e_layers=3
#     n_heads=2
#     d_model=32
#     # d_ff=128
#     for d_ff in 128 256
#     do
#         for lr in 0.001 0.005
#         do
#             python -u run_longExp.py \
#             --random_seed $random_seed \
#             --is_training 1 \
#             --root_path $root_path_name \
#             --data_path $endo_path \
#             --model_id $model_id_name'_'$seq_len'_'$pred_len \
#             --model $model_name \
#             --data $data_name \
#             --features $features \
#             --target $target_node \
#             --seq_len $seq_len \
#             --pred_len $pred_len \
#             --enc_in $enc_in \
#             --exo \
#             --exo_future \
#             --e_layers $e_layers \
#             --n_heads $n_heads \
#             --d_model $d_model \
#             --d_ff $d_ff \
#             --dropout 0.5\
#             --stride 2\
#             --des 'Exp_exo' \
#             --lradj 'constant'\
#             --learning_rate $lr\
#             --train_epochs 100\
#             --patience 10\
#             --gpu 0\
#             --itr 1 --batch_size 32 >logs_pt/Hydrique_Lausanne_"$data_version"/$target_node/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_ftr'$features'_exo_exo_future_dm'$d_model'_dff'$d_ff'_lr'$lr.log
#         done
#     done
# done