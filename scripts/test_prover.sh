python run_experiment_prover.py --model_type roberta_rr --model_name_or_path ./output/prover_all --task_name rr --do_eval --do_lower_case --data_dir ./data/depth-5 --max_seq_length 300 --per_gpu_eval_batch_size 8 --per_gpu_train_batch_size 1 --learning_rate 1e-5 --num_train_epochs 5 --output_dir ./output/prover_all --logging_steps 4752 --save_steps 4750 --seed 42 --data_cache_dir ./output/cache/ --evaluate_during_training