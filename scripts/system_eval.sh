LOG_FILE_RANK="results/system_eval_rank.txt"
python system_eval.py --memory 50 --num_threads 10 --model-dir output --data_path data/data_1l.csv --rank 1 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_RANK
python system_eval.py --memory 50 --num_threads 10 --model-dir output --data_path data/data_1l.csv --rank 5 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_RANK
python system_eval.py --memory 50 --num_threads 10 --model-dir output --data_path data/data_1l.csv --rank 10 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_RANK
python system_eval.py --memory 50 --num_threads 10 --model-dir output --data_path data/data_1l.csv --rank 20 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_RANK
python system_eval.py --memory 50 --num_threads 10 --model-dir output --data_path data/data_1l.csv --rank 40 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_RANK
python system_eval.py --memory 50 --num_threads 10 --model-dir output --data_path data/data_1l.csv --rank 100 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_RANK



LOG_FILE_THREAD="results/system_eval_threads.txt"
python system_eval.py --memory 50 --num_threads 1 --model-dir output --data_path data/data_1l.csv --rank 100 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_THREAD
python system_eval.py --memory 50 --num_threads 5 --model-dir output --data_path data/data_1l.csv --rank 100 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_THREAD
python system_eval.py --memory 50 --num_threads 10 --model-dir output --data_path data/data_1l.csv --rank 100 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_THREAD
python system_eval.py --memory 50 --num_threads 15 --model-dir output --data_path data/data_1l.csv --rank 100 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_THREAD
python system_eval.py --memory 50 --num_threads 20 --model-dir output --data_path data/data_1l.csv --rank 100 --max_iter 10 --reg_param 0.5 >> $LOG_FILE_THREAD

# python system_eval.py --memory 50 --num_threads 20 --model-dir output --data_path data/data_1l.csv --rank 100 --max_iter 10 --reg_param 0.5