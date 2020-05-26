python  ./src/run_squad.py \
	--model_type bert \
	--do_eval \
	--model_name_or_path $1 \
	--tokenizer_name bert-base-chinese \
	--predict_file $2 \
	--max_seq_length 384 \
	--per_gpu_eval_batch_size 50 \
	--output_dir ./ \

python ./src/process_ans.py ./predictions_.json $3
rm ./predictions_.json