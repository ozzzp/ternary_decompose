import os

def cmd_assert(cmd):
    assert(cmd ==0)


task = {
    "cola":3,
    "mnli":3,
    "mrpc":5,
    "qnli": 3,
    "qqp": 3,
    "rte": 3,
    "sst2": 3,
    "stsb": 3,
    #"wnli":5
}

tol = [
    #0,
    1e-2,
    5e-2,
    #6e-2,
    #7e-2,
    #8e-2,
    #9e-2,
    #1e-1
]

finetune = [False]

cmd = "ACCELERATE_LOG_LEVEL=INFO TRANSFORMERS_OFFLINE=1 accelerate launch run_glue.py " \
      "--model_name_or_path bert-base-cased " \
      "--task_name {} " \
      "--max_length 128 " \
      "--per_device_train_batch_size 5 " \
      "--learning_rate 2e-5 " \
      "--num_train_epochs {} " \
      "--output_dir {} " \
      "--pad_to_max_length " \
      "--bin_svd_tol {} "

path = "/mnt/runs/exp/glue_bert_{}_{}"

for e in tol:
    for name, epochs in task.items():
        dir_path = path.format(name, e)
        cmd_patt = cmd
        for F in finetune:
            if F:
                dir_path += '_F'
                cmd_patt += '--bin_svd_finetune --bin_svd_eta 3'
            run_cmd = cmd_patt.format(name, epochs, dir_path, e) + " 2>&1 | tee {}/log.txt".format(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            os.system("rm -r {}/*".format(dir_path))
            print(run_cmd)
            cmd_assert(os.system(run_cmd))
            cmd_assert(os.system("rm {}/pytorch_model.bin".format(dir_path)))
            cmd_assert(os.system("cat {}/all_results.json >> {}/log.txt".format(dir_path, dir_path)))