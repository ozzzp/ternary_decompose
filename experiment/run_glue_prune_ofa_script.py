import os

def cmd_assert(cmd):
    assert(cmd ==0)


task = {
    "mrpc":5,
    "cola":3,
    "mnli":3,
    "qnli": 3,
    "qqp": 3,
    "rte": 3,
    "sst2": 3,
    "stsb": 3,
    #"wnli":5
}

cmd = "ACCELERATE_LOG_LEVEL=INFO TRANSFORMERS_OFFLINE=1 accelerate launch run_glue_prune_ofa.py " \
      "--model_name_or_path Intel/bert-base-uncased-sparse-{}-unstructured-pruneofa " \
      "--task_name {} " \
      "--max_length 128 " \
      "--per_device_train_batch_size 5 " \
      "--learning_rate 2e-5 " \
      "--num_train_epochs {} " \
      "--output_dir {} " \
      "--pad_to_max_length "

path = "/mnt/runs/exp/glue_bert_ofa_{}_{}"

sparse = [90]

for s in sparse:
    for name, epochs in task.items():
        dir_path = path.format(s, name)
        run_cmd = cmd.format(s, name, epochs, dir_path) + " 2>&1 | tee {}/log.txt".format(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        os.system("rm -r {}/*".format(dir_path))
        print(run_cmd)
        cmd_assert(os.system(run_cmd))
        os.system("rm {}/pytorch_model.bin".format(dir_path))
        cmd_assert(os.system("cat {}/all_results.json >> {}/log.txt".format(dir_path, dir_path)))