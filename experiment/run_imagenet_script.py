import os

def cmd_assert(cmd):
    assert(cmd ==0)

tol = [
    #0,
    1e-2,
    #5e-2,
    #6e-2,
    #7e-2,
    #8e-2,
    #9e-2,
    #1e-1
]

finetune = [False] #True]

model = [
    #'timm/resnet50.tv_in1k',
    'facebook/convnext-tiny-224',
    #'microsoft/swin-tiny-patch4-window7-224',
]

cmd = "ACCELERATE_LOG_LEVEL=INFO TRANSFORMERS_OFFLINE=1 accelerate launch run_imagenet.py " \
      "--model_name_or_path {} " \
      "--train_dir /mnt/data/imagenet/train "\
      "--validation_dir /mnt/data/imagenet/val "\
      "--per_device_train_batch_size 128 " \
      "--per_device_eval_batch_size 128 " \
      "--gradient_accumulation_steps 4 " \
      "--learning_rate 0 " \
      "--num_train_epochs 0 " \
      "--output_dir {} " \
      "--bin_svd_tol {} "

path = "/mnt/runs/exp/imagenet_{}_{}_get_graph"

for m in model:
    for e in tol:
        dir_path = path.format('_'.join(m.split('/')), e)
        cmd_patt = cmd
        for F in finetune:
            if F:
                dir_path += '_F'
                cmd_patt += '--bin_svd_finetune --bin_svd_eta 3'
            run_cmd = cmd_patt.format(m, dir_path, e) + " 2>&1 | tee {}/log.txt".format(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            os.system("rm -r {}/*".format(dir_path))
            print(run_cmd)
            cmd_assert(os.system(run_cmd))
            os.system("rm {}/pytorch_model.bin".format(dir_path))
            cmd_assert(os.system("cat {}/all_results.json >> {}/log.txt".format(dir_path, dir_path)))