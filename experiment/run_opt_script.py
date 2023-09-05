import os

def cmd_assert(cmd):
    assert(cmd ==0)

tol = [
    #4e-2,
    #3e-2,
    1e-2,
    2e-2,
    1.5e-2,
    #0,
    #7.5e-3,
    #5e-3
]

cmd = "TRANSFORMERS_OFFLINE=1 " \
      "python -u run_opt.py facebook/opt-6.7b c4 " \
      "--bin_svd_tol {} 2>&1 |tee opt_6.7B_{}_C4.log"


for e in tol:
    run_cmd = cmd.format(e, e)
    print(run_cmd)
    os.system(run_cmd)