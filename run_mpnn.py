import subprocess


def run_subprocess(cmd):
    sp = subprocess.run(cmd, shell=True)
    if sp.returncode != 0:
        raise Exception('error')


# run task for t1 and st
# training
for task in ['t1', 'st']:
    for seed in [0]:
        cmd = 'python ./models/train_mpnn.py --task ' + str(task) + ' --seed ' + str(seed)
        run_subprocess(cmd)
        
# robustness test
for task in ['t1', 'st']:
    for seed in [49, 97, 53, 5, 33, 65, 62, 51, 38, 61]:
        cmd = 'python ./models/train_mpnn.py --task ' + str(task) + ' --seed ' + str(seed)
        run_subprocess(cmd)

