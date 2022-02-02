'''
GPU contention generator:
The optional argument "GPU" means the GPU utilization in percentage and should
  be one from [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99];

Usage:
python LiteReconfig_CG.py --GPU 50
'''

import subprocess, argparse, os, time, socket

def contention_generator_kill():

    cmd = "pkill -f contention_module"
    p = subprocess.Popen(cmd, shell=True)
    output = p.communicate()[0]
    time.sleep(1)
    print("All contentions are killed!")

def contention_generator_launch(gpu_level):

    contention_generator_kill()
    cpu_core_list = [0]
    if socket.gethostname() in ['tx2-1', 'tx2-2']:
        cpu_core_list = [1, 2, 0, 3, 4, 5]
    elif socket.gethostname() in ['xv3']:
        cpu_core_list = [0, 1, 2, 3, 4, 5, 6, 7]
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    if gpu_level > 0:
        bin_path = os.path.join(cur_dir, "contention_module_gpu.py")
        cmd = f"taskset -c {cpu_core_list[0]} python3 {bin_path} --GPU {gpu_level}"
        _ = subprocess.Popen(cmd, shell = True)
        print(f"{gpu_level}% GPU contention created.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', type=int, default=0, help='GPU contention level')
    args = parser.parse_args()
    contention_generator_launch(args.GPU)

