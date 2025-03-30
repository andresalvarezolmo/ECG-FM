import subprocess
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 

def get_least_used_gpu():
    smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'])
    memory_free = [int(x) for x in smi_output.decode('utf-8').strip().split('\n')]
    return memory_free.index(max(memory_free))

GPU_ID = get_least_used_gpu()
print(GPU_ID)

cmd = r"""source /home/aa2650/playground/ECG-FM/virtualenv/bin/activate && \    
    export CUDA_VISIBLE_DEVICES=6 && \
    fairseq-hydra-train \
    task.data=/home/aa2650/datasets/code_15/manifests \
    model.model_path=/home/aa2650/playground/ECG-FM/ckpts/mimic_iv_ecg_physionet_pretrained.pt \
    model.num_labels=8 \
    optimization.lr=[1e-06] \
    optimization.max_epoch=140 \
    dataset.batch_size=64 \
    dataset.num_workers=5 \
    dataset.disable_validation=true \
    distributed_training.distributed_world_size=1 \
    distributed_training.find_unused_parameters=True \
    checkpoint.save_dir=/home32/aa2650/playground/ECG-FM/experiments \
    checkpoint.save_interval=1 \
    checkpoint.keep_last_epochs=0 \
    common.log_format=csv \
    +task.label_file=/home/aa2650/datasets/code_15/y.npy \
    --config-dir /home/aa2650/playground/fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer \
    --config-name diagnosis
"""

# Launch in a shell subprocess (new process = fresh CUDA context)
subprocess.run(cmd, shell=True, executable="/bin/bash")
