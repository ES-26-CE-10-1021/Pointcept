for jid in $(squeue -h -u $USER -t R -o "%i"); do
    idx=$(scontrol show job $jid -d | grep -oP 'GRES=gpu:[0-9]+\(IDX:\K[0-9,\-]+')
    name=$(scontrol show job $jid | grep -oP 'JobName=\K\S+')
    echo "Job $jid | GPU $idx | $name"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
