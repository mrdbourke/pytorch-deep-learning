# python environment
python_path="${HOME}/anaconda3/envs/tadac/bin/python"

## home path
home_path=$PWD
echo "HOME: $home_path"
current_time="$(date +'%Y-%m-%d')"

## names
train_name="tensorflow" # or "pytorch"
folder_name="alexnet" # change to another name here

## create a tmux log for checking if errors occur
tmux_log_folder="${home_path}/logs/tmux-log-${current_time}/$folder_name"
mkdir -p $tmux_log_folder

echo "Created a tmux log folder: $tmux_log_folder"

# path to the main.py
train_py='path/to/main.py'

tmux new-session -d -s "${train_name}_train_job" \; attach -t "${train_name}_train_job" \; send-keys "${python_path} ${train_py}" Enter  \; pipe-pane "cat > ${tmux_log_folder}/tmux_${train_name}_train_job_${current_time}.txt" \; detach \;

echo "Created train_py session" 