import os
import re
import subprocess
import shutil

# Define configurations
configs = [
    {"task": "Covid19", "session": "Supervised"},
    {"task": "Covid19", "session": "Semi_supervised"},
    {"task": "MosMedDataPlus", "session": "Supervised"},
    {"task": "MosMedDataPlus", "session": "Semi_supervised"}
]

config_path = "Config.py"
backup_path = "Config.py.bak"

def modify_config(task_name, test_session):
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace task_name line
    content = re.sub(
        r"task_name\s*=\s*['\"].*?['\"]",
        f"task_name = '{task_name}'",
        content
    )
    
    # Replace test_session line
    content = re.sub(
        r"test_session\s*=\s*['\"].*?['\"]",
        f"test_session = '{test_session}'",
        content
    )
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

def parse_log(log_path):
    print(f"Parsing log: {log_path}")
    if not os.path.exists(log_path):
        print(f"Log path {log_path} does not exist.")
        return None
        
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_epoch = None
    epoch_data = {}
    best_epoch = None
    best_dice = 0.0
    
    epoch_re = re.compile(r"========= Epoch \[(\d+)/\d+\] =========")
    train_re = re.compile(r"\[Train\] Epoch: \[\d+\]\[(\d+)/\d+\].*Loss:[\d.]+\s+\(Avg ([\d.]+)\)\s+IoU:[\d.]+\s+\(Avg ([\d.]+)\)\s+Dice:[\d.]+\s+\(Avg ([\d.]+)\)")
    val_re = re.compile(r"\[Val\] Epoch: \[\d+\]\[(\d+)/\d+\].*Loss:[\d.]+\s+\(Avg ([\d.]+)\)\s+IoU:[\d.]+\s+\(Avg ([\d.]+)\)\s+Dice:[\d.]+\s+\(Avg ([\d.]+)\)")
    saving_best_re = re.compile(r"Saving best model, mean dice increased from: ([\d.]+) to ([\d.]+)")
    best_is_still_re = re.compile(r"the best is still: ([\d.]+) in epoch (\d+)")
    
    for line in lines:
        epoch_match = epoch_re.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epoch_data[current_epoch] = {
                'train_loss': None, 'train_iou': None, 'train_dice': None,
                'val_loss': None, 'val_iou': None, 'val_dice': None,
                'exact_val_dice': None
            }
            continue
            
        if current_epoch is None:
            continue
            
        train_match = train_re.search(line)
        if train_match:
            epoch_data[current_epoch]['train_loss'] = float(train_match.group(2))
            epoch_data[current_epoch]['train_iou'] = float(train_match.group(3))
            epoch_data[current_epoch]['train_dice'] = float(train_match.group(4))
            continue
            
        val_match = val_re.search(line)
        if val_match:
            epoch_data[current_epoch]['val_loss'] = float(val_match.group(2))
            epoch_data[current_epoch]['val_iou'] = float(val_match.group(3))
            epoch_data[current_epoch]['val_dice'] = float(val_match.group(4))
            continue
            
        saving_best_match = saving_best_re.search(line)
        if saving_best_match:
            val_dice_val = float(saving_best_match.group(2))
            epoch_data[current_epoch]['exact_val_dice'] = val_dice_val
            best_epoch = current_epoch
            best_dice = val_dice_val
            continue
            
        best_is_still_match = best_is_still_re.search(line)
        if best_is_still_match:
            best_epoch = int(best_is_still_match.group(2))
            best_dice = float(best_is_still_match.group(1))
            
    if best_epoch and best_epoch in epoch_data:
        best_data = epoch_data[best_epoch]
        best_data['exact_val_dice'] = best_dice
        return best_epoch, best_data
    return None

results = []

# Backup config
if os.path.exists(config_path):
    shutil.copy(config_path, backup_path)

try:
    for cfg in configs:
        task = cfg["task"]
        session = cfg["session"]
        print(f"\n==========================================")
        print(f"Running Evaluation: Task={task}, Session={session}")
        print(f"==========================================\n")
        
        # Modify config
        modify_config(task, session)
        
        # Run test_model.py
        process = subprocess.Popen(
            ["uv", "run", "python", "test_model.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        # Extract Test Metrics
        test_dice = None
        test_iou = None
        
        # parse output
        for line in stdout.splitlines():
            print(line)
            dice_match = re.search(r"dice_pred\s+([\d.]+)", line)
            if dice_match:
                test_dice = float(dice_match.group(1))
            iou_match = re.search(r"iou_pred\s+([\d.]+)", line)
            if iou_match:
                test_iou = float(iou_match.group(1))
                
        # Parse corresponding log file
        log_file = f"{task}/LViT/{session}/Session_Kaggle.log"
        log_metrics = parse_log(log_file)
        
        if log_metrics:
            best_epoch, best_data = log_metrics
        else:
            best_epoch = "N/A"
            best_data = {
                'train_loss': "N/A", 'train_iou': "N/A", 'train_dice': "N/A",
                'val_loss': "N/A", 'val_iou': "N/A", 'val_dice': "N/A",
                'exact_val_dice': "N/A"
            }
            
        results.append({
            "task": task,
            "session": session,
            "best_epoch": best_epoch,
            "train_loss": best_data['train_loss'],
            "train_iou": best_data['train_iou'],
            "train_dice": best_data['train_dice'],
            "val_loss": best_data['val_loss'],
            "val_iou": best_data['val_iou'],
            "val_dice": best_data['exact_val_dice'] if best_data.get('exact_val_dice') is not None else best_data['val_dice'],
            "test_dice": test_dice,
            "test_iou": test_iou
        })

finally:
    # Restore config
    if os.path.exists(backup_path):
        shutil.copy(backup_path, config_path)
        os.remove(backup_path)

# Generate markdown table
md_content = """# Model Evaluation and Training Summary

This report displays the training, validation (at the best epoch), and test metrics for the LViT model across all 4 settings.

| Dataset / Setting | Best Epoch | Train Loss | Train IoU | Train Dice | Val Loss | Val IoU | Val Dice | Test Dice | Test IoU |
|---|---|---|---|---|---|---|---|---|---|
"""

for res in results:
    def format_val(val):
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)
        
    md_content += (
        f"| **{res['task']}** ({res['session']}) | {res['best_epoch']} | "
        f"{format_val(res['train_loss'])} | {format_val(res['train_iou'])} | {format_val(res['train_dice'])} | "
        f"{format_val(res['val_loss'])} | {format_val(res['val_iou'])} | {format_val(res['val_dice'])} | "
        f"{format_val(res['test_dice'])} | {format_val(res['test_iou'])} |\n"
    )

print("\n\n==========================================")
print("FINAL SUMMARY RESULTS")
print("==========================================\n")
print(md_content)

with open("evaluation_summary.md", "w", encoding="utf-8") as f:
    f.write(md_content)
print("Saved summary to evaluation_summary.md")
