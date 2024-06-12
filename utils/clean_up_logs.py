import glob, os, shutil

# Really dirty script to remove training logs that had less than 20 epochs

dst = "cleaned_logs"
min_epoch = 20
files = glob.glob("lightning_logs/*/checkpoints/*.ckpt")
print(files)
for ckpt in files:

    ckpt_epochs = int(ckpt.split("epoch=")[1].split("-step")[0])
    if ckpt_epochs > min_epoch:
        new_ckpt = ckpt.replace("lightning_logs", dst)
        print(new_ckpt)
        
        if not os.path.exists("/".join(new_ckpt.split("/")[:-1])):
            os.makedirs("/".join(new_ckpt.split("/")[:-1]))

        shutil.copyfile(ckpt, new_ckpt)

        for logs_src in glob.glob("/".join(ckpt.split("/")[:-2]) + "/*"):
            if "tfevents" in logs_src or ".csv" in logs_src:
                new_logs = logs_src.replace("lightning_logs", dst)
                print(new_logs)

                if not os.path.exists("/".join(new_logs.split("/")[:-1])):
                    os.makedirs("/".join(new_logs.split("/")[:-1]))

                shutil.copyfile(logs_src, new_logs)

        
