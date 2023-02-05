import torch
import numpy as np
import cogmtc
import sys
from cogmtc.models import *
import matplotlib.pyplot as plt
import cv2
import os

"""
Must argue the path to a model folder for viewing. This script
automatically selects the best model from the training.

$ python3 record_model.py exp_name/model_folder/

"""

n_episodes = 1 # Set this to longer to get more unique game xp
repeat = 4 # Set this to longer to linger on images longer
fps = 1
targ_range = [2,5] # max not inclusive

assert targ_range[0] < targ_range[1], "max is not inclusive"

if __name__ == "__main__":
    if not os.path.exists("./vids/"): os.mkdir("vids/")
    if not os.path.exists("./imgs/"): os.mkdir("imgs/")
    os.system("rm -rf ./imgs/*")
    model_folder = sys.argv[1]
    checkpt = cogmtc.utils.save_io.load_checkpoint(
        model_folder,
        use_best=False
    )
    hyps = checkpt["hyps"]
    hyps["n_eval_eps"] = n_episodes
    if targ_range is None: targ_range = hyps["val_targ_range"]
    hyps["val_targ_range"] = targ_range
    model = globals()[hyps["model_type"]](**hyps).cuda()
    model.load_state_dict(checkpt["state_dict"])
    model.eval()
    model.reset()
    val_runner = cogmtc.experience.ValidationRunner(hyps)
    val_runner.phase = 2
    model.reset(1)

    env_type = hyps["env_types"][0]
    val_runner.oracle = val_runner.oracles[env_type]
    all_data = None
    for n_targs in range(*targ_range):
        model.reset()
        data = val_runner.collect_data(
            model, n_targs=n_targs, env_type=env_type, n_eps=n_episodes
        )
        if all_data is None:
            all_data = data
        else:
            all_data["states"] = np.concatenate(
                [all_data["states"], data["states"]],
                axis=0
            )
    data = all_data
    torch.cuda.empty_cache()

    frames = np.asarray(data["states"])
    print("collected data:", frames.shape)
    output_root = "imgs/pic{:03d}.png"
    for i,frame in enumerate(frames.squeeze()):
        plt.imshow(frame)
        plt.savefig(output_root.format(i))
        plt.clf()
    output_name = "vids/"+hyps["exp_name"] + str(hyps["exp_num"]) + ".mp4"
    s = "ffmpeg -r {} -f image2 -s 90x90 -i imgs/pic%03d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {}"
    os.system(
        s.format(fps, output_name)
    )
    #frames = np.repeat(frames.transpose((0,2,3,1)), 3, axis=-1)

    #frames[frames==0] = -2
    #frames = (frames+2)
    #frames = frames/np.max(frames)
    #frames = np.uint8(frames*255).repeat(3,axis=1).repeat(3,axis=2)
    #frames = np.repeat(frames, repeat, axis=0)
    #output_name = hyps["exp_name"] + str(hyps["exp_num"]) + ".mp4"
    #print("shape:", frames.shape)
    #out = cv2.VideoWriter("vids/"+output_name,
    #    cv2.VideoWriter_fourcc(*'mp4v'),
    #    fps,
    #    frames[0].shape[:2]
    #)
    #for frame in frames:
    #    out.write(frame)
    #out.release()
