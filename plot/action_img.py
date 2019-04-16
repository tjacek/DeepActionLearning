import numpy as np
import seq.io,utils
import cv2

def make_action_imgs(in_path,out_path):
    action_reader=seq.io.build_action_reader(img_seq=False,as_dict=False,as_group=False)
    actions=action_reader(in_path)
    print(len(actions))
    utils.make_dir(out_path)
    for action_i in actions:
        action_img_i=np.array(action_i.img_seq)
        out_i=out_path+"/"+utils.set_prefix(action_i.name,".png")
        cv2.imwrite(out_i,action_img_i)