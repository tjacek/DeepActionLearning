import numpy as np
import seq,seq.io,utils
import cv2

def make_action_imgs(in_path,out_path,by_cat=True):
    action_reader=seq.io.build_action_reader(img_seq=False,as_dict=False,as_group=False)
    actions=action_reader(in_path)
    print(len(actions))
    utils.make_dir(out_path)
    if(by_cat):
       actions=seq.by_cat(actions)
       for cat_j,actions_j in actions.items():
            cat_path=out_path+"/"+str(cat_j)
       	    utils.make_dir(cat_path)
       	    for action_ij in actions_j:
                save_action_img(action_ij,cat_path)
    else:
        for action_i in actions:
            save_action_img(action_i,out_path)

def save_action_img(action_i,out_path):
    action_img_i=np.array(action_i.img_seq)
    out_i=out_path+"/"+utils.set_prefix(action_i.name,".png")
    cv2.imwrite(out_i,action_img_i)