import numpy as np
import cv2
import seq,seq.io,utils,basic.group

def group_action_imgs(in_path,out_path,by_cat=True):
    other_args={"by_cat":by_cat}
    group_fun=basic.group.GroupFun(make_action_imgs,dirs=True)
    group_fun(in_path,out_path,other_args)

def make_action_imgs(in_path,out_path,by_cat=True):
    print(in_path)
    action_reader=seq.io.build_action_reader(img_seq=False,as_dict=False,as_group=True)
    actions=action_reader(in_path)
    actions.normalization()
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
    action_img_i*=25.0
    action_img_i=enlarge(action_img_i)
    action_img_i=enlarge(action_img_i.T)
    out_i=out_path+"/"+utils.set_prefix(action_i.name,".png")
    cv2.imwrite(out_i,action_img_i)

def enlarge(action_img_i):
    thick_img=[]
    for x_j in action_img_i:
        thick_img+=[x_j for k in range(5)]
    return np.array(thick_img)