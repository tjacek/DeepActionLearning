import seqs.io

class ModuloSelector(object):
    def __init__(self,n,unpack=None,k=2):
        self.k=k
        self.n=n
        self.unpack=unpack_person if(unpack is None) else unpack

    def __call__(self,i):
        i=self.unpack(i)
        return (i % self.k)==self.n        

def select_actions(in_path,out_path=None,selector=1,img_seq=True):
    read_actions=seqs.io.build_action_reader(img_seq=img_seq,as_dict=False)
    actions=read_actions(in_path)
    new_actions=select(actions,selector)
    save_actions=seqs.io.ActionWriter(img_seq=img_seq)  
    save_actions(new_actions,out_path)

def select(actions,selector):
    if(type(selector)==int):
        selector=ModuloSelector(selector,unpack=unpack_person)
    return [ action_i
                for action_i in actions
                    if(selector(action_i))]

def unpack_cat(action_i):
    return action_i.cat 

def unpack_person(action_i):
    return action_i.person	