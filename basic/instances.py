import numpy as np
import sklearn.datasets
import utils

class InstsGroup(object):
    def __init__(self,instances):
        if(type(instances)==list):
            instances={ inst_i.name:inst_i for inst_i in instances}
        if(type(instances)==InstsGroup):
            instances=instances.instances
        self.instances=instances
    
    def __len__(self):
        return len(self.instances.keys())

    def __getitem__(self, key):
        return self.instances[key]
    
    def raw(self):
        return self.instances.values()

    def ordered_raw(self):
        names=self.names()
        return [self.instances[name_i] for name_i in names]

    def names(self):
        names=list(self.instances.keys())
        names.sort()
        return names
    
    def data(self):
        return [self.instances[name_i].data for name_i in self.names()] 

    def persons(self):
        return [self.instances[name_i].person for name_i in self.names()] 

    def cats(self):
        return [self.instances[name_i].cat for name_i in self.names()] 
    
    def get_cat(self,i):
        return [inst_i.name for inst_i in self.instances.values()
                    if(inst_i.cat==i)]

    def n_cats(self):
        cats=np.unique(self.cats())
        return cats.shape[0]

    def to_txt(self,out_path):
        lines=[str(inst_i) for inst_i in self.instances.values()]
        utils.save_string(out_path,lines)
    
    def split(self, selector=None):
        if(selector is None):
            selector= lambda inst_i: (inst_i.person % 2)==1
        train,test= utils.split(self.raw(),selector)
        return InstsGroup(train),InstsGroup(test)

class Instance(object):
    def __init__(self,data,cat,person,name):
        self.data = data
        self.cat=int(cat)
        self.person=int(person)
        self.name=name

    def  __str__(self):
        feats=[ str(feat_i) for feat_i in list(self.data)]
        feats=",".join(feats)
        name=self.name.replace('\n',"")
        return "%s#%s#%s#%s" % (feats,self.cat,self.person,name)

def make_instance(data_i,inst_i):
    return Instance(data_i,inst_i.cat,inst_i.person,inst_i.name)

def get_descs(names):
    if(type(names)==dict):
        names=list(names.keys())
    insts=[empty_instance(name_i)
            for name_i in names]
    return InstsGroup(insts)

def from_files(in_path):
    with open(in_path) as f:
         lines=f.readlines()
    lines=[ line_i 
                for line_i in lines
                    if(line_i.count('#')>=3)]
    insts=[ parse_instance(line_i)
                    for line_i in lines]                 
    return InstsGroup(insts)

def parse_instance(line_i):
    feats,cat,person,name=line_i.split("#")
    name=name.strip()
    data=utils.str_to_vector(feats)
    return Instance(data,cat,person,name)

def empty_instance(name):
    cat,person,e=utils.extract_numbers(name)
    return Instance(None,cat,person,name)