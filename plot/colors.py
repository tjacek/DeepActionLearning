import numpy as np
import random

class CatColors(object):
    def __init__(self,cats):
        self.cats=cats
        self.n_cats= cats.shape[0]

    def __call__(self,i,y_i):    
        num=float(self.cats[int(y_i)-1])
        div=float(self.n_cats)
        return  num/div

class PersonColors(object):
    def __init__(self, persons):
        self.persons = persons
    
    def __call__(self,i,y_i):
        return float(self.persons[i]%2)

def make_cat_colors(y,highlist=None):
    cats=np.unique(y)
    if(highlist):
        highlist=Set(highlist)
        for i in range(n_cats):
            cat_i= int(cats[i])
            if(not cat_i in highlist):
                cats[i]=0
    else:
        random.shuffle(cats)
    return CatColors(cats)