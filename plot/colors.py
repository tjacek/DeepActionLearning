import numpy as np
import random
import basic

class CatColors(object):
    def __init__(self,cats):
        self.cats=cats
        self.n_cats=cats.shape[0]

    def __call__(self,i,y_i):    
        num=float(self.cats[int(y_i)-1])
        div=float(self.n_cats)
        return  num/div

class PersonColors(object):
    def __init__(self, persons):
        self.persons = persons
    
    def __call__(self,i,y_i):
        return float(self.persons[i]%2)

def make_person_colors(persons):
    if(type(persons)==basic.Dataset):
        persons=persons.persons
    return PersonColors(persons)

def make_cat_colors(y,highlist=None):
    if(type(y)==basic.Dataset):
        y=y.y
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