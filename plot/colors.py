import numpy as np
import random
import basic,ensemble.outliner

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

class MakeOutlinerColors(object):
    def __init__(self,outliners):
        self.outliners=outliners
        self.n_cats=len(self.outliners)
        self.index=0

    def __call__(self,dataset_j):
        outliner_j=self.outliners[self.index]
        outliner_points_j=outliner_j.predict(dataset_j.X)
        outliner_points_j[outliner_points_j==(-1)]==0.0
        def color_helper(i,y_i):
            return 2*float(outliner_points_j[i])
        self.next_index()
        return color_helper

    def next_index(self):
        self.index= (self.index+1) % self.n_cats

def built_make_outlines(in_path):
    outliners=ensemble.outliner.read_detectors(in_path)
    return MakeOutlinerColors(outliners.outliner_detectors)