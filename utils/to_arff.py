import os,sys,re

def to_arff(in_path):
    instances=read_instances(in_path)
    arff=create_arff(instances)
    out_path=in_path+".arff"
    file_ = open(out_path, 'w')
    file_.write(arff)
    file_.close()

def read_instances(path):
    raw_file = open(path)
    raw=raw_file.read()
    instances=[]
    for line in raw.split("\n"):
        line=line.split(",#")
        if(len(line)>1):
            data=extract_data(line[0])
            category=int(line[1])
            instances.append([data,category])
    return instances

def extract_data(raw_data):
    #raw_data=map(lambda x:x.replace(" ",""),raw_data.split(","))   
    return raw_data#map(float,raw_data)

def create_arff(instances):
    num_features=len(instances[0][0].split(","))
    num_cat=max(map(lambda x:x[1],instances))
    categories=get_categories(num_cat)
    header=create_header(num_features,categories)
    data=create_dataset(instances,categories)
    return header+data

def create_header(num_features,categories):
    header="@RELATION deep \n"
    for i in range(num_features):
        header+="@ATTRIBUTE conv  NUMERIC\n"
    class_line="@ATTRIBUTE class {"
    for cat in categories:
	class_line+=cat+","
    class_line+="}\n" 
    return header+class_line+"\n"

def create_dataset(instances,categories):
    data="@DATA\n"
    for instance in instances:
        data+=instance[0] +","+ categories[instance[1]-1] +"\n"
    return data

def get_categories(num_cat):
    return map(lambda x: "a"+str(x),range(num_cat))
	

to_arff(sys.argv[1])
