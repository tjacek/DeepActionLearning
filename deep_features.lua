require 'learning_nn'
require 'torch'
require 'io'

function deep_features(dataset_path,nn_path,out_path)
  local nn=torch.load(nn_path)
  nn:remove() --remove soft max
  nn:remove() --remove last linear layer
  local input_files=get_input_files(dataset_path)
  local dataset=get_dataset(input_files)
  local instances={}
  to_instances(instances,dataset.train,dataset.train_labels)
  to_instances(instances,dataset.test,dataset.test_labels)
  new_dataset=transform_instances(instances,nn)
  to_file(new_dataset,out_path)
end

function to_instances(raw_instances,data,labels)
  for i=1,labels:size()[1] do
    local _example=data[i]
    local dim=_example:size()
    local example=torch.Tensor(1,dim[1],dim[2]):zero()
    example[1]=_example
    local instance={example,labels[i]}
    raw_instances[#raw_instances+1] = instance
  end
  return raw_instances
end

function transform_instances(instances,nn)
  local new_dataset={}
  for i=1,#instances do
    local instance=instances[i]
    local reduced=nn:forward(instance[1]):clone()
    local new_instance={reduced,instance[2]}
    new_dataset[#new_dataset+1]=new_instance
  end
  return new_dataset
end

function to_file(new_dataset,path)
  local f_data=io.open(path,"w")
  for i=1,#new_dataset do
    local instance=new_dataset[i]
    local line=""
    line=data_to_str(line,instance[1])
    line=line .. "#".. tostring(instance[2]) .. "\n"
    f_data:write(line)
  end
  f_data:close()
end

function data_to_str(line,instance)
  for i=1,instance:size()[1] do
    line=line .. tostring(instance[i]) ..","
  end
  return line
end

dataset_path="/home/user/Desktop/dataset_2/"
deep_features(dataset_path,"nn_xy","raw")
