require 'lfs'
require 'torch'

function create_dataset(input_dir,output_data,output_labels,image)
  local filenames=get_filenames(input_dir)
  local dataset=get_torch_dataset(filenames)
  torch.save(output_data,dataset[1])
  torch.save(output_labels,dataset[2])
end

function get_torch_dataset(filenames,image)
  local nframes=table.getn(filenames)  
  local dim=torch.load(filenames[1]):size()
  local img_size=dim[1]*dim[2]
  local dataset=torch.Tensor(nframes,img_size)
  local labels=torch.Tensor(nframes)
  for i,filename in pairs(filenames) do
    local image=torch.load(filename)
    if not image then
      image=image:resize(img_size)
    end
    dataset[i]=image
    labels[i]=get_label(filename)
  end
  return {dataset,labels}
end

function get_simple_dataset(filenames)
  local nframes=table.getn(filenames)
  local dim=torch.load(filenames[1]):size()
  local dataset=torch.Tensor(nframes,dim[1],dim[2])
  for i,filename in pairs(filenames) do
    --local label=get_label(filename)
    dataset[i]=torch.load(filename)
  end
  return dataset
end

function get_label(path)
  local filename=string.gsub(path,".+/","")
  local prefix=string.gsub(filename,"_.+","")
  label=string.gsub(prefix,"a","")
  return tonumber(label)
end

function get_filenames(dir)
  local filenames={}
  for file in lfs.dir(dir) do   
    if(string.find(file, "_")==4) then
      table.insert(filenames, dir .. file)
    end
  end
  return filenames
end

if table.getn(arg) > 0 then
  input="/home/user/Desktop/dataset_1/train/"
  output_data="/home/user/Desktop/dataset_1/train.tensor"
  output_labels="/home/user/Desktop/dataset_1/train_labels.tensor"
  create_dataset(input,output_data,output_labels)
end
