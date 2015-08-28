require 'lfs'
require 'torch'

function create_dataset(input_dir,output)
  local filenames=get_filenames(input_dir)
  local dataset=get_dataset(filenames)
  torch.save(output,dataset)
end

function get_dataset(filenames)
  local nframes=table.getn(filenames)
  local dim=torch.load(filenames[1]):size()
  local dataset=torch.Tensor(nframes,dim[1],dim[2])
  for i,filename in pairs(filenames) do
    dataset[i]=torch.load(filename)
  end
  return dataset
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
  output="/home/user/Desktop/dataset_1/train.tensor"
  create_dataset(input,output)
end
