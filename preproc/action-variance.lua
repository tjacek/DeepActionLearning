require 'torch'
require 'math'
require 'image'

function action_variance(action)
  local nframes=action:size()[1]
  local dim=torch.Tensor(2)
  dim[1]=action:size()[2];  dim[2]=action:size()[3];
  print(dim)
  local mean=mean_of_image(action,dim,nframes)
  return variance_of_image(action,mean,dim,nframes)
  --return mean
end

function mean_of_image(action,dim,nframes)
  local mean=torch.Tensor(dim[1],dim[2])
  for x_i=1,dim[1] do
    for y_i=1,dim[2] do
      for t=1,nframes do
        mean[x_i][y_i]=mean[x_i][y_i]+action[t][x_i][y_i]
      end
      mean[x_i][y_i]=mean[x_i][y_i]/nframes
    end
  end
  return mean
end

function variance_of_image(action,mean,dim,nframes)
  local variance=torch.Tensor(dim[1],dim[2]):zero()
  for x_i=1,dim[1] do
    for y_i=1,dim[2] do
      for t=1,nframes do
        diff=action[t][x_i][y_i] - mean[x_i][y_i]
        diff=diff^2
        variance[x_i][y_i]=variance[x_i][y_i]+diff
      end
      variance[x_i][y_i]=variance[x_i][y_i]/(nframes-1)
      variance[x_i][y_i]= math.sqrt(variance[x_i][y_i])
    end
  end
  return variance
end

function uniform_size(img)
  return image.scale(img,84,42)
end

function to_action_variance(in_file,out_file)
  local action=torch.load(in_file)
  local var=action_variance(action,1)
  local img=uniform_size(var)
  torch.save(out_file,img)
  --torch.save(out_file,var)
end

if table.getn(arg) > 1 then
   to_action_variance(arg[1],arg[2])
end
