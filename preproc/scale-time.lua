require 'torch'
require 'math'
require 'image'

function scale_time(new_time,action)
  local nframes=action:size()[1]
  local dim=torch.Tensor({action:size()[2],action:size()[3]})
  local new_action=torch.Tensor(nframes,dim[1],dim[2]):zero()
  local step=new_time/time
  for x_i=1,dim[1] do
    for y_i=1,dim[2] do
      for t=1,nframes do
        local t_i=t*step
        print(t_i)
      end
    end 
  end
  return scaled_action
end

if table.getn(arg) > 1 then
  local action=torch.load(arg[1])
  scaled_action=scale_time(action)
  torch.save(arg[2],scaled_action)
end
