require 'torch'
require 'math'
require 'image'

function scale_time(new_time,action)
  local nframes=action:size()[1]
  local dim=torch.Tensor({action:size()[2],action:size()[3]})
  local new_action=torch.Tensor(new_time,dim[1],dim[2]):zero()
  local step=nframes/new_time
  print(nframes)
  for x_i=1,dim[1] do
    for y_i=1,dim[2] do
      for t=1,new_time do
        local t_i=t*step
        new_action[t][x_i][y_i]=apply_kernel(t_i,x_i,y_i,nil,action)
      end
    end 
  end
  return new_action
end

function apply_kernel(t,x,y,kernel,action)
  local t_i=math.floor(t)
  local value=0
  for i=1,5 do
    local k=t_i-2+1
    value=value+get_value(k,x,y,action)
  end
  return value / 4 
end

function get_value(t,x,y,action)
  local dim=action:size()
  print(dim)
  if(t<1 or t>dim[1]) then
    return 0
  end
  if(x<1 or x>dim[2]) then
    return 0
  end
  if(y<1 or y>dim[3]) then
    return 0
  end
  return action[t][x][y]
end

if table.getn(arg) > 1 then
  local action=torch.load(arg[1])
  scaled_action=scale_time(10,action)
  torch.save(arg[2],scaled_action)
end
