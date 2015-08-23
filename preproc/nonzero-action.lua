require 'image'
require 'torch'
require 'os'
require 'io'

function nonzero_action(action)
  local extr=find_extreme_points(action[1])
  local nframes=action:size()[1]
  for i=2,nframes do
    local cnd_extr=find_extreme_points(action[i])
    if extr[1][1] > cnd_extr[1][1] then
      extr[1][1]=cnd_extr[1][1]
    end
    if extr[1][2] > cnd_extr[1][2] then
      extr[1][2]=cnd_extr[1][2]
    end
    if extr[2][1] < cnd_extr[2][1] then
      extr[2][1]=cnd_extr[2][1]
    end
    if extr[2][2] < cnd_extr[2][2] then
      extr[2][2]=cnd_extr[2][2]
    end
  end
  local dim=extr[2]-extr[1]
  print(dim)
  print(dim[1]*dim[2])
  nonzero=remove_zeros(extr,action)
  return nonzero
end

function find_extreme_points(frame)
  local size=frame:size()
  local min_x=nil
  local min_y=nil--size[2]
  local max_x=1
  local max_y=1
  for x_i=1,size[1] do
    for y_i=max_y,size[2] do
      if not (frame[x_i][y_i]==0) then
         max_x=x_i
         max_y=y_i
         if not min_x then
           min_x=x_i
         end 
      end
    end
  end
  min_y=max_y
  for x_i=max_x,size[1] do
    for y_i=1,size[2] do
      if not (frame[x_i][y_i]==0) then
         max_x=x_i
         if y_i<min_y then
           min_y=y_i
         end
         break 
      end
    end
  end
  return pack_extreme(min_x,min_y,max_x,max_y)
end

function pack_extreme(min_x,min_y,max_x,max_y)
  return torch.Tensor({{min_x,min_y},{max_x,max_y}})
end

function remove_zeros(extreme_points,action)
  local size=action:size()
  local min=extreme_points[1]
  local max=extreme_points[2]
  return action:sub(1,size[1],min[1],max[1],min[2],max[2])
end

function to_nonzero(in_file,out_file)
  action=torch.load(in_file)
  nonzero= nonzero_action(action)
  torch.save(out_file,nonzero)
end

if table.getn(arg) > 0 then
  to_nonzero(arg[1],arg[2])
end
