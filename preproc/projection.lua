require 'torch' 
require 'math'
 
function projection(action,plane)
  local max_z=torch.max(action)
  local min_z=first_nonzero(action)
  local size=action:size()
  local action_zx=torch.Tensor(size[1],size[2],max_z-min_z+2):zero()
  for i=1,size[1] do
    for j=1,size[2] do
      for k=1,size[3] do
        plane(i,j,k,min_z,action)
      end
    end
  end
  return action_zx
end

function plane_zx(i,j,k,min_z,action)
  local value=math.ceil(action[i][j][k],1)
  if( value>2) then
    action_zx[i][j][value-min_z+1]=k
  end
end

function plane_zy(i,j,k,min_z,action)
  local value=math.ceil(action[i][j][k],1)
  if( value>2) then
    action_zx[i][value-min_z+1][k]=j
  end
end

function first_nonzero(action)
  local min =math.huge
  local size=action:size()
  for i=1,size[1] do
    for j=1,size[2] do
      for k=1,size[3] do
        local value=action[i][j][k]
        if(value<min and (not (value==0))) then
           min=value
        end
      end
    end
  end
  return min
end

function projected_img(input_file,output_file)
  input=torch.load(input_file)
  output=projection_zy(input)
  torch.save(output_file,output)
end

if table.getn(arg) > 1 then
  --input="/home/user/Desktop/dataset_1/train/"
  --output_data="/home/user/Desktop/dataset_1/train.tensor"
  projected_img(arg[1],arg[2],plane_zx)
end
