require 'torch'
require 'math'

function compute_gradient(action)
  local size=action:size()
  local action_diff=torch.Tensor(size[1]-2,size[2]-2,size[3]-2)
  sobel=get_sobol_operator()
  local new_size=action_diff:size()
  for fr=1,new_size[1] do
    print(fr)
    for x=1,new_size[2] do
      for y=1,new_size[3] do
        action_diff[fr][x][y]=apply_operator(fr,x,y,action,sobel)
      end
    end
  end
  return action_diff
end

function apply_operator(fr,x,y,action,operator)
  local value=0
  for i=1,3 do
    for j=1,3 do
      local x_i=x+i-1
      local y_j=y+j-1
      value=value + action[fr][x_i][y_j]*operator[1][i][j]
      value=value+action[fr+2][x_i][y_j]*operator[2][i][j]
    end
  end  
  return math.abs(value)
end

function get_sobol_operator()
  local sobel3D=torch.Tensor(2,3,3)
  sobel3D[2][1][1]=1.0
  sobel3D[2][1][2]=2.0
  sobel3D[2][1][3]=1.0
  sobel3D[2][2][1]=2.0
  sobel3D[2][2][2]=4.0
  sobel3D[2][2][3]=2.0
  sobel3D[2][3]=sobel3D[2][1]
  sobel3D[1]=sobel3D[2]*(-1.0)
  return sobel3D
end

if table.getn(arg) > 0 then
  action=torch.load(arg[1])
  conv_filename=string.gsub(arg[1],".tensor",".diff")
  action_diff=compute_gradient(action)
  torch.save(conv_filename,action_diff)
end
