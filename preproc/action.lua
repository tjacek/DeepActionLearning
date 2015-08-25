require 'torch'
require 'math'

function simple_gradient(action) 
  local size=action:size()
  local action_diff=torch.Tensor(size[1]-2,size[2]-2,size[3]-2)
  local new_size=action_diff:size()
  for fr=1,new_size[1] do
    print(fr)
    for x=1,new_size[2] do
      for y=1,new_size[3] do
        delta=action[fr+2][x][y]-action[fr][x][y]
        action_diff[fr][x][y]=math.abs(delta)
      end
    end
  end
  return action_diff
end

function sobel_gradient(action)
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

function action_summary(action,n)
  n=n or 4
  local size=action:size()
  local summary=torch.Tensor(n,size[2],size[3])
  local segment_size=math.floor(size[1]/n)
  for i=1,n do
    local first=(i-1)*segment_size +1
    local last=i*segment_size
    if i==n then
      last=size[1]
    end
    segment_summary(first,last,action,summary[i])
  end
  return summary
end

function segment_summary(first,last,input,output)
  local frame_size=output:size()
  for x=1,frame_size[1] do
    for y=1,frame_size[2] do
      output[x][y]=0.0
      for j=first,last do
        output[x][y]=output[x][y]+input[j][x][y]
      end
      output[x][y]= output[x][y]/(last -first+1)
    end
  end 
end

function conv_diff(filename)
  action=torch.load(filename)
  conv_filename=string.gsub(arg[1],".tensor",".diff")
  action_diff=compute_gradient(action)
  torch.save(conv_filename,action_diff)
end

function get_action_summary(filename)
  action=torch.load(filename)
  conv_filename=string.gsub(arg[1],".diff",".summary")
  action_diff=action_summary(action)
  torch.save(conv_filename,action_diff)
end

function to_diff(in_file,out_file)
  action=torch.load(in_file)
  diff=simple_gradient(action)
  torch.save(out_file,diff)
end

if table.getn(arg) > 1 then
  to_diff(arg[1],arg[2])
end
