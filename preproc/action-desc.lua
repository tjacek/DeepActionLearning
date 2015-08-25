require 'torch'
require 'math'
require 'image'

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

function uniform_size(summary)
  local square=to_square(summary)
  return image.scale(square,84,84)
end

function to_square(summary)
  local size=summary:size()
  local square=nil
  if(size[3]<size[2]) then
    square=torch.Tensor(size[2],size[2]):zero()
  else
    square=torch.Tensor(size[2],size[3]):zero()
  end
  for i=1,size[2] do
    for j=1,size[3] do
      square[i][j]=summary[1][i][j]
    end
  end
  return square
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

function get_action_summary(filename)
  action=torch.load(filename)
  conv_filename=string.gsub(arg[1],".diff",".summary")
  action_diff=action_summary(action)
  torch.save(conv_filename,action_diff)
end

function to_summary(in_file,out_file)
  local action=torch.load(in_file)
  local summary=action_summary(action,1)
  local img=uniform_size(summary)
  torch.save(out_file,img)
end

if table.getn(arg) > 1 then
  to_summary(arg[1],arg[2])
end
