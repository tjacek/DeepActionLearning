require 'torch'
require 'xlua'
require 'image'

function read_action(input)
  local binary = assert(io.open(input,"rb"))
  --local raw_action = {}
  --repeat
  --  local str = binary:read(4*1024)
  --  for c in (str or ''):gmatch'.' do
  --    t[#raw_action+1] = c:byte()
  --  end
  --until not str
  --binary:close()
  raw_action = binary:read("*all") 
  binary:close()
  --print(string.byte(raw_action,1000))
  header=read_header(raw_action)
  data=read_data(raw_action,header)
  return data
end

local Header = {}
Header.__index = Header

function Header.new(num_frames, ncols, nrows)  
  data={num_frames = num_frames, ncols = ncols, nrows = nrows}  
  return setmetatable(data, Header)
end

function read_header(raw_action)
  local num_frames=read_int(raw_action,1)
  local ncols=read_int(raw_action,5)
  local nrows=read_int(raw_action,9)
  return Header.new(num_frames,ncols,nrows)
end

function read_data(raw_action,header)
  frames=header['num_frames']
  cols=header['ncols']
  rows=header['nrows']
  data = torch.Tensor(frames,rows,cols)
  index=13
  for i=1,frames do
    --print(i)
    for j=1,rows do
      for k=1,cols do
        data[i][j][k]=read_int(raw_action,index)
        index=index+4 
      end
    end
  end
  return data
end

function read_int(raw_action,pos)
  integer=raw_action:byte(pos)
  base=256
  --print(integer)
  integer=integer+raw_action:byte(pos+1)*base
  --integer=integer+raw_action:byte(pos+2)*(base^2)
  --integer=integer+raw_action:byte(pos+3)*(base^3)
  return integer
end

function raw_to_tensor(filename)
  data=read_action(filename)
  print(data:size())  
  conv_filename=string.gsub(filename,".bin",".tensor")
  torch.save(conv_filename,data)
end

function transform_file(in_file,out_file)
  data=read_action(in_file)
  torch.save(out_file,data)
end

if table.getn(arg) > 0 then
  transform_file(arg[1],arg[2])
end
