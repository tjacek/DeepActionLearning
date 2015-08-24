require 'torch'

if table.getn(arg) > 0 then
  local action=torch.load(arg[1])
  local size=action:size()
 -- print(size)
  print(size[3]/size[2])
end
