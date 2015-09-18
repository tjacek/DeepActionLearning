require 'torch'
require 'math'
require 'image'

function scale_space(action)
  local dim={60,50}
  local nframes=action:size()[1]
  local scaled_action=torch.Tensor(nframes,dim[1],dim[2]):zero()
  for t=1,nframes do
    scaled_action[t]=image.scale(action[t],dim[1],dim[2],bilinear)
  end
  return scaled_action
end

if table.getn(arg) > 1 then
  local action=torch.load(arg[1])
  scaled_action=scale_space(action)
  torch.save(arg[2],scaled_action)
end
