require 'xlua'
require 'image'
require 'torch'
require 'os'
require 'io'

mode_timer=1
mode_key=2

function show_action(action,mode)
  mode=mode or mode_timer
  nframes=action:size()[1]
  for i=1,nframes do
    frame=action[i]
    display=image.display(frame)
    window=display["window"]
    print(i)
    if mode==mode_timer then
      sleep(1)
    else
      local answer=io.read()
    end
    window:close()
  end
end

function sleep(s)
  local ntime = os.time() + s
  repeat until os.time() > ntime
end

function show_object(object)
  for key,value in pairs(object) do
    print("found member " .. key);
  end
end

if table.getn(arg) > 0 then
  action=torch.load(arg[1])
  show_action(action)
end

