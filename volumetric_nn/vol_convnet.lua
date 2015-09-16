require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'

function learning(path)
  hyper_params=default_hyper_params()
  input_files=get_input_files(path)
  dataset=get_dataset(input_files)
  local model=create_model()
  global_vars(model)
  for i=1,1000 do
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Epoch ")
    print(i)
    train(model,dataset.train,dataset.train_labels,hyper_params)
    test(model,dataset.test,dataset.test_labels,hyper_params)
  end
end

function default_hyper_params()
  hyper_params={}
  hyper_params.maxIter=100
  hyper_params.lswolfe=true
  hyper_params.learningRate= 0.01
  hyper_params.momentum=0
  hyper_params.batchSize=293
  return hyper_params
end

function global_vars(model)
  geometry = {10,40,40}
  parameters,gradParameters = model:getParameters()
  criterion = nn.ClassNLLCriterion()
  n_categories=20
  confusion = optim.ConfusionMatrix(n_categories)
end

function create_model()
  n_categories=20
  local first=8
  local second=16
  local model = nn.Sequential()
  model:add(nn.VolumetricConvolution(1,first, 4, 4,2,2,2,2)) -- Input 10*40*40
  model:add(nn.Tanh())
  model:add(nn.VolumetricConvolution(first,second, 4,5,5, 1,2,2))--Input 6 *5*19*19
  model:add(nn.Reshape(second*1*8*8))  --Input 6*2*8*8
  model:add(nn.Linear(second*1*8*8, 500))
  model:add(nn.Tanh())
  model:add(nn.Linear(500, n_categories))
  model:add(nn.LogSoftMax())
  return model
end

function train(model,dataset,labels,hyper_params)
  local time = sys.clock()
  local nsamples=dataset:size()[1]
  for t = 1,nsamples,hyper_params.batchSize do
    local inputs = torch.Tensor(hyper_params.batchSize,1,geometry[1],geometry[2],geometry[3])
    local targets = torch.Tensor(hyper_params.batchSize)
    local k = 1
    for i = t,math.min(t+hyper_params.batchSize-1,nsamples) do
      local sample = dataset[i]
      local input = sample:clone()
      --local target= torch.Tensor(hyper_params.batchSize)
      inputs[k] = input
      targets[k] = labels[i]
      k = k + 1
    end

    local feval = function(x)
      collectgarbage()
      if x ~= parameters then
         parameters:copy(x)
      end
      -- reset gradients
      gradParameters:zero()

         -- evaluate function for complete mini batch
   
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)

      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      for i = 1,hyper_params.batchSize do
        confusion:add(outputs[i], targets[i])
      end
      return f,gradParameters
    end

    sgd_optimisation(feval, parameters,hyper_params)   
   
  end

  time = sys.clock() - time
  time = time / dataset:size()[1]
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
  print(confusion)
  print('% mean class accuracy (train set)' .. (confusion.totalValid * 100)) 
  local valid=  confusion.totalValid
  confusion:zero()

  return valid
end

function sgd_optimisation(feval, parameters,hyper_params)
  sgdState = sgdState or {
            learningRate = hyper_params.learningRate,
            momentum = hyper_params.momentum,
            learningRateDecay = 5e-7
         }
  optim.sgd(feval, parameters, sgdState)
end

function test(model,dataset,labels,hyper_params)
  local time = sys.clock()

  -- test over given dataset
  print('<trainer> on testing Set:')
  for t = 1,dataset:size()[1] do
    --xlua.progress(t, dataset:size())
    local inputs = torch.Tensor(1,1,geometry[1],geometry[2],geometry[3])
    local targets = torch.Tensor(1)
    inputs[1]=dataset[t]
    targets[1]=labels[t]

    local preds = model:forward(inputs)
    confusion:add(preds[1], targets[1])
  end

  time = sys.clock() - time
  time = time / dataset:size()[1]
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  print(confusion)
  confusion:zero()
end

--path="/home/user/Desktop/dataset_3/"
--learning(path)
