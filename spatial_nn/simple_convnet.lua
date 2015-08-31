require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'

function learning(train_data_file,train_labels_file,test_data_file,test_labels_file)
  hyper_params=default_hyper_params()
  model=create_model()
  global_vars(model)
  train_dataset=torch.load(train_data_file)
  train_labels=torch.load(train_labels_file)
  test_dataset=torch.load(test_data_file)
  test_labels=torch.load(test_labels_file)
  for i=1,100 do
    train(model,train_dataset,train_labels,hyper_params)
    test(model,test_dataset,test_labels,hyper_params)
  end
end


function global_vars(model)
  geometry = {84,42}
  parameters,gradParameters = model:getParameters()
  criterion = nn.ClassNLLCriterion()
  n_categories=20
  confusion = optim.ConfusionMatrix(n_categories)
end

function create_model()
  n_categories=20
  local model = nn.Sequential() --Input 84x42
  model:add(nn.SpatialConvolution(1, 16, 5, 5,2,2)) --output 40x19 
  model:add(nn.Tanh())

  model:add(nn.SpatialConvolutionMM(16, 32,4,4,2,2)) -- output 19*8

      -- stage 3 : standard 2-layer MLP:
  model:add(nn.Reshape(32*19*8))
  model:add(nn.Linear(32*19*8, 200))
  model:add(nn.Tanh())
  model:add(nn.Linear(200, n_categories))
  model:add(nn.LogSoftMax())
  return model
end

function train(model,dataset,labels,hyper_params)

  epoch = epoch or 1
  local time = sys.clock()
  for t = 1,dataset:size()[1] do
    local inputs = torch.Tensor(1,1,geometry[1],geometry[2])
    local targets = torch.Tensor(1)
    inputs[1]=dataset[t]
    targets[1]=labels[t]
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

      confusion:add(outputs[1], targets[1])
      return f,gradParameters
    end

    lbfgsState = lbfgsState or {
            maxIter = hyper_params.maxIter,
            lineSearch = hyper_params.lswolfe
    }
    optim.lbfgs(feval, parameters, lbfgsState)

    print('LBFGS step')
    print(' - progress in batch: ' .. t .. '/' .. dataset:size()[1])
    print(' - nb of iterations: ' .. lbfgsState.nIter)
    print(' - nb of function evalutions: ' .. lbfgsState.funcEval)
  end

  time = sys.clock() - time
  time = time / dataset:size()[1]
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
  print(confusion)
  print('% mean class accuracy (train set)' .. (confusion.totalValid * 100))
  confusion:zero()

  epoch = epoch + 1
end

function test(model,dataset,labels,hyper_params)
  local time = sys.clock()

  -- test over given dataset
  print('<trainer> on testing Set:')
  for t = 1,dataset:size()[1] do
    xlua.progress(t, dataset:size())
    local inputs = torch.Tensor(1,1,geometry[1],geometry[2])
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

function default_hyper_params()
  hyper_params={}
  hyper_params.maxIter=100
  hyper_params.lswolfe=true
  return hyper_params
end

path="/home/user/Desktop/dataset_2/"
train_data_file=path .. "train.tensor"
train_labels_file=path .. "train_labels.tensor"
test_data_file=path .. "test.tensor"
test_labels_file=path .. "test_labels.tensor"
learning(train_data_file,train_labels_file,test_data_file,test_labels_file)

