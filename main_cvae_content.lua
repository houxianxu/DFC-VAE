require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'
require 'image'
disp = require 'display'
util = paths.dofile('util.lua')

VAE = require 'CVAE'

require 'src/utils'
require 'src/descriptor_net'


opt = {
  dataset = 'folder',
  batchSize = 40,
  loadSize = 128,         -- use bigger size for this version
  fineSize = 128,
  nz = 100,               -- #  of dim for Z
  ngf = 32,               -- #  of gen filters in first conv layer
  ndf = 32,               -- #  of discrim filters in first conv layer
  nThreads = 4,           -- #  of data loading threads to use
  niter = 10,             -- #  of iter at starting learning rate
  lr = 0.0005,            -- initial learning rate for adam
  beta1 = 0.5,            -- momentum term of adam
  ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
  display = 1,            -- display samples while training. 0 = false
  display_out = 'images',        -- display window id or output folder
  gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  name = 'cvae_content',

  proto_file = 'data/pretrained/VGG_ILSVRC_19_layers_deploy.prototxt',
  model_file = 'data/pretrained/VGG_ILSVRC_19_layers.caffemodel',
  backend = 'cudnn',

  vgg_no_pad = false,
  image_size = 256,
  content_layers = 'relu1_1,relu2_1,relu3_1',
  content_weight = 0.5,
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end


opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

local descriptor_net, vgg_conv, content_losses = create_descriptor_net()

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nz = opt.nz

local encoder = VAE.get_encoder(3, opt.ndf, nz)
local decoder = VAE.get_decoder(3, opt.ngf, nz)

local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})

local reconstruction, reconstruction_var, model

reconstruction = decoder(z)
model = nn.gModule({input},{reconstruction, mean, log_var})
criterion = nn.BCECriterion()
criterion.sizeAverage = false

encoder:apply(weights_init)
decoder:apply(weights_init)

KLD = nn.KLDCriterion():cuda()

local parameters, gradients = model:getParameters()

---------------------------------------------------------------------------
optimState = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
local lowerbound = 0
----------------------------------------------------------------------------

local fx = function(x)
    if x ~= parameters then
        parameters:copy(x)
    end

    model:zeroGradParameters()
    local reconstruction, reconstruction_var, mean, log_var

    data_tm:reset(); data_tm:resume()
    local real = data:getBatch()
    while real == nil do
        print('Got nil batch for real')
        real = data:getBatch()
    end
    data_tm:stop()
    input:copy(real)

    for i = 1, input:size(1) do
      input[i] = util.preprocess(input[i]:float():clone())
    end    
    input = input:cuda()
    vgg_conv:forward(input:clone())
    for i = 1, #content_losses do
      for j = 1, #vgg_conv do
        local layer = vgg_conv:get(j)
        if content_losses[i].name == layer.name then
          content_losses[i].target = layer.output:clone()
        end
      end
    end

    reconstruction, mean, log_var = unpack(model:forward(input))

    -- use content loss
    descriptor_net:forward(reconstruction)
    local df_dw_content = descriptor_net:updateGradInput(reconstruction, nil)
    local content_loss = 0
    for _, mod in ipairs(content_losses) do
      content_loss = content_loss + mod.loss
    end

    err = content_loss
    df_dw = df_dw_content

    local KLDerr = KLD:forward(mean, log_var)
    local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))


    error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var}
    
    model:backward(input, error_grads)

    local batchlowerbound = {KLDerr,err}

    return batchlowerbound, gradients
end


-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0

   lowerbound = 0
   recons_bound = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      -- print(data:size())
      tm:reset()

      -- Update model
      if epoch % 2 == 0 then
        optimState.learningRate = optimState.learningRate*0.5
      end

      x, batchlowerbound = optim.adam(fx, parameters, optimState)
      lowerbound = lowerbound + batchlowerbound[1][1]
      recons_bound = recons_bound + batchlowerbound[1][2]
      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(input))
          -- print(reconstruction:min(), reconstruction:max())
          if reconstruction then
            for i = 1, input:size(1) do
              input[i] = util.deprocess(input[i]:float():clone())
              reconstruction[i] = util.deprocess(reconstruction[i]:float():clone())
            end

            disp.image(input, {win=2, title=opt.name})
            disp.image(reconstruction, {win=2*2, title=opt.name})
          else
            print('Fake image is Nil')
          end
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Lowerbound: %.4f' .. ' reconstruction: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 lowerbound/((i-1)/opt.batchSize),
                 recons_bound/((i-1)/opt.batchSize)))
      end
   end

   lowerboundlist = torch.Tensor(1,1):fill(lowerbound/(epoch * math.min(data:size(), opt.ntrain)))

   paths.mkdir('checkpoints')
   encoder:clearState()
   decoder:clearState()
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_encoder.t7', encoder)
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_decoder.t7', decoder)
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end


