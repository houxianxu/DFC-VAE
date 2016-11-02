require 'torch'
require 'nn'
require 'image'
require 'Sampler'
disp = require 'display'

util = paths.dofile('util.lua')

opt = {
  dataset = 'folder',
  batchSize =100,
  loadSize = 128,
  fineSize = 128,
  nz = 100,               -- #  of dim for Z
  nThreads = 3,           -- #  of data loading threads to use
  display = 1,            -- display samples while training. 0 = false
  gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  decoder = 'checkpoints/cvae_content_123_decoder.t7',
  encoder = 'checkpoints/cvae_content_123_encoder.t7',
  image_folder = 'images/',
  reconstruction = 0,
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
end
local decoder = torch.load(opt.decoder):type(dtype)
local encoder = torch.load(opt.encoder):type(dtype)

local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
local real = data:getBatch()
local input = real:type(dtype)

 for i = 1, input:size(1) do
   input[i] = util.preprocess(input[i]:float():clone())
 end    
 input = input:type(dtype)

local results = encoder:forward(input)
local mean = results[1]
local log_var = results[2]
local z = nn.Sampler():forward({mean, log_var})

if not (opt.reconstruction == 1) then
  z = torch.randn(z:size()):type(dtype):mul(0.4) -- can play with this parameter
end

reconstruct_results = decoder:forward(z)
fake = reconstruct_results

for i = 1, input:size(1) do
  input[i] = util.deprocess(input[i]:float():clone())
  input[i] = torch.clamp(input[i], 0, 1)
  fake[i] = util.deprocess(fake[i]:float():clone())
end

disp.image(fake, {win=1000, title='test'})
if opt.reconstruction == 1 then
  disp.image(input, {win=12, title='test input'})
end
-- image.save(opt.image_folder .. 'low_level_output_content.jpg', image.toDisplayTensor{input=fake, nrow=10})
-- image.save(opt.image_folder .. 'real.jpg', image.toDisplayTensor{input=real, nrow=10})
