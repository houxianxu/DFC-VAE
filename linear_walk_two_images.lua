require 'torch'
require 'nn'
require 'image'
require 'Sampler'
disp = require 'display'

util = paths.dofile('util.lua')

opt = {
  linear_length = 20,
  fineSize = 128,
  nz = 100,               -- #  of dim for Z
  display = 1,            -- display images in browser. 0 = false
  gpu = 1,                -- currently only GPU support
  image_folder = 'images/',
  right_image = '007113.jpg', -- 000632.jpg
  left_image = '004465.jpg',  -- 001290.jpg
  decoder = 'checkpoints/cvae_content_123_decoder.t7',
  encoder = 'checkpoints/cvae_content_123_encoder.t7',
  input_images = '1.jpg,000632.jpg,001290.jpg,004465.jpg,007113.jpg,011116.jpg,011121.jpg,011125.jpg,011141.jpg,011149.jpg,011190.jpg,032439.jpg,032534.jpg,111331.jpg,111332.jpg,111333.jpg,111334.jpg,111335.jpg,111336.jpg,111337.jpg,111338.jpg,111339.jpg',
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

input_images_table = opt.input_images:split(',')

imgR = image.load(opt.image_folder .. opt.right_image, 3)

imgL = image.load(opt.image_folder .. opt.left_image, 3)

imgL = image.scale(imgL, 128)
imgR = image.scale(imgR, 128)

original = torch.cat(imgL:clone():reshape(1, 3, 128, 128), imgR:clone():reshape(1, 3, 128, 128), 1)

imgL = util.preprocess(imgL:float()):reshape(1, 3, 128, 128)
imgR = util.preprocess(imgR:float()):reshape(1, 3, 128, 128)
input = torch.cat(imgL, imgR, 1):type(dtype)

function compute_z(input)
    local results = encoder:forward(input)
    local mean = results[1]
    local log_var = results[2]
    local z = nn.Sampler():forward({mean, log_var})
    return z
end

noise = compute_z(input)
noiseL = noise[1]
noiseR = noise[2]

noise_linear = torch.Tensor(opt.linear_length, 100):type(dtype)

-- do a linear interpolation in Z space between point A and point B
-- each sample in the mini-batch is a point on the line
line = torch.linspace(0, 1, opt.linear_length)
for i = 1, opt.linear_length do
    noise_linear[i] = (noiseL * line[i] + noiseR * (1 - line[i])):clone()
end

reconstruct_results = decoder:forward(noise_linear)

fake = reconstruct_results


for i = 1, reconstruct_results:size(1) do
  fake[i] = util.deprocess(fake[i]:float():clone())
end

disp.image(fake, {win=1000, title='test'})
disp.image(original, {win=12, title='test input'})

-- image.save(opt.image_folder .. 'linear_walk_random.jpg', image.toDisplayTensor{input=fake, nrow=opt.linear_length})

