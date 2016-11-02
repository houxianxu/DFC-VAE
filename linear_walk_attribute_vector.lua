require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'Sampler'
disp = require 'display'

util = paths.dofile('util.lua')

opt = {
  face_folder = 'celebA/img_align_celeba',
  attribute_file = 'celebA/attr_binary_list.txt',
  num =10000,
  nz = 100,               -- #  of dim for Z
  img_size =  128,
  display = 1,            -- display samples while training. 0 = false
  gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  batch_size = 100,
  encoder = 'checkpoints/cvae_content_123_encoder.t7',
  decoder = 'checkpoints/cvae_content_123_decoder.t7',
  attr_extracted = '5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young',
  selected_attr = 'Smiling',
  attr_scale = 20,
  image_folder = 'images/',
  test_image = '001290.jpg'
}

opt.attr_extracted = opt.attr_extracted:split(',')

local encoder = torch.load(opt.encoder):cuda()
local decoder = torch.load(opt.decoder):cuda()

-- load list attr
local attrs_image_list = {}
local f = io.open(opt.attribute_file, "r")
for line in f:lines() do
    line = line:split(',')
    attr_name = line[1]  -- attr + _pos or _neg
    line[1] = nil
    attrs_image_list[attr_name] = line
end

function get_dataset_attr(attr, type)
    local attr_img_list = attrs_image_list[attr .. '_' .. type]
    print(#attr_img_list)
    local imgs = torch.FloatTensor(opt.num, 3, opt.img_size, opt.img_size)
    for k, v in pairs(attr_img_list) do
        -- print(k-1)
        if k-1 > opt.num then break end
        local img_name = opt.face_folder .. '/' .. v
        local img = image.load(img_name)
        img = util.preprocess(img:float())
        imgs[k-1] = img
    end
    return imgs
end

function compute_vect(dataset, encoder)
    local input = dataset:cuda()
    local results = encoder:forward(input)
    local mean = results[1]
    local log_var = results[2]
    local z = nn.Sampler():forward({mean, log_var})
    return z
end

function get_batch(dataset)
    -- use the last 5000 as validation images
    local batch_size = opt.batch_size
    local mask = torch.randperm(batch_size):long()
    local img_batch = torch.ByteTensor(batch_size, 3, opt.img_size, opt.img_size)
    img_batch = dataset:index(1, mask)
    return img_batch
end

function get_attr_vect(attr)
    local dataset_pos = get_dataset_attr(attr, 'pos')
    local dataset_neg = get_dataset_attr(attr, 'neg')
    local pos_vec_list = {}
    local neg_vec_list = {}
    for i = 1, 10 do
        local tmp_data_pos = get_batch(dataset_pos)
        local tmp_pos_vec = compute_vect(tmp_data_pos, encoder)
        pos_vec_list[i] = tmp_pos_vec

        local tmp_data_neg = get_batch(dataset_neg)
        local tmp_neg_vec = compute_vect(tmp_data_neg, encoder)
        neg_vec_list[i] = tmp_neg_vec
    end
    local pos_vec = torch.cat(pos_vec_list, 1)
    local neg_vec = torch.cat(neg_vec_list, 1)
    local ave_pos_vec = torch.mean(pos_vec, 1)
    local ave_neg_vec = torch.mean(neg_vec, 1)
    return {ave_pos_vec, ave_neg_vec}
end

ave_vec_results = get_attr_vect(opt.selected_attr)
ave_pos_vec = ave_vec_results[1]
ave_neg_vec = ave_vec_results[2]

local ave_vec = (ave_pos_vec - ave_neg_vec)

-- input images
img = image.load(opt.image_folder .. opt.test_image, 3) -- 001290
img = image.scale(img, 128):reshape(1, 3, 128, 128)
-- need a batch >= 2 for BN, it is ugly but works
input = torch.expand(img, 2, 3, 128, 128)
input = torch.cat(img:clone(), img:clone(), 1)

for i = 1, input:size(1) do
   input[i] = util.preprocess(input[i]:float():clone())
end    
input = input:cuda()

local results = encoder:forward(input)
local mean = results[1]
local log_var = results[2]
local z = nn.Sampler():forward({mean, log_var})

-- add attribute-specific vector
local attr_z = torch.zeros(opt.attr_scale, opt.nz):cuda()
for i = 1, opt.attr_scale do
    attr_z[i] = z[1]:clone() + ave_vec:clone():mul((i-1) / opt.attr_scale):mul(2)
end
z = attr_z

reconstruct_results = decoder:forward(attr_z)
fake = reconstruct_results

for i = 1, z:size(1) do
  fake[i] = util.deprocess(fake[i]:float():clone())
end

for i = 1, input:size(1) do
  input[i] = util.deprocess(input[i]:float():clone())
end

disp.image(fake, {win=1000, title='test'})
disp.image(input, {win=12, title='test input'})
-- image.save(opt.image_folder .. 'linear_' .. opt.test_image, image.toDisplayTensor{input=fake, nrow=opt.attr_scale})





