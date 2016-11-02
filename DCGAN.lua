require 'torch'
require 'cunn'
require 'cudnn'

local DCGAN = {}


function DCGAN.get_discriminator(nc, ndf)
    -- this code is based on DCGAN
    local netD = nn.Sequential()
    -- input is (nc) x 128 x 138
    netD:add(cudnn.SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf)):add(nn.LeakyReLU(0.2, true))
    -- input is (nc) x 64 x 64
    netD:add(cudnn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 32 x 32
    netD:add(cudnn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 16 x 16
    netD:add(cudnn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 8 x 8
    netD:add(cudnn.SpatialConvolution(ndf * 8, ndf * 16, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 16)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 4 x 4
    netD:add(cudnn.SpatialConvolution(ndf * 16, 1, 4, 4))
    netD:add(nn.Sigmoid())
    -- state size: 1 x 1 x 1
    netD:add(nn.View(1):setNumInputDims(3))
    -- state size: 1
    return netD
end

return DCGAN