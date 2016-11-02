require 'torch'
require 'cunn'
require 'cudnn'
local CVAE = {}

function CVAE.get_encoder(nc, ndf, latent_variable_size)
    local encoder = nn.Sequential()

    encoder:add(nn.SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.SpatialBatchNormalization(ndf))
    encoder:add(nn.LeakyReLU(0.2, true))

    encoder:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.SpatialBatchNormalization(ndf * 2))
    encoder:add(nn.LeakyReLU(0.2, true))

    encoder:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.SpatialBatchNormalization(ndf * 4))

    encoder:add(nn.LeakyReLU(0.2, true))

    encoder:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.SpatialBatchNormalization(ndf * 8))

    encoder:add(nn.LeakyReLU(0.2, true))
    encoder:add(nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.SpatialBatchNormalization(ndf * 8))

    encoder:add(nn.View(ndf * 8 * 4 * 4))

    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(ndf * 8 * 4 * 4, latent_variable_size))
    mean_logvar:add(nn.Linear(ndf * 8 * 4 * 4, latent_variable_size))

    encoder:add(mean_logvar)

    return encoder:cuda()
end


function CVAE.get_decoder(nc, ngf, latent_variable_size)
    local decoder = nn.Sequential()

    decoder:add(nn.Linear(latent_variable_size, ngf * 8 *2 * 4 * 4))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.View(ngf * 8 * 2, 4, 4))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
    decoder:add(cudnn.SpatialConvolution(ngf*8*2, ngf*8, 3, 3, 1, 1))
    decoder:add(cudnn.SpatialBatchNormalization(ngf*8,1e-3))
    decoder:add(nn.LeakyReLU(0.2, true))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
    decoder:add(cudnn.SpatialConvolution(ngf*8, ngf*4, 3, 3, 1, 1))
    decoder:add(cudnn.SpatialBatchNormalization(ngf*4,1e-3))
    decoder:add(nn.LeakyReLU(0.2, true))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
    decoder:add(cudnn.SpatialConvolution(ngf*4, ngf*2, 3, 3, 1, 1))
    decoder:add(cudnn.SpatialBatchNormalization(ngf*2,1e-3))
    decoder:add(nn.LeakyReLU(0.2, true))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
    decoder:add(cudnn.SpatialConvolution(ngf*2, ngf*1, 3, 3, 1, 1))
    decoder:add(cudnn.SpatialBatchNormalization(ngf*1,1e-3))
    decoder:add(nn.LeakyReLU(0.2, true))

    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
    decoder:add(cudnn.SpatialConvolution(ngf, nc, 3, 3, 1, 1))

    return decoder:cuda()
end



return CVAE