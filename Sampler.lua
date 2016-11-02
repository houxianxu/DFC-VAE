require 'nn'
require 'cunn'


local Sampler, parent = torch.class('nn.Sampler', 'nn.Module')

function Sampler:__init()
    parent.__init(self)
    self.gradInput = {}
    self.output = self.output:cuda()
end 

function Sampler:updateOutput(input)
    self.eps = self.eps or input[1].new()
    self.eps:resizeAs(input[1]):copy(torch.randn(input[1]:size()))

    self.output = self.output or self.output.new()
    self.output:resizeAs(input[2]):copy(input[2])
    self.output:mul(0.5):exp():cmul(self.eps)

    self.output:add(input[1])
    return self.output
end

function Sampler:updateGradInput(input, gradOutput)
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)
    
    self.gradInput[2] = self.gradInput[2] or input[2].new()
    self.gradInput[2]:resizeAs(gradOutput):copy(input[2])
    
    self.gradInput[2]:mul(0.5):exp():mul(0.5):cmul(self.eps)
    self.gradInput[2]:cmul(gradOutput)
    return self.gradInput
end


function coordinates(batch_size, x_dim, y_dim, scale)
    n_pixel = x_dim * y_dim
    x_range = (torch.range(1, x_dim) - (x_dim + 1)/2):div(x_dim-1):mul(2):mul(scale)
    y_range = (torch.range(1, y_dim) - (y_dim + 1)/2):div(y_dim-1):mul(2):mul(scale)
    x_mat = torch.repeatTensor(x_range, y_dim, 1)
    y_mat = x_mat:transpose(1, 2):clone()
    r_mat = (torch.cmul(x_mat, x_mat) + torch.cmul(y_mat, y_mat)):sqrt()
    x_mat = torch.repeatTensor(x_mat:view(-1), batch_size, 1)
    y_mat = torch.repeatTensor(y_mat:view(-1), batch_size, 1)
    r_mat = torch.repeatTensor(r_mat:view(-1), batch_size, 1)
    x_unroll = x_mat:view(batch_size*n_pixel, 1)
    y_unroll = y_mat:view(batch_size*n_pixel, 1)
    r_unroll = r_mat:view(batch_size*n_pixel, 1)
    return torch.cat({x_unroll, y_unroll, r_unroll}, 2)
end


-- do not use
local SamplerXYR, parent = torch.class('nn.SamplerXYR', 'nn.Module')

function SamplerXYR:__init(x_dim, y_dim, batch_size)
    parent.__init(self)
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.batch_size = batch_size
    self.xyr_batch = coordinates(batch_size, x_dim, y_dim, 8):cuda()
end

function SamplerXYR:updateOutput(input)
    self.z_dim = input:size(2)
    self.z_module = nn.Sequential():cuda()
    self.z_module:add(nn.Replicate(self.x_dim * self.y_dim, 2):cuda())
    self.z_module:add(nn.Reshape(self.batch_size * self.x_dim * self.y_dim, self.z_dim):cuda())
    self.output = self.z_module:forward(input)
    self.output = torch.cat(self.output, self.xyr_batch, 2)
    return self.output
end

function SamplerXYR:updateGradInput(input, gradOutput)
    local gradInput_z = gradOutput[{ {}, {1, self.z_dim} }]:clone() -- batch x z x h x w
    self.gradInput = self.z_module:backward(input, gradInput_z)
    return self.gradInput
end


