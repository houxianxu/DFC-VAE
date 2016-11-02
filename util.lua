local util = {}

function util.save(filename, net, gpu)

    net:float() -- if needed, bring back to CPU
    local netsave = net:clone()
    if gpu > 0 then
        net:cuda()
    end

    for k, l in ipairs(netsave.modules) do
        netsave.modules[k] = util.optimizemodule(netsave.modules[k])
    end

    if torch.type(netsave.output) == 'table' then
        for k, o in ipairs(netsave.output) do
            netsave.output[k] = o.new()
        end
    else
        netsave.output = netsave.output.new()
    end
    netsave.gradInput = netsave.gradInput.new()

    netsave:apply(function(m) if m.weight then m.gradWeight = nil; m.gradBias = nil; end end)

    torch.save(filename, netsave)
end

function util.optimizemodule(m)
    -- convert to CPU compatible model
    if torch.type(m) == 'cudnn.SpatialConvolution' then
        local new = nn.SpatialConvolution(m.nInputPlane, m.nOutputPlane,
                      m.kW, m.kH, m.dW, m.dH,
                      m.padW, m.padH)
        new.weight:copy(m.weight)
        new.bias:copy(m.bias)
        m = new
    elseif torch.type(l) == 'fbnn.SpatialBatchNormalization' then
        new = nn.SpatialBatchNormalization(m.weight:size(1), m.eps,
                       m.momentum, m.affine)
        new.running_mean:copy(m.running_mean)
        new.running_std:copy(m.running_std)
        if m.affine then
            new.weight:copy(m.weight)
            new.bias:copy(m.bias)
        end
        m = new
    elseif m['modules'] then
        for k, l in ipairs(m.modules) do
            m.modules[k] = util.optimizemodule(m.modules[k])
        end
        return m
    end

    -- clean up buffers
    m.output = m.output.new()
    m.gradInput = m.gradInput.new()
    m.finput = m.finput and m.finput.new() or nil
    m.fgradInput = m.fgradInput and m.fgradInput.new() or nil
    m.buffer = nil
    m.buffer2 = nil
    m.centered = nil
    m.std = nil
    m.normalized = nil
-- TODO: figure out why giant storage-offsets being created on typecast
    if m.weight then
        m.weight = m.weight:clone()
        m.gradWeight = m.gradWeight:clone()
        m.bias = m.bias:clone()
        m.gradBias = m.gradBias:clone()
    end

    return m
end

function util.load(filename, gpu)
   local net = torch.load(filename)
   net:apply(function(m) if m.weight then 
	    m.gradWeight = m.weight:clone():zero(); 
	    m.gradBias = m.bias:clone():zero(); end end)
   return net
end

function util.cudnn(net)
    for k, l in ipairs(net.modules) do
        -- convert to cudnn
        if torch.type(l) == 'nn.SpatialConvolution' and pcall(require, 'cudnn') then
            local new = cudnn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
						 l.kW, l.kH, l.dW, l.dH, 
						 l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            net.modules[k] = new
        end
    end
    return net
end

-- a function to do memory optimizations by 
-- setting up double-buffering across the network.
-- this drastically reduces the memory needed to generate samples.
function util.optimizeInferenceMemory(net)
    local finput, output, outputB
    net:apply(
        function(m)
            if torch.type(m):find('Convolution') then
                finput = finput or m.finput
                m.finput = finput
                output = output or m.output
                m.output = output
            elseif torch.type(m):find('ReLU') then
                m.inplace = true
            elseif torch.type(m):find('BatchNormalization') then
                outputB = outputB or m.output
                m.output = outputB
            end
    end)
end


function util.preprocess(img)
  local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function util.deprocess(img)
  local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(255.0)
  return img
end

return util
