require 'src/content_loss'

require 'loadcaffe'

function nop()
  -- nop.  not needed by our net
end

function create_descriptor_net()
    
  local cnn = loadcaffe.load(opt.proto_file, opt.model_file, opt.backend):cuda()
  print(cnn)

  opt.content_layers = opt.content_layers or ''

  local content_layers = opt.content_layers:split(",") 

  -- Set up the network, inserting content loss modules
  local content_losses = {}
  local next_content_idx = 1
  local net = nn.Sequential()
  local vgg_conv = nn.Sequential()
  for i = 1, #cnn do
    if next_content_idx <= #content_layers then
      local layer = cnn:get(i)
      vgg_conv:add(layer)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      
      if layer_type == 'nn.SpatialConvolution' or layer_type == 'nn.SpatialConvolutionMM' or layer_type == 'cudnn.SpatialConvolution' then
        layer.accGradParameters = nop
      end
      
      if opt.vgg_no_pad and (layer_type == 'nn.SpatialConvolution' or layer_type == 'nn.SpatialConvolutionMM' or layer_type == 'cudnn.SpatialConvolution') then
          print (name, ': padding set to 0')
          layer.padW = 0 
          layer.padH = 0 
      end

      net:add(layer)
   
      ---------------------------------
      -- Content
      ---------------------------------
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)

        local this_contents = {}

        local target = torch.Tensor()

        local norm = false
        local loss_module = nn.ContentLoss(opt.content_weight, name, target, norm):cuda()
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
    end
  end

  net:add(nn.DummyGradOutput())

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' or torch.type(module) == 'nn.SpatialConvolution' or torch.type(module) == 'cudnn.SpatialConvolution' then
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
      
  return net, vgg_conv, content_losses
end
