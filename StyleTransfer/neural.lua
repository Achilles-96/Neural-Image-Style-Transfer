require 'nn'

require 'loadcaffe'

require 'optim'
require 'image'

-- Create a custom loss module for the content and insert this in the network

local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(target)
  parent.__init(self)
  self.target = target
  self.criterion = nn.MSECriterion()
  self.loss = 0
end

function ContentLoss:updateGradInput(input, gradOut)
  if self.target:nElement() == input:nElement() then
    -- Consider this as an ouput layer and calculate the error
    self.gradInput = self.criterion:backward(input, self.target)
  end
  -- Add this error to the current error
  self.gradInput:add(gradOut)
  return self.gradInput
end

function ContentLoss:updateOutput(input)
  self.output = input
  if self.target:nElement() == input:nElement() then
    self.loss = self.criterion:forward(input, self.target)
  end
  return self.output
end

-- End of content loss module


local new_net = nn.Sequential()
local vggnet = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'nn'):double()
local content_image = image.load('InputContentImages/golden_gate.jpg', 3)
-- Convert the image to a 512x512 size
content_image = image.scale(content_image, 512, 'bilinear')

-- Generally we select relu layers to apply content losses
local content_layers = {}
content_layers[1] = 'relu4_2'

local content_losses = {}
local content_idx = 1

for i=1, #vggnet do
  if content_idx <= #content_layers then
    local cur_layer = vggnet:get(i)
    local layer_type = torch.type(cur_layer)
    new_net:add(cur_layer)
    if cur_layer.name == content_layers[content_idx] then
      local target = new_net:forward(content_image):clone()
      local content_loss_module = nn.ContentLoss(target):double()
      table.insert(content_losses, content_loss_module)
      new_net:add(content_loss_module)
      content_idx = content_idx + 1
    end
  end
end

for i=1,#new_net.modules do
    local module = new_net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end

local new_img = nil
new_img = torch.randn(content_image:size()):double():mul(0.001)

local new_img_out = new_net:forward(new_img)
local dnew_img_out = torch.randn(#new_img_out):zero()

-- Use the lbfgs optimizer for training

local optim_state = nil
optim_state = {
  maxIter = 1000,
  verbose=true,
}

cur_iter = 1

function feval(x)
    new_net:forward(x)
    print 'Forward complete'
    local grad = new_net:updateGradInput(x, dnew_img_out)
    print 'GradInput generated'
    local loss = 0
    -- To calculate the loss currently
    for _, content_loss in ipairs(content_losses) do
      loss = loss + content_loss.loss
    end
    print 'Iteration: '
    print(cur_iter)
    print 'Total Loss: '
    print(loss)

    if 1 then
      print 'Saving image'
      local res = new_img:clone()
      res = res:double():mul(255.0)
      local filename = "output-" .. cur_iter .. ".png" 
      image.save(filename, res)
      print 'Completed saving image'
    end

    cur_iter = cur_iter + 1
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
end

print('Running optimizer' .. ' now....')
local x, lossx = optim.lbfgs(feval, new_img, optim_state)
