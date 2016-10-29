require 'nn'
require 'loadcaffe'
require 'optim'
require 'image'

-- Define a network which computes the gram matrix

function GramMat()
  local network = nn.Sequential()
  network:add(nn.View(-1):setNumInputDims(2))
  local conc = nn.ConcatTable()
  conc:add(nn.Identity())
  conc:add(nn.Identity())
  network:add(conc)
  network:add(nn.MM(false,true))
  return network
end

-- Create a custom loss module for the image as a regularization

local RegularizeLoss, parent = torch.class('nn.RegularizeLoss', 'nn.Module')

function RegularizeLoss:__init()
  parent.__init(self)
  self.loss = 0
  self.criterion = nn.MSECriterion()
end

function RegularizeLoss:updateGradInput(input, gradOutput)
  local required = torch.randn(input:size()):zero()
  self.gradInput = self.criterion:backward(input, required):div(input:nElement())
  self.gradInput:mul(1e8)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

function RegularizeLoss:updateOutput(input)
  self.output = input
  local required = torch.randn(input:size()):zero()
  self.loss = self.criterion:forward(input, required) / (input:nElement())
  self.loss = self.loss * (1e8)
  return self.output
end

-- Create a custom loss module for the style and insert this in the network

local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(target)
  parent.__init(self)
  self.target = target
  self.loss = 0
  self.gram = GramMat()
  self.criterion = nn.MSECriterion()
end

function StyleLoss:updateGradInput(input, gradOutput)
  local G = self.gram:forward(input)
  G:div(input:nElement())
  -- This is the error at the gram matrix output
  local dG = self.criterion:backward(G, self.target)
  dG:div(input:nElement())
  -- Backprop the error to get the deltas at the input to the gram matrix (ie the deltas at this StyleLoss layer)
  self.gradInput = self.gram:backward(input, dG)
  -- Add this error to the current error
  self.gradInput:mul(4e6)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

function StyleLoss:updateOutput(input)
  local G = self.gram:forward(input)
  G:div(input:nElement())
  self.loss = self.criterion:forward(G, self.target)
  self.loss = self.loss * 4e6
  self.output = input
  return self.output
end

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
  self.gradInput:mul(5.0)
  self.gradInput:add(gradOut)
  return self.gradInput
end

function ContentLoss:updateOutput(input)
  self.output = input
  if self.target:nElement() == input:nElement() then
    self.loss = self.criterion:forward(input, self.target)
    self.loss = self.loss * 5.0
  end
  return self.output
end

local new_net = nn.Sequential()
local vggnet = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'nn'):double()
local content_image = image.load('InputContentImages/tubingen.jpg', 3)
local style_image = image.load('InputStyleImages/shipwreck.jpg', 3)
-- Convert the image to a 512x512 size
content_image = image.scale(content_image, 512, 'bilinear')
style_image = image.scale(style_image, 512, 'bilinear')

-- Generally we select relu layers at the further end of the network to apply content losses
local content_layers = {}
-- Generally we select the relu layers at the start of the network to apply the style losses
local style_layers = {}
content_layers[1] = 'relu4_2'
style_layers[1] = 'relu1_1'
style_layers[2] = 'relu2_1'
style_layers[3] = 'relu3_1'

local content_losses = {}
local style_losses = {}
local content_idx = 1
local style_idx = 1

regularize_loss_module = nn.RegularizeLoss()
new_net:add(regularize_loss_module)
for i=1, #vggnet do
  if content_idx <= #content_layers or style_idx <= #style_layers then
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
    if cur_layer.name == style_layers[style_idx] then
      local GramM = GramMat()
      local target_layer = new_net:forward(style_image):clone()
      local target = GramM:forward(target_layer):clone()
      target:div(target_layer:nElement())
      local style_loss_module = nn.StyleLoss(target):double()
      table.insert(style_losses, style_loss_module)
      new_net:add(style_loss_module)
      style_idx = style_idx + 1
    end
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
    print 'Checking'
    new_net:forward(x)
    print 'Forward complete'
    local grad = new_net:updateGradInput(x, dnew_img_out)
    print 'GradInput generated'
    local c_loss = 0
    local s_loss = 0
    local loss = 0
    -- To calculate the loss currently
    for _, content_loss in ipairs(content_losses) do
      c_loss = loss + content_loss.loss
    end
    print 'Content Loss'
    print (c_loss)
    for _, style_loss in ipairs(style_losses) do
      s_loss = loss + style_loss.loss
    end
    r_loss = regularize_loss_module.loss
    print 'Style Loss'
    print (s_loss)
    print 'Regularize Loss'
    print (r_loss)
    print 'Iteration: '
    print(cur_iter)
    loss = s_loss + c_loss + r_loss
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
