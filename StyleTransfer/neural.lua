require 'nn'
require 'loadcaffe'
require 'optim'
require 'image'
require 'tvloss'
require 'cunn'

local use_gpu = 1

-- Define a network which computes the gram matrix

function GramMat()
  local net = nn.Sequential()
  -- Set input as batches with the last 2 dimensions belonging to each input.
  -- The last 2 dimensions are merged and flattened into a 1D array
  net:add(nn.View(-1):setNumInputDims(2))
  -- A concat table essentially takes the input given to it and runs the
  -- different modules in the concat table in parallel.
  local concat = nn.ConcatTable()
  -- Identity passes the input as the output without any changes
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  -- MM multiplies two input matrices by computing a dot product. The parameters
  -- false, true indicate if the matrices should be transposed or not.
  net:add(nn.MM(false, true))
  if use_gpu == 1 then
    net = net:cuda()
  end
  return net
end

-- Create a custom loss module for the image as a regularization

local RegularizeLoss, parent = torch.class('nn.RegularizeLoss', 'nn.Module')

function RegularizeLoss:__init()
  parent.__init(self)
  self.loss = 0
  self.criterion = nn.MSECriterion()
  if use_gpu == 1 then
    self.criterion = self.criterion:cuda()
  end
end

function RegularizeLoss:updateGradInput(input, gradOutput)
  local required = torch.randn(input:size()):zero()
  if use_gpu == 1 then
    required = required:cuda()
  end
  self.gradInput = self.criterion:backward(input, required):div(input:nElement())
  self.gradInput:mul(5e4)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

function RegularizeLoss:updateOutput(input)
  self.output = input
  local required = torch.randn(input:size()):zero()
  if use_gpu == 1 then
    required = required:cuda()
  end
  self.loss = self.criterion:forward(input, required) / (input:nElement())
  self.loss = self.loss * (5e4)
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
  if use_gpu == 1 then
    self.criterion = self.criterion:cuda()
  end
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
  self.gradInput:mul(1e2)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

function StyleLoss:updateOutput(input)
  local G = self.gram:forward(input)
  G:div(input:nElement())
  self.loss = self.criterion:forward(G, self.target)
  self.loss = self.loss * 1e2
  self.output = input
  return self.output
end

-- Create a custom loss module for the content and insert this in the network

local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(target)
  parent.__init(self)
  self.target = target
  self.criterion = nn.MSECriterion()
  if use_gpu == 1 then
    self.criterion = self.criterion:cuda()
  end
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

-- Pre and post processing of images
-- Convert from [0,1] range to [0,255]
function preproc(img)
  -- This is what we observed from lots of training samples
  local mean = torch.DoubleTensor({104, 117, 124})
  img = img:mul(255.0)
  mean = mean:view(3, 1, 1):expandAs(img)
  img:add(-1, mean)
  return img
end

-- Undo the above preprocessing.
function postproc(img)
  local mean = torch.DoubleTensor({104, 117, 124})
  mean = mean:view(3, 1, 1):expandAs(img)
  img = img + mean
  img = img:div(255.0)
  return img
end


local new_net = nn.Sequential()
local vggnet = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'nn'):double()
local content_image = image.load('InputContentImages/brad_pitt.jpg', 3)
local style_image = image.load('InputStyleImages/color.jpg', 3)
-- Convert the image to a 512x512 size
content_image = preproc(image.scale(content_image, 512, 'bilinear'))
style_image = preproc(image.scale(style_image, 512, 'bilinear'))
gaussian_filter = torch.Tensor(3,3):fill(1)
gaussian_filter = gaussian_filter:div(gaussian_filter:sum())
content_image = image.convolve(content_image, gaussian_filter)

if use_gpu == 1 then
  content_image = content_image:cuda()
  style_image = style_image:cuda()
end

-- Generally we select relu layers at the further end of the network to apply content losses
local content_layers = {}
-- Generally we select the relu layers at the start of the network to apply the style losses
local style_layers = {}

content_layers[1] = 'relu4_2'

style_layers[1] = 'relu1_1'
style_layers[2] = 'relu2_1'
style_layers[3] = 'relu3_1'
style_layers[4] = 'relu4_1'

local content_losses = {}
local style_losses = {}
local content_idx = 1
local style_idx = 1

local tv_loss_module = nn.TVLoss()
local regularize_loss_module = nn.RegularizeLoss()
if use_gpu == 1 then
  tv_loss_module = tv_loss_module:cuda()
  regularize_loss_mmodule = regularize_loss_module:cuda()
end
new_net:add(tv_loss_module)
new_net:add(regularize_loss_module)
for i=1, #vggnet do
  if content_idx <= #content_layers or style_idx <= #style_layers then
    local cur_layer = vggnet:get(i)
    local layer_type = torch.type(cur_layer)
    new_net:add(cur_layer)
    if use_gpu == 1 then
      new_net = new_net:cuda()
    end
    if cur_layer.name == content_layers[content_idx] then
      local target = new_net:forward(content_image):clone()
      local content_loss_module = nn.ContentLoss(target):double()
      if use_gpu == 1 then
        content_loss_module = content_loss_module:cuda()
      end
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
      if use_gpu == 1 then
        style_loss_module = style_loss_module:cuda()
      end
      table.insert(style_losses, style_loss_module)
      new_net:add(style_loss_module)
      style_idx = style_idx + 1
    end
  end
end

local new_img = nil
new_img = torch.randn(content_image:size()):double():mul(0.001)
if use_gpu == 1 then
  new_img = new_img:cuda()
end

local new_img_out = new_net:forward(new_img)
local dnew_img_out = torch.randn(#new_img_out):zero()
if use_gpu == 1 then
  dnew_img_out = dnew_img_out:cuda()
end

-- Use the lbfgs optimizer for training

local optim_state = nil
optim_state = {
  maxIter = 1001,
  verbose=true,
}

cur_iter = 1

function feval(x)
    new_net:forward(x)
    local grad = new_net:updateGradInput(x, dnew_img_out)
    local c_loss = 0
    local s_loss = 0
    local loss = 0
    -- To calculate the loss currently
    for _, content_loss in ipairs(content_losses) do
      c_loss = loss + content_loss.loss
    end
    for _, style_loss in ipairs(style_losses) do
      s_loss = loss + style_loss.loss
    end
    r_loss = regularize_loss_module.loss
    loss = s_loss + c_loss + r_loss
    if cur_iter%1000 == 0 then
      print 'Content Loss'
      print (c_loss)
      print 'Style Loss'
      print (s_loss)
      print 'Regularize Loss'
      print (r_loss)
      print 'Iteration: '
      print(cur_iter)
      print 'Total Loss: '
      print(loss)
      print 'Saving image'
      local res = postproc(new_img:clone():double())
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
