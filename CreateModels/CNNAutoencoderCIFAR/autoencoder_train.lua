#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"
require "image"
require "math"
require "cutorch"
require "cunn"

classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

function error(a,b)
	local err = 0.0
	for j=1,32*32*3 do
		err = err + (a[j] - b[j])*(a[j]- b[j])
	end
	return err
end

new_model = 1
use_cuda = 1

-- Parametrs for SpatialConvolution = (inputlayers, outputlayers, kernel_width, kernel_height, x_stride, y_stride, x_padding, y_padding)
-- Parametrs for SpatialMaxPooling = (width, height, x_stride, y_stride, x_padding, y_padding)
-- Input = 3*32*32 image, Output = 10 sized vector
if new_model==1 then
    cnn = nn.Sequential()
    cnn:add(nn.View(3,32,32))
    cnn:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
    cnn:add(nn.SpatialConvolution(16, 8, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
    cnn:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    cnn:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialUpSamplingNearest(2))
    cnn:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialUpSamplingNearest(2))
    cnn:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialUpSamplingNearest(2))
    cnn:add(nn.SpatialConvolution(16, 3, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.Sigmoid(true))
    cnn:add(nn.View(3*32*32))
    
else
    cnn = torch.load('model.torch')
    print('Using existing network')
end

criterion = nn.BCECriterion()
if use_cuda == 1 then
	criterion = criterion:cuda()
	cnn = cnn:cuda()
end

-- Run the training 'iterations' number of times
iterations = 5

for tt=1,iterations do
	for iti=0,4 do
		data_subset = torch.load('data_batch_' .. (iti+1) .. '.t7', 'ascii')
		image_data = data_subset.data:t()
		image_labels = data_subset.labels[1]
		no_of_training_cases = 10000
	    for p=1,no_of_training_cases do
		    local cur_image = image_data[p]
	        input = cur_image:double()
	   	    input = input / 255.0
	   	    output = input
	   	    if use_cuda == 1 then
	   	    	input = input:cuda()
	   	    	output = output:cuda()
	   	    end
	        -- Forward prop in the neural network
	        local outputs_cur = cnn:forward(input)
	        local current_error = error(outputs_cur,input)
	        local errs = criterion:forward(outputs_cur, output)
	        local df_errs = criterion:backward(outputs_cur, output)
	        -- Reset gradient accumulation
	        cnn:zeroGradParameters()
	        -- Accumulate gradients and back propogate
	        cnn:backward(input, df_errs)
	        -- Update with a learning rate
	        cnn:updateParameters(0.04)
	        print(tt,iti+1,p,current_error)
        end
	end
	print('Save model after each training epoch')
	torch.save('model.torch', cnn)
end

print('Done training, saving model if needed')
torch.save('model.torch', cnn)
