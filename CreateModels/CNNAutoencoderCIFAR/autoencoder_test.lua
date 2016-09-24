#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"
require "image"
require "qtwidget"
require "os"

function get_image(cur_image)
	local cur_r = cur_image[{{1,1024}}]
	local cur_g = cur_image[{{1025,2048}}]
	local cur_b = cur_image[{{2049,3072}}]
	-- Generate an image of size 3*32*32
	-- First is the color channel, second is the height and last is the width
	local image = torch.Tensor(3,32,32)
	for j=1,32 do
		for k=1,32 do
			image[1][j][k] = cur_r[(j-1)*32+k]
			image[2][j][k] = cur_g[(j-1)*32+k]
			image[3][j][k] = cur_b[(j-1)*32+k]
		end
	end
	return image
end

correct_matches = 0
accuracy = 0
total = 0

cnn = torch.load('modelfinal.torch')
cnn = cnn:double()
torch.save('modelfinal.torch', cnn)

print(cnn)

data_subset = torch.load('test_batch.t7', 'ascii')
image_data = data_subset.data:t()
image_labels = data_subset.labels[1]

w = qtwidget.newwindow(100, 100)
w2 = qtwidget.newwindow(100, 100)

no_of_tests = 15
-- Test the network
for q=1,no_of_tests do
	local cur_image = image_data[q]
	image.display{image=(get_image(cur_image) * 255.0), win=w}
    local class = image_labels[p]
    input = cur_image:double()
    input = input / 255.0
    results = cnn:forward(input)
    image.display{image=(get_image(results) * 255.0), win=w2}
    os.execute("sleep " .. tonumber(5))
    print(q)
end
print('End-x:')
