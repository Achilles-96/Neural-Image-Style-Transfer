#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"
require "image"
require "cunn"

correct_matches = 0
accuracy = 0
total = 0

use_cuda = 1

cnn = torch.load('model.torch')

if use_cuda == 1 then
    cnn = cnn:cuda()
end

print(cnn)

data_subset = torch.load('test_batch.t7', 'ascii')
image_data = data_subset.data:t()
image_labels = data_subset.labels[1]

answers = torch.Tensor(10,1)
for p=1,10 do
    answers[p] = 0
end

no_of_tests = 10000
-- Test the network
for q=1,no_of_tests do
    local cur_image = image_data[q]
    local cur_r = cur_image[{{1,1024}}]
    local cur_g = cur_image[{{1025,2048}}]
    local cur_b = cur_image[{{2049,3072}}]
    -- Generate an image of size 3*32*32
    -- First is the color channel, second is the height and last is the width
    local input = torch.Tensor(3,32,32)
    for j=1,32 do
        for k=1,32 do
            input[1][j][k] = cur_r[(j-1)*32+k]
            input[2][j][k] = cur_g[(j-1)*32+k]
            input[3][j][k] = cur_b[(j-1)*32+k]
        end
    end
    local output_val = image_labels[p]
    input = input:double()
    input = input / 255.0
    if use_cuda == 1 then
        input = input:cuda()
    end
    results = cnn:forward(input):exp() -- Since we use logsoftmax, the output is the exponent of probabilites
    best_result = torch.max(results)
    print(results)
    local output_val = image_labels[q]
    local cur_acc = results[output_val+1]
    total = 0
    for p=1,10 do
        total = total + results[p]
        if results[p] == best_result then
            answer = p-1
        end
    end
    accuracy = accuracy + cur_acc / total
    if answer == output_val then
        correct_matches = correct_matches + 1
    end
    answers[answer+1][1] = answers[answer+1][1] + 1
    print(results)
    res="No"
    if output_val == answer then
        res="Yes"
    end
    print(q .. ' => ' .. output_val .. '|' .. answer .. ' => ' .. res)
end
print('Final Results:')
print('Correct Matches = ' .. correct_matches .. '/' .. no_of_tests)
print('Accuracy = ' .. accuracy / no_of_tests)
print(answers:t())