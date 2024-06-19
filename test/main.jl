
using MLDatasets: MNIST
using Statistics
using ProgressMeter, BenchmarkTools
using LinearAlgebra
using Random
include("../src/graphutils.jl")
include("../src/autodifflib.jl")

# Ładowanie danych MNIST
train_data = MNIST(:train)
test_data  = MNIST(:test)
learning_rate = 15e-3
epochs = 5

# Funkcja do ładowania danych
function initRNN(learning_rate::Int, epochs::Int) #returns layers
    num_of_classes = 10; #0-9 cyfry
    input_size = length(vec(train_data[1].features)) #feature_dim
    hidden_size = 64 #liczba neuronów ukrytych

    rng = Random.default_rng()

    Wxh = xavier_init(rng, hidden_size, input_size) 
    Whh = xavier_init(rng, hidden_size, hidden_size)
    Why = xavier_init(rng, num_of_classes, hidden_size) 
    bx = xavier_init(rng, hidden_size,1) 
    bh = xavier_init(rng, hidden_size,1) 
    b = xavier_init(rng, num_of_classes,1)

    hiddens = Vector{Vector{Float32}}()
    outputs = Vector{Vector{Float32}}()
    inputs = Vector{Vector{Float32}}()

    Wxh_grad = zeros(Float32, hidden_size,  input_size)
    Whh_grad = zeros(Float32, hidden_size, hidden_size)
    Why_grad = zeros(Float32, num_of_classes, hidden_size)
    bx_grad = zeros(Float32, hidden_size,1) 
    bh_grad = zeros(Float32, hidden_size,1) 
    b_grad = zeros(Float32, num_of_classes,1)

    input_layer = InputLayer(x -> x,() -> 1, nothing, Wxh, Wxh_grad, bx, bx_grad, inputs)

    recusive_layer = HiddenLayer(tanhip, tanhip_derivative, nothing, Whh, Whh_grad, bh, bh_grad, hiddens)

    output_layer = OutputLayer(softmax, x -> x, nothing, Why, Why_grad, b, b_grad, outputs)

    return input_layer, recusive_layer, output_layer
end

function RNNloader(batchsize::Int)
    l = length(train_data.targets)  # Zakła
    indices = shuffle(collect(1:l))  # Tasowanie indeksów
    batches = floor(Int, l/batchsize)

    batch_x = Vector{Array{Float32, 3}}()
    batch_y = Vector{Vector{Int64}}()

    # Ładowanie partii danych
    for i in 1:batchsize:batches*batchsize
        push!(batch_x, train_data.features[:, :, indices[i:i+batchsize-1]])
        push!(batch_y, train_data.targets[indices[i:i+batchsize-1]])
    end

    return batch_x, batch_y, batches
end



function trainRNN(learning_rate::Float64, epochs::Int) 
    clamp = 5.0
    batchsize = 100

    model = Model(initRNN(1,1)...,learning_rate,x->x,cross_grad,clamp) #TODO implement loss
    
    for i in 1:epochs
        batch_x, batch_y , b = RNNloader(batchsize)

        for i in 1:b
            forward_pass(model, batch_x[i], batchsize) 
        
            backward_pass(model, one_hot.(batch_y[i]), batchsize)
            resetRNN(model)
        end
        testRNN(model,i)
        resetRNN(model)
    end
    return model
end

function resetRNN(model::Model)
    empty!(model.hid.hiddens)
    empty!(model.out.outputs)
    empty!(model.in.inputs)
    model.hid.weights_grad = zeros(size(model.hid.weights_grad))
    model.out.weights_grad = zeros(size( model.out.weights_grad))
    model.in.weights_grad = zeros(size(model.in.weights_grad))

    model.hid.bias_grad = zeros(size(model.hid.bias_grad))
    model.out.bias_grad = zeros(size(model.out.bias_grad))
    model.in.bias_grad = zeros(size(model.in.bias_grad))
end

function testRNN(model::Model, i::Int)

    batchsize = length(test_data)
    resetRNN(model)
    indices = shuffle(collect(1:batchsize))
    test_x =  myshuffle(indices,batchsize,test_data.features)
    test_y =  [test_data.targets[j] for j in indices]
    forward_pass(model, test_x, batchsize) 

    println(i, "th epoch test accuracy: ", calculate_accuracy(model.out.outputs,test_y))
end

@time begin
    m = trainRNN(learning_rate,epochs)
    print()
end


