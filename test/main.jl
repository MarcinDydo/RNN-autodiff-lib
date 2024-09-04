
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
NUM_OF_CLASSES = 10; #0-9 cyfry
HIDDENSIZE = 64 #liczba neuronów ukrytych

LEARNING_RATE = 15e-3
EPOCHS = 5
CLAMP = 5.0
BATCHSIZE = 100
#activation functions (softmax, tanh) and their derivatives can be configured in initRNN function


# Funkcja do inicjalizowania wag danych
function initRNN() #returns layers
    input_size = length(vec(train_data[1].features)) #feature_dim

    rng = Random.default_rng()

    Wxh = xavier_init(rng, HIDDENSIZE, input_size) 
    Whh = xavier_init(rng, HIDDENSIZE, HIDDENSIZE)
    Why = xavier_init(rng, NUM_OF_CLASSES, HIDDENSIZE) 
    bx = xavier_init(rng, HIDDENSIZE,1) 
    bh = xavier_init(rng, HIDDENSIZE,1) 
    b = xavier_init(rng, NUM_OF_CLASSES,1)

    hiddens = Vector{Vector{Float32}}()
    outputs = Vector{Vector{Float32}}()
    inputs = Vector{Vector{Float32}}()

    Wxh_grad = zeros(Float32, HIDDENSIZE,  input_size)
    Whh_grad = zeros(Float32, HIDDENSIZE, HIDDENSIZE)
    Why_grad = zeros(Float32, NUM_OF_CLASSES, HIDDENSIZE)
    bx_grad = zeros(Float32, HIDDENSIZE,1) 
    bh_grad = zeros(Float32, HIDDENSIZE,1) 
    b_grad = zeros(Float32, NUM_OF_CLASSES,1)

    input_layer = InputLayer(x -> x,() -> 1, nothing, Wxh, Wxh_grad, bx, bx_grad, inputs, CLAMP)

    recusive_layer = HiddenLayer(tanhip, tanhip_derivative, nothing, Whh, Whh_grad, bh, bh_grad, hiddens, CLAMP)

    output_layer = OutputLayer(softmax, x -> x, nothing, Why, Why_grad, b, b_grad, outputs, CLAMP)

    return [input_layer, recusive_layer, output_layer]
end

# Funkcja do ładowania danych
function RNNloader()
    l = length(train_data.targets)  # Zakła
    indices = shuffle(collect(1:l))  # Tasowanie indeksów
    batches = floor(Int, l/BATCHSIZE)

    batch_x = Vector{Array{Float32, 3}}()
    batch_y = Vector{Vector{Int64}}()

    # Ładowanie partii danych
    for i in 1:BATCHSIZE:batches*BATCHSIZE
        push!(batch_x, train_data.features[:, :, indices[i:i+BATCHSIZE-1]])
        push!(batch_y, train_data.targets[indices[i:i+BATCHSIZE-1]])
    end

    return batch_x, batch_y, batches
end


function trainRNN() 
    model = Model(initRNN(), LEARNING_RATE, x->x , cross_grad)
    
    for i in 1:EPOCHS
        batch_x, batch_y , b = RNNloader()

        for i in 1:b
            forward_pass(model, batch_x[i], BATCHSIZE) 
        
            backward_pass(model, one_hot.(batch_y[i],NUM_OF_CLASSES), BATCHSIZE)
            reset_layers(model)
        end
        testRNN(model,i)
        reset_layers(model)
    end
    return model
end


function testRNN(model::Model, i::Int)

    reset_layers(model)
    test_x = test_data.features 
    test_y = test_data.targets
    forward_pass(model, test_x, length(test_data)) 

    println(i, "th epoch test accuracy: ", calculate_accuracy(model.layers[length(model.layers)].outputs,test_y))
end

@time begin
    m = trainRNN()
    print()
end


