
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
learning_rate = 15e-4
epochs = 150

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

function trainRNN(learning_rate::Float64, epochs::Int) 
    clamp = 5.0
    epoch_loss = 0
    batchsize = 1000
    model = Model(initRNN(1,1)...,learning_rate,x->x,mse_grad,clamp) #TODO implement loss
    for i in 1:epochs
        indices = shuffle(collect(1:batchsize))
        batch_x =  myshuffle(indices,batchsize,train_data.features) #shuffle??
        #debug(batch_x)
        batch_y =  [train_data.targets[j] for j in indices]
        println(i," epoch")
        forward_pass(model, batch_x, batchsize) 
        #println("fp", i, " Maximum and pos in output : ",argmax(model.out.outputs[10]), model.out.outputs[10][argmax(model.out.outputs[10])],model.out.outputs[10])
        println(i, " Accuracy: ", calculate_accuracy(model.out.outputs,batch_y))
        #println(i," Loss: ", mean(cross_entropy_loss.(batch_y,model.out.outputs)))
        println(i," bp")
        backward_pass(model, batch_y, batchsize)
        #println("bp", i, " Maximum and pos in Whh : ",argmax(model.hid.Whh), model.hid.Whh[argmax(model.hid.Whh)])
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

function myshuffle(indices,batchsize, mat)
    height = 28
    width = 28
    res = Array{Float32}(undef, height, width, batchsize)
    n =1
    #debug(indices)
    for i in indices
        #debug(train_data.features[:,:,i])
        res[:, :, n] = mat[:,:,i]
        n +=1
    end
    #debug(res)
    return res
end

function calculate_accuracy(predictions, targets)
    n_samples = length(targets)
    n_correct = 0
    #actuals = one_hot.(targets)
    #loss = mse.(actuals,model.out.outputs)
    #println(predictions,targets)
    #debug(predictions)
    #debug(targets)
    for i in 1:n_samples
        if argmax(predictions[i])[1]-1 == targets[i] #because of 0 
            #println("correct prediction! pred:",argmax(predictions[i]),"y:",predictions[i],"actual:",targets[i])
            n_correct+=1
        end
    end
    return n_correct/n_samples
end

function testRNN(model::Model)

    batchsize = length(test_data)
    println(length(test_data))
    resetRNN(model)
    indices = shuffle(collect(1:batchsize))
    println("random test sample:", display(indices))
    test_x =  myshuffle(indices,batchsize,test_data.features) #shuffle??
    #debug(batch_x)
    test_y =  [test_data.targets[j] for j in indices]
    forward_pass(model, test_x, batchsize) 

    println( " Accuracy: ", calculate_accuracy(model.out.outputs,test_y))
    println(" Loss: ", mean(cross_entropy_loss.(test_y,model.out.outputs)))
    println(" bp")

end

m = trainRNN(learning_rate,epochs)
for a in 1:100
    testRNN(m)
end
