
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
epochs = 9

# Funkcja do ładowania danych
function initRNN(learning_rate::Int, epochs::Int) #returns layers
    num_of_classes = 10; #0-9 cyfry
    input_size = length(vec(train_data[1].features)) #feature_dim
    hidden_size = 64 #liczba neuronów ukrytych

    rng = Random.default_rng()

    Wxh = xavier_init(rng, input_size, hidden_size) 
    Whh = xavier_init(rng, hidden_size, hidden_size)
    Why = xavier_init(rng, hidden_size, num_of_classes) 
    bh = xavier_init(rng, 1, hidden_size) 
    b = xavier_init(rng, 1, num_of_classes)

    hiddens = Vector{Matrix{Float32}}()
    outputs = Vector{Vector{Float32}}()
    inputs = Vector{Matrix{Float32}}()

    Wxh_grad = zeros(Float32, input_size, hidden_size)
    Whh_grad = zeros(Float32, hidden_size, hidden_size)
    Why_grad = zeros(Float32, hidden_size, num_of_classes)
    bh_grad = zeros(Float32, 1, hidden_size) 
    b_grad = zeros(Float32, 1, num_of_classes)

    input_layer = InputLayer(Wxh,Node(),inputs,Wxh_grad)
    recusive_layer = RNNLayer(Node(), Whh,bh,hiddens,Whh_grad,bh_grad,relu,relu_derivative)
    output_layer = OutputLayer(Node(),outputs,Why,b,Why_grad,b_grad,softmax)
    return input_layer, recusive_layer, output_layer
end

function trainRNN(learning_rate::Float64, epochs::Int) #TODO: zrobic to porzadnie XD
    clamp = 5.0
    epoch_loss = 0
    batchsize = 100 
    model = Model(initRNN(1,1)...,learning_rate,clamp)
    for i in 1:epochs
        #println("Maximum and pos in Whh: ",argmax(model.hid.Whh), model.hid.Whh[argmax(model.hid.Whh)])
        for j in 1:(length(train_data.targets) \ batchsize)
            batch_indices = randperm(length(train_data.features))[1:batchsize]
            debug(batch_indices)
            # Select the batch data
            batch_x =  train_data.features[batch_indices]
            batch_y =  train_data.targets[batch_indices]
            #println("Maximum and pos in Wxh: ",argmax(model.in.Wxh), model.in.Wxh[argmax(model.in.Wxh)])
            #println("iniitial outputs: ",model.out.outputs)
            forward_pass(model, batch_x, batchsize) #train_data.features powinno byc losowym batchem
            #actuals = one_hot.(train_data.targets)
            #grad = mse_grad(actuals,model.out.outputs)
            #println("fp", i, " Maximum and pos in hidden : ",argmax(model.hid.hiddens[10]), model.hid.hiddens[10][argmax(model.hid.hiddens[10])])
            println("fp", i, " Maximum and pos in output : ",argmax(model.out.outputs[10]), model.out.outputs[10][argmax(model.out.outputs[10])],model.out.outputs[10])
            println(i, "Accuracy: ", calculate_accuracy(model.out.outputs,batch_y))
            backward_pass(model,batch_y, batchsize)
            #epoch_loss += mean(mse.(actuals,model.out.outputs))
            #println("epoch loss ",epoch_loss)
            println("bp", i, " Maximum and pos in Whh : ",argmax(model.hid.Whh), model.hid.Whh[argmax(model.hid.Whh)])
            empty!(model.hid.hiddens)
            empty!(model.out.outputs)
            empty!(model.in.inputs)
        end
    end
end

function calculate_accuracy(predictions, targets)
    n_samples = length(targets)
    n_correct = 0
    #actuals = one_hot.(targets)
    #loss = mse.(actuals,model.out.outputs)
    #println(predictions,targets)
    debug(predictions)
    debug(targets)
    for i in 1:n_samples
        if argmax(predictions[i])[1]-1 == targets[i] #because of 0 
            #println("correct prediction! pred:",argmax(predictions[i]),"y:",predictions[i],"actual:",targets[i])
            n_correct+=1
        end
    end
    return n_correct/n_samples
end



#println(typeof(train_data.targets),size(train_data.targets),train_data.targets)

trainRNN(learning_rate,epochs)
