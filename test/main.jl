
using MLDatasets: MNIST
using Statistics
using ProgressMeter, BenchmarkTools
using LinearAlgebra
include("../src/graphutils.jl")
include("../src/autodifflib.jl")

# Ładowanie danych MNIST
train_data = MNIST(:train)
test_data  = MNIST(:test)

# Funkcja do ładowania danych
function initRNN(learning_rate::Int, epochs::Int) #returns layers
    num_of_classes = 10; #0-9 cyfry
    input_size = length(vec(train_data[1].features)) #feature_dim
    hidden_size = 100 #liczba neuronów ukrytych

    Wxh = randn(Float32, input_size, hidden_size) * 0.1 # wagi wejsciowe # U
    Whh = randn(Float32, hidden_size, hidden_size) * 0.1   # Losowa inicjalizacja wag ukrytych # W
    Why = randn(Float32, hidden_size, num_of_classes)  *0.1 # wagi wyjsciowe # V
    bh = randn(Float32, 1, hidden_size) * 0.1                # Losowa inicjalizacja biasów TODO:fix
    b = randn(Float32, 1, num_of_classes) * 0.1
    hiddens = Vector{Matrix{Float32}}()
    outputs = Vector{Vector{Float32}}()
    inputs = Vector{Matrix{Float32}}()

    Wxh_grad = zeros(Float32, input_size, hidden_size)
    Whh_grad = zeros(Float32, hidden_size, hidden_size)
    Why_grad = zeros(Float32, hidden_size, num_of_classes)
    bh_grad = zeros(Float32, 1, hidden_size) #TODO
    b_grad = zeros(Float32, 1, num_of_classes)

    input_layer = InputLayer(Wxh,Node(),inputs,Wxh_grad)
    recusive_layer = RNNLayer(Node(), Whh,bh,hiddens,Whh_grad,bh_grad)
    output_layer = OutputLayer(Node(),outputs,Why,b,Why_grad,b_grad)
    return input_layer, recusive_layer, output_layer
end

function trainRNN(learning_rate::Float64, epochs::Int) #TODO: zrobic to porzadnie XD
    model = Model(initRNN(1,1)...,learning_rate)
    println("iniitial W: ",model.in.Wxh[1,1])
    println("iniitial outputs: ",model.out.outputs)
    forward_pass(model, train_data.features)
    backward_pass(model, train_data.targets)
    model.hid.hiddens = Vector{Matrix{Float32}}()
    model.out.outputs = Vector{Vector{Float32}}()
    model.in.inputs = Vector{Matrix{Float32}}()
    println("calculated Wxh: ",argmax(model.in.Wxh), model.in.Wxh[argmax(model.in.Wxh)])
    forward_pass(model, train_data.features)
    backward_pass(model, train_data.targets)
    model.hid.hiddens = Vector{Matrix{Float32}}()
    model.out.outputs = Vector{Vector{Float32}}()
    model.in.inputs = Vector{Matrix{Float32}}()
    println("calculated Wxh: ",argmax(model.in.Wxh), model.in.Wxh[argmax(model.in.Wxh)])
    forward_pass(model, train_data.features)
    backward_pass(model, train_data.targets)
    model.hid.hiddens = Vector{Matrix{Float32}}()
    model.out.outputs = Vector{Vector{Float32}}()
    model.in.inputs = Vector{Matrix{Float32}}()
    println("calculated Wxh: ",argmax(model.in.Wxh), model.in.Wxh[argmax(model.in.Wxh)])
    forward_pass(model, train_data.features)
    backward_pass(model, train_data.targets)
    model.hid.hiddens = Vector{Matrix{Float32}}()
    model.out.outputs = Vector{Vector{Float32}}()
    model.in.inputs = Vector{Matrix{Float32}}()
    println("calculated Wxh: ",argmax(model.in.Wxh), model.in.Wxh[argmax(model.in.Wxh)])
end



#println(typeof(train_data.targets),size(train_data.targets),train_data.targets)

trainRNN(1e-2,1)