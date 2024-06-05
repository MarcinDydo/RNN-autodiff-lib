
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

    Wxh = randn(Float32, input_size, hidden_size) * 0.01 # wagi wejsciowe # U
    Whh = randn(Float32, hidden_size, hidden_size) * 0.01   # Losowa inicjalizacja wag ukrytych # W
    Why = randn(Float32, hidden_size, num_of_classes)  *0.01 # wagi wyjsciowe # V
    bh = randn(Float32, num_of_classes) * 0.01                # Losowa inicjalizacja biasów TODO:fix
    b = randn(Float32, num_of_classes) * 0.01
    hiddens = Vector{Matrix{Float32}}()
    outputs = Vector{Vector{Float32}}()

    input_layer = InputLayer(Wxh,Node())
    recusive_layer = RNNLayer(Node(), Whh,bh,hiddens)
    output_layer = OutputLayer(Node(),outputs,Why,b)
    return input_layer, recusive_layer, output_layer
end

function trainRNN(learning_rate::Int, epochs::Int)
    model = Model(initRNN(1,1)...)
    forward_pass(model, train_data.features)
    println(model.out.outputs)
    
end

function backward_pass()
    targets = train_data.targets[1:test_size]
    actuals = one_hot.(targets)
    loss_grad = outputs - actuals #mse gradient - mse = mean((actuals - outputs)**2)
    println(out.outputs)
end

trainRNN(1,1)
trainRNN(1,1)
trainRNN(1,1)