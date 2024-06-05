
using MLDatasets: MNIST
using Statistics
using ProgressMeter, BenchmarkTools
using LinearAlgebra
include("../src/graphutils.jl")

# Ładowanie danych MNIST
train_data = MNIST(:train)
test_data  = MNIST(:test)

# Funkcja do ładowania danych
function trainRNN(learning_rate::Int, epochs::Int)
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
    forward_pass(input_layer, recusive_layer, output_layer)
    
end


function forward_pass(in::InputLayer, hid::RNNLayer, out::OutputLayer)
    
    for i in 1:5
        input_value = transpose(vec(train_data[i].features))
        println("(input)multiplying",typeof(input_value),size(input_value),"with",typeof(in.Wxh),size(in.Wxh))
        tmp = input_value * in.Wxh
        println("result",typeof(tmp),size(tmp))
        in.node.state = tmp
        forward_step(in,hid,out)
    end
    println(out.outputs)
end


trainRNN(1,1)
trainRNN(1,1)
trainRNN(1,1)