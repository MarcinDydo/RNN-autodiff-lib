#DICLAIMER: this coputational graph approach only supports making recurisve neural networks
using LinearAlgebra
abstract type Layer end

mutable struct InputLayer <: Layer 
    activation::Function #
    derivative::Function
    state::Union{Vector{Float32}, Nothing} #state of nodes as a vector #
    weights::Matrix{Float32} #w * input = output #
    weights_grad::Matrix{Float32} #used in backpropagarion
    bias::Matrix{Float32} 
    bias_grad::Matrix{Float32}                          
    inputs::Vector{Vector{Float32}} #
    c::Float32
end

mutable struct HiddenLayer <: Layer 
    activation::Function #
    derivative::Function
    state::Union{Vector{Float32}, Nothing} #state of nodes as a vector #
    weights::Matrix{Float32} #w * input = output #
    weights_grad::Matrix{Float32} #used in backpropagarion
    bias::Matrix{Float32} 
    bias_grad::Matrix{Float32}
    hiddens::Vector{Vector{Float32}} 
    c::Float32
end

mutable struct OutputLayer <: Layer
    activation::Function #
    derivative::Function
    state::Union{Vector{Float32}, Nothing} #state of nodes as a vector #
    weights::Matrix{Float32} #w * input = output #
    weights_grad::Matrix{Float32} #used in backpropagarion
    bias::Matrix{Float32} 
    bias_grad::Matrix{Float32}
    outputs::Vector{Vector{Float32}}
    c::Float32
end

struct Model
    layers::Vector{Layer} #layers are numbered from 1 (input) to n (output)
    learning_rate::Float32
    loss_function::Function
    loss_gradient::Function
end

#Forward  
function forward(layer::InputLayer, input::Vector{Float32})
    push!(layer.inputs, input)
    z = (layer.weights * input) + layer.bias
    layer.state = vec(layer.activation(z))
    return layer.state
end

function forward(layer::HiddenLayer, input::Vector{Float32}, prev_state::Union{Vector{Float32},Nothing}) #input is a vector from previous layer
    if isnothing(prev_state) 
        z = input
    else
        z = vec(input + layer.weights * prev_state + layer.bias) #recursive logic for hidden nodes
    end
    layer.state = vec(layer.activation(z,layer.c))
    push!(layer.hiddens, layer.state)
    return layer.state
end

function forward(layer::OutputLayer, input::Vector{Float32})
    z = layer.weights * input + layer.bias
    layer.state = vec(layer.activation(z))
    push!(layer.outputs, layer.state)
    return layer.state
end

function forward_pass(model::Model, features::Array{Float32,3}, batchsize::Int)
    for i in 1:batchsize
        input_value = vec(features[:,:,i])
        for j in 1:length(model.layers)
            if j == 1
                forward(model.layers[j] , input_value) 
                continue
            elseif j == length(model.layers)
                forward(model.layers[j] , model.layers[j-1].state)
            else
                forward(model.layers[j] , model.layers[j-1].state, model.layers[j].state)       
            end
        end
    end
end

#Backpropagarion

function backward_pass(model::Model, actuals::Vector{Vector{Int}} , batchsize::Int)
    n = length(model.layers)
    loss_grad = model.loss_gradient(actuals,model.layers[n].outputs) #now we have all output gradients relative to actual classes

    next = nothing

    for i in reverse(1:batchsize)
        l_grad = loss_grad[i]
        
        running_grad = backward(model.layers[n],l_grad, model.layers[n-1].hiddens[i])
        for j in reverse(2:n-1)
            if i > 1
                running_grad = backward(model.layers[j],running_grad,next,model.layers[j].hiddens[i],model.layers[j].hiddens[i-1])
            else
                running_grad = backward(model.layers[j],running_grad,next,model.layers[j].hiddens[i],zeros(Float32,size(model.layers[j].hiddens[i])))
            end#końcowy case
        end
        backward(model.layers[1],running_grad, model.layers[1].inputs[i])
    end

    #update wag
    update_parameters(model)

end

function backward(layer::OutputLayer, dL_dy::Vector{Float32}, hid_state::Vector{Float32}) # gradient of the loss with respect to the predictions 
    # Calculate gradients for output layer
    layer.weights_grad += dL_dy * hid_state'
    layer.bias_grad .+= mean(dL_dy)
    # Gradient to propagate to the hidden layer
    return transpose(layer.weights) * dL_dy
end

function backward(layer::HiddenLayer, prev_grad::Vector{Float32}, next_grad::Union{Vector{Float32},Nothing}, hid_state::Vector{Float32}, next_state::Vector{Float32})
    if isnothing(next_grad)
        grad = prev_grad
    else
        grad = prev_grad + layer.weights' * next_grad
    end
    
    grad = grad .* layer.derivative(hid_state,layer.c)
    # Calculate gradients for hidden layer
    layer.weights_grad += grad * next_state'
    layer.bias_grad .+= mean(grad)
    # Gradient to propagate to the output layer
    return grad
end

function backward(layer::InputLayer, grad::Vector{Float32}, in_state::Vector{Float32})
    layer.weights_grad += grad * in_state'
    layer.bias_grad .+= mean(grad)
    return
end

function update_parameters(model::Model)
    lr = model.learning_rate
    for j in 1:length(model.layers)
        model.layers[j].weights -= model.layers[j].weights_grad * lr
        replace!(model.layers[j].weights , NaN=>0.0)
        model.layers[j].bias -= model.layers[j].bias_grad * lr
        replace!(model.layers[j].bias , NaN=>0.0)
    end
end

function reset_layers(model::Model)
    for j in 1:length(model.layers)
        if j == 1
            empty!(model.layers[j].inputs)
        elseif j == length(model.layers)
            empty!(model.layers[j].outputs)
        else
            empty!(model.layers[j].hiddens)
        end
        model.layers[j].weights_grad = zeros(size(model.layers[j].weights_grad))
        model.layers[j].bias_grad = zeros(size(model.layers[j].bias_grad))
    end
end
