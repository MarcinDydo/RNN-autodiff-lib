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
end

struct Model
    in::InputLayer
    hid::HiddenLayer
    out::OutputLayer
    learning_rate::Float32
    loss_function::Function
    loss_gradient::Function
    c::Float32
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
        z = vec(input + layer.weights * prev_state + layer.bias)
    end
    layer.state = vec(layer.activation(z))
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
        forward(model.in , input_value) 
        forward(model.hid , model.in.state, model.hid.state)
        forward(model.out , model.hid.state)
    end
end

#Backpropagarion

function backward_pass(model::Model, actuals::Vector{Vector{Int}} , batchsize::Int)
    lr = model.learning_rate #/ batchsize
    loss_grad = model.loss_gradient(actuals,model.out.outputs) #now we have all output gradients relative to actual classes
    next = nothing

    for i in reverse(1:batchsize)
        l_grad = loss_grad[i]
        output_grad = backward(model.out,l_grad,model.hid.hiddens[i])
        if i > 1
            hidden_grad = backward(model.hid,output_grad,next,model.hid.hiddens[i],model.hid.hiddens[i-1])
        else
            hidden_grad = backward(model.hid,output_grad,next,model.hid.hiddens[i],zeros(Float32,size(model.hid.hiddens[i])))
        end#końcowy case
        backward(model.in,hidden_grad,model.in.inputs[i])
    end
    #update wag
    model.in.weights -= model.in.weights_grad * lr
    replace!(model.in.weights , NaN=>0.0)
    model.hid.weights -= model.hid.weights_grad * lr
    replace!(model.hid.weights , NaN=>0.0)
    model.out.weights -= model.out.weights_grad * lr
    replace!(model.out.weights , NaN=>0.0)

    model.hid.bias -= model.hid.bias_grad * lr
    replace!(model.hid.bias , NaN=>0.0)
    model.out.bias -= model.out.bias_grad * lr
    replace!(model.out.bias , NaN=>0.0)
    model.in.bias -= model.in.bias_grad * lr
    replace!(model.out.bias , NaN=>0.0)

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
    
    grad = grad .* layer.derivative(hid_state)
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
