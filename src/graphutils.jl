using LinearAlgebra
abstract type Layer end

mutable struct Node 
    state::Union{Matrix{Float32}, Nothing}
    Node() =  new()
end

#LAYERS
mutable struct InputLayer <:Layer                         
    Wxh::Matrix{Float32} #macierz wag (dla danego punktu czasowego) przekształcająca wejście na stan ukryty.(input weights)
    node::Node #może być jeden lub więcej node-ów początkowych (value in given timestep)
    inputs::Vector{Matrix{Float32}}
    Wxh_grad::Matrix{Float32}
end

mutable struct RNNLayer <:Layer #tutaj przechowuje gradienty i wagi
    node::Node
    Whh::Matrix{Float32} #macierz wag przekształcająca poprzedni stan ukryty (Hidden weights)
    bh::Matrix{Float32} #Wektor biasów dodawanych do stanu ukrytego.
    hiddens::Vector{Matrix{Float32}} #pamięć ukryta 
    Whh_grad::Matrix{Float32} 
    bh_grad::Matrix{Float32}
    #activation::Function            # Funkcja aktywacji na razie jest relu
end

mutable struct OutputLayer <:Layer
    node::Node
    outputs::Vector{Vector{Float32}}
    Why::Matrix{Float32} #Wagi wyjściowe
    b::Matrix{Float32} #bias 
    Why_grad::Matrix{Float32} 
    b_grad::Matrix{Float32} 
    #activation::Function       #funkcja aktywacji narazie jest softmax
end

mutable struct Model
    in::InputLayer
    hid::RNNLayer
    out::OutputLayer
    learning_rate::Float32
end

#FUNCTIONS
function forward_step(input::InputLayer, hidden::RNNLayer, output::OutputLayer)
    if isempty(hidden.hiddens)
        hidden.node.state = input.node.state
    else
        #println( " (hidden)multiplying " ,typeof(hidden.node.state),size(hidden.node.state), " with " ,typeof(hidden.Whh),size(hidden.Whh))
        hidden.node.state = (input.node.state + hidden.node.state) * hidden.Whh + hidden.bh
        #println("hidden", typeof(hidden.node.state), size(hidden.node.state))
    end
    
    hidden.node.state = max.(input.node.state + hidden.node.state, 0.0) #TODO: relu change to hidden.activation    
    push!(hidden.hiddens, hidden.node.state)
    #println( " (after relu)multiplying " ,typeof(hidden.node.state),size(hidden.node.state), " with " ,typeof(output.Why),size(output.Why))
    output.node.state = (hidden.node.state * output.Why)  + output.b 
    output.node.state = exp.(output.node.state) ./ sum(exp.(output.node.state)) #softmax
    #println( " klasa " ,argmax(output.node.state)[2], " pewność ", output.node.state[argmax(output.node.state)[2]])
    push!(output.outputs, vec(output.node.state)) #predykcja
end

function forward_pass(model::Model, features::Array{Float32,3})
    for i in 1:5  #full forward pass TODO: change to 1:size of features
        input_value = transpose(vec(features[:,:,i]))
        push!(model.in.inputs, input_value) #later used for backpropagation
        #println("(input)multiplying",typeof(input_value),size(input_value),"with",typeof(model.in.Wxh),size(model.in.Wxh))
        model.in.node.state = input_value * model.in.Wxh
        forward_step(model.in,model.hid,model.out)
    end
end

function backward_pass(model::Model, targets::Vector{Int64})
    #o_s = size(model.out.outputs)[1] #TODO: check if works
    actuals = one_hot.(targets[1:5])
    loss_grad = mse_grad(actuals,model.out.outputs)
    next_hidden = nothing
    for i in reverse(1:5)
        l_grad = transpose(loss_grad[i]) #softmax backwards is just pass
        model.out.Why_grad += transpose(model.hid.hiddens[i]) * l_grad
        model.out.b_grad .+= mean(l_grad) 
        o_grad = l_grad * transpose(model.out.Why)
        if isnothing(next_hidden)
            h_grad = o_grad 
        else
            h_grad = o_grad + next_hidden * transpose(model.hid.Whh)
        end
        h_grad = h_grad .* relu_derivative(model.hid.hiddens[i]) #derivative of relu
        next_hidden = h_grad
        if i > 1
            model.hid.Whh_grad += transpose(model.hid.hiddens[i-1]) * h_grad 
            model.hid.bh_grad .+= mean(h_grad)
        end
        
        model.in.Wxh_grad += transpose(model.in.inputs[i]) * h_grad

    end
    model.in.Wxh -= model.in.Wxh_grad * model.learning_rate
    model.hid.Whh -= model.hid.Whh_grad * model.learning_rate
    model.out.Why -= model.out.Why_grad * model.learning_rate
    model.hid.bh -= model.hid.bh_grad * model.learning_rate
    model.out.b -= model.out.b_grad * model.learning_rate
end