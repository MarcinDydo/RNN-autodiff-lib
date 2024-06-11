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
    activation::Function            # Funkcja aktywacji na razie jest relu lub tanh
    activation_d::Function 
end

mutable struct OutputLayer <:Layer
    node::Node
    outputs::Vector{Vector{Float32}}
    Why::Matrix{Float32} #Wagi wyjściowe
    b::Matrix{Float32} #bias 
    Why_grad::Matrix{Float32} 
    b_grad::Matrix{Float32} 
    activation::Function       #funkcja aktywacji narazie jest softmax
    #activation_d::Function
end

mutable struct Model
    in::InputLayer
    hid::RNNLayer
    out::OutputLayer
    learning_rate::Float32
    c::Float32
end

#FUNCTIONS
function forward_step(input::InputLayer, hidden::RNNLayer, output::OutputLayer,c::Float32)
    if isempty(hidden.hiddens)
        hidden.node.state = input.node.state
    else
        #println( " (hidden)multiplying " ,typeof(hidden.node.state),size(hidden.node.state), " with " ,typeof(hidden.Whh),size(hidden.Whh))
        hidden.node.state = (input.node.state + hidden.node.state) * hidden.Whh + hidden.bh
        #println("hidden", typeof(hidden.node.state), size(hidden.node.state))
    end
    
    #hidden.node.state = relu(input.node.state + hidden.node.state) #TODO: relu change to hidden.activation  
    hidden.node.state = hidden.activation(input.node.state + hidden.node.state,c) 
    #if any(isnan.(hidden.node.state))*
    #    println("there is NaN in hidden node: ")
    #    return -1
    #end  
    push!(hidden.hiddens, hidden.node.state)
    #println( " (after relu)multiplying " ,typeof(hidden.node.state),size(hidden.node.state), " with " ,typeof(output.Why),size(output.Why))
    output.node.state = (hidden.node.state * output.Why)  + output.b 
    output.node.state = output.activation(output.node.state) #softmax
    #if any(isnan.(output.node.state))
    #    println("there is NaN in out node: ")
    #    println(output.node.state)
    #    println(output.Why)
     #   println(hidden.node.state)
     #   println((hidden.node.state * output.Why)  + output.b )
     #   println(norm(hidden.node.state))
     #   return -1
    #end  
    #println( " klasa " ,argmax(output.node.state)[2], " pewność ", output.node.state[argmax(output.node.state)[2]])
    push!(output.outputs, vec(output.node.state)) #predykcja
    #return 1
end

function forward_pass(model::Model, features::Array{Float32,3}, batchsize::Int)
    #s_size = size(features)[3]
    for i in 1:batchsize #full forward pass TODO: change to 1:size of features
        input_value = transpose(vec(features[:,:,i]))
        push!(model.in.inputs, input_value) #later used for backpropagation
        #println("(input)multiplying",typeof(input_value),size(input_value),"with",typeof(model.in.Wxh),size(model.in.Wxh))
        model.in.node.state = input_value * model.in.Wxh
        forward_step(model.in,model.hid,model.out,model.c) 
        debug(model.hid.node.state)
        #if forward_step(model.in,model.hid,model.out,model.c) < 0
        #    println("iter of fail NaN = ",i)
        #    break
        #end
    end
end

function backward_pass(model::Model, targets::Vector{Int64}, batchsize::Int)
    #init
    #s_size = length(targets)
    actuals = one_hot.(targets)
    loss_grad = mse_grad(actuals,model.out.outputs) 
    #loss_grad = compute_grad(actuals,model.out.outputs)
    #debug(loss_grad)
    next_hidden = nothing
    #dh_next = zeros(size(model.hid.hiddens[1]))'

    #calculate gradients
    for i in reverse(1:batchsize)
        #softmax backwards is just pass 
        #derivative of output probability vector
        l_grad = loss_grad[i]'
        model.out.Why_grad += transpose(model.hid.hiddens[i]) * l_grad #first gradient for output weights
        #model.out.Why_grad = norm_clipping(model.out.Why_grad)
        model.out.b_grad .+= mean(l_grad) 
        #model.out.b_grad = norm_clipping(model.out.b_grad)
        debuguj(i,model.out.Why_grad)

        o_grad = l_grad * transpose(model.out.Why)
        if isnothing(next_hidden)
            h_grad = o_grad 
        else
            h_grad = o_grad + next_hidden * transpose(model.hid.Whh)
        end
        h_grad = h_grad .* model.hid.activation_d(model.hid.hiddens[i],model.c) #derivative of relu or tanhip
        next_hidden = h_grad
        if i > 1
            model.hid.Whh_grad += transpose(model.hid.hiddens[i-1]) * h_grad #second gradient for hidden weights
            #model.hid.Whh_grad = norm_clipping(model.hid.Whh_grad)
            model.hid.bh_grad .+= mean(h_grad)
            #model.hid.bh_grad = norm_clipping(model.hid.bh_grad)
        end
        debuguj(i,model.hid.Whh_grad)
        
        model.in.Wxh_grad += transpose(model.in.inputs[i]) * h_grad #output gradient
        #model.in.Wxh_grad = norm_clipping(model.in.Wxh_grad)
        debuguj(i,model.in.Wxh_grad)

        # Clip gradients to avoid exploding gradients
        for g in [model.out.Why_grad, model.hid.Whh_grad, model.in.Wxh_grad]
            g .= clamp.(g, -5, 5)
        end
        #TODO: biasy
    end

    #update wag
    model.in.Wxh -= model.in.Wxh_grad * model.learning_rate
    replace!(model.in.Wxh , NaN=>0.0)
    model.hid.Whh -= model.hid.Whh_grad * model.learning_rate
    replace!(model.hid.Whh , NaN=>0.0)
    model.out.Why -= model.out.Why_grad * model.learning_rate
    replace!(model.out.Why , NaN=>0.0)
    model.hid.bh -= model.hid.bh_grad * model.learning_rate
    replace!(model.hid.bh , NaN=>0.0)
    model.out.b -= model.out.b_grad * model.learning_rate
    replace!(model.out.b , NaN=>0.0)
end

function norm_clipping(grad, threshold=6000)
    if norm(grad) > threshold
        grad .*= (threshold / norm(grad))
    end
    return grad
end

function debuguj(i, mat)
    a=0
    if i % 50 == 0
        println(i, "th norma Whx_grad " ,norm(mat), "max", maximum(mat))
        a =1
    end
end

function debug(mat)
    println("typeof mat:", typeof(mat), size(mat), maximum(mat))
end