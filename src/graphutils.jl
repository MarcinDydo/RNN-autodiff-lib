using LinearAlgebra
abstract type Layer end

mutable struct Node 
    state::Union{Matrix{Float32}, Nothing}
    Node() =  new()
end

#LAYERS
struct InputLayer <: Layer                         
    Wxh::Matrix{Float32} #macierz wag (dla danego punktu czasowego) przekształcająca wejście na stan ukryty.(input weights)
    node::Node #może być jeden lub więcej node-ów początkowych (value in given timestep)
end

mutable struct RNNLayer <: Layer #tutaj przechowuje gradienty i wagi
    node::Node
    Whh::Matrix{Float32} #macierz wag przekształcająca poprzedni stan ukryty (Hidden weights)
    bh::Vector{Float32} #Wektor biasów dodawanych do stanu ukrytego.
    #activation::Function            # Funkcja aktywacji na razie jest relu
    hiddens::Vector{Matrix{Float32}} #pamięć ukryta 
end

struct OutputLayer <: Layer
    node::Node
    outputs::Vector{Vector{Float32}}
    Why::Matrix{Float32} #Wagi wyjściowe
    b::Vector{Float32} #bias 
    #activation::Function       #funkcja aktywacji narazie jest softmax
end

#FUNCTIONS
function forward_step(input::InputLayer, hidden::RNNLayer, output::OutputLayer)
    if isempty(hidden.hiddens)
        hidden.node.state = input.node.state
    else
        println( " (hidden)multiplying " ,typeof(hidden.node.state),size(hidden.node.state), " with " ,typeof(hidden.Whh),size(hidden.Whh))
        hidden.node.state = (input.node.state + hidden.node.state) * hidden.Whh #+ hidden.bh
        println( " result " ,typeof(hidden.node.state),size(hidden.node.state))
    end
    hidden.node.state = max.(input.node.state + hidden.node.state, 0.0) #TODO: relu change to hidden.activation    
    push!(hidden.hiddens, hidden.node.state)
    println( " (after relu)multiplying " ,typeof(hidden.node.state),size(hidden.node.state), " with " ,typeof(output.Why),size(output.Why))
    output.node.state = (hidden.node.state * output.Why) # +output.b 
    println( " output " ,typeof(output.node.state),size(output.node.state))
    output.node.state = exp.(output.node.state) ./ sum(exp.(output.node.state)) #softmax
    println("output matrix ", output.node.state)
    println( " klasa " ,argmax(output.node.state)[2], " pewność ", output.node.state[argmax(output.node.state)[2]])
    push!(output.outputs, vec(output.node.state)) #predykcja
end
