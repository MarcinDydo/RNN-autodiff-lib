using LinearAlgebra
using Random
include("../src/graphutils.jl")
# Przykładowa funkcja w bibliotece
function one_hot(digit::Int64)
    one_hot_vector = zeros(Int, 10) #TODO: change to num_of_classes
    one_hot_vector[digit + 1] = 1
    return one_hot_vector
end

nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end]) 

function xavier_init(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    scale = Float32(gain) * sqrt(24.0f0 / sum(nfan(dims...)))
    (rand(rng, Float32, dims...) .- 0.5f0) .* scale
end

# Funkcja obliczająca gradient MSE
function mse_grad(y_true, y_pred)
    return 2*(y_pred - y_true) / prod(size(y_pred)[1:end-1])
end

function cross_entropy_loss(y_pred, y_true)
    # Ensure numerical stability and prevent log(0)
    epsilon = 1e-12
    y_pred = clamp.(y_pred, epsilon, 1 - epsilon)
    return -sum(y_true .* log.(y_pred))
end

function cross_grad(y_true, y_pred) #cross entropy loss with softmax
    return (y_pred - y_true) 
end

function relu(matrix::AbstractVector{T}, c = 5.0) where T<:Real
    m = matrix # clamp.(matrix, -c, c)
    return max.(m, 0.0)
end

function relu_derivative(matrix::AbstractVector{T}, c = 5.0) where T<:Real
    m = matrix # clamp.(matrix, -c, c)
    map(x -> x > 0 ? 1 : 0, m)
end

function tanhip(matrix::AbstractVector{T}, c=5.0) where T<:Real
    m = matrix # clamp.(matrix, -c, c)
    return tanh.(m)
end

function tanhip_derivative(matrix::AbstractVector{T}, c=5.0) where T<:Real
    m = matrix # clamp.(matrix, -c, c)
    return Vector{Float32}(map(x -> 1 - tanh(x)^2, m))
end

function softmax(matrix::AbstractMatrix{T}) where T<:Real
    v_max = maximum(matrix)
    exp_matrix = exp.(matrix .- v_max)
    sum_exp_matrix = sum(exp_matrix)
    return exp_matrix ./ sum_exp_matrix
end

function softmax_derivative(matrix::AbstractVector{T}) where T<:Real #actually jacobian
    n = length(matrix)
    # Define the mapping function for the Jacobian element
    jacobian_element(i, j) = i == j ? matrix[i] * (1 - matrix[i]) : -matrix[i] * matrix[j]
    return [jacobian_element(i, j) for i in 1:n, j in 1:n]

end

function debug(mat)
    println("typeof mat:", display(mat))
end


function myshuffle(indices,batchsize, mat)
    height = 28
    width = 28
    res = Array{Float32}(undef, height, width, batchsize)
    n =1
    for i in indices
        res[:, :, n] = mat[:,:,i]
        n +=1
    end
    return res
end

function calculate_accuracy(predictions, targets)
    n_samples = length(targets)
    n_correct = 0
    for i in 1:n_samples
        if argmax(predictions[i])[1]-1 == targets[i] #because of 0 
            n_correct+=1
        end
    end
    return n_correct/n_samples
end