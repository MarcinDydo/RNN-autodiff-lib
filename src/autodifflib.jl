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
    #return 2*(y_pred - y_true) / prod(size(y_pred)[1:end-1])
    return (y_pred - y_true) 
end

function compute_grad(y_true, y_pred) #cross entropy loss?
    grad = []
    for i in 1:length(y_pred)
        g = zeros(length(y_pred[i]))
        y = y_true[i]
        y_hat = y_pred[i]
        if y_hat[argmax(y)] != 0
            g[argmax(y)] = -1 / y_hat[argmax(y)]
        end
        push!(grad,g) 
    end
    return grad
end

function cgrad()
    
end

function relu(matrix::AbstractMatrix{T}, c::Float32) where T<:Real
    #m = clamp.(matrix, -c, c)
    return max.(matrix, 0.0)
end

function relu_derivative(matrix::AbstractMatrix{T}, c::Float32) where T<:Real
    #m = clamp.(matrix, -c, c)
    map(x -> x > 0 ? 1 : 0, matrix)
end

function tanhip(matrix::AbstractMatrix{T}, c::Float32) where T<:Real
    m = clamp.(matrix, -c, c)
    return tanh.(m)
end

function tanhip_derivative(matrix::AbstractMatrix{T}) where T<:Real
    map(x -> 1 - tanh(x)^2, matrix)
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

