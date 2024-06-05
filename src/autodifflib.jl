using LinearAlgebra
include("../src/graphutils.jl")
# Przykładowa funkcja w bibliotece
function one_hot(digit::Int64)
    one_hot_vector = zeros(Int, 10) #TODO: change to num_of_classes
    one_hot_vector[digit + 1] = 1
    return one_hot_vector
end

# Funkcja obliczająca MSE
function mse(y_true, y_pred)
    return mean((y_true .- y_pred) .^ 2)
end

# Funkcja obliczająca gradient MSE
function mse_grad(y_true, y_pred)
    return y_pred - y_true
end


function gradient_descent() #TODO: make it return type ::function so is more abstract
    normalized = 1
end

function relu_derivative(matrix::AbstractMatrix{T}) where T<:Real
    map(x -> x > 0 ? 1 : 0, matrix)
    return matrix
end

function softmax_derivative(matrix::AbstractVector{T}) where T<:Real #actually jacobian
    n = length(matrix)
    # Define the mapping function for the Jacobian element
    jacobian_element(i, j) = i == j ? matrix[i] * (1 - matrix[i]) : -matrix[i] * matrix[j]
    return [jacobian_element(i, j) for i in 1:n, j in 1:n]

end