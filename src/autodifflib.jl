
# Przykładowa funkcja w bibliotece
function one_hot(digit::Int)
    one_hot_vector = zeros(Int, 10) #TODO: change to num_of_classes
    one_hot_vector[digit + 1] = 1
    return one_hot_vector
end
