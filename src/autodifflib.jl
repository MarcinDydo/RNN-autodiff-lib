module autodifflib

# Tutaj importujesz potrzebne pakiety
using Flux

# Przykładowa funkcja w bibliotece
function hello(name::String)
    println("Hello, $name!")
end

export hello

end # module
