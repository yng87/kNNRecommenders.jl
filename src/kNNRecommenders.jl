module kNNRecommenders

using CSV
using Random
using SparseArrays

export MovieLens, load_data, split_data, transform_to_rating_matrix
export kNNRecommender, compute_similarity_matrix, predict, evaluate_recall

include("MovieLens.jl")
include("ItemkNN.jl")

end # module
