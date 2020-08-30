using kNNRecommenders
using Test

function test()
    println("Loading MovieLens 1M data...")
    mldata = MovieLens()
    load_data(mldata)
    split_data(mldata, random_state=220)

    transform_to_rating_matrix(mldata, 5)

    println("Performing ItemkNN")
    knn = kNNRecommender(260, 0.0, :tfidf)
    knn.rating_matrix = mldata.rating_matrix_train
    compute_similarity_matrix(knn)

    recalls = evaluate_recall(knn, 20, mldata.rating_matrix_test)
    mean_racall = sum(recalls) / length(recalls)
    @show mean_racall

    return mean_racall ≈ 0.2774901060050168
end

@test test()

