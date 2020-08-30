mutable struct kNNRecommender
    # hyper parameters
    topK::Int
    shrink::Float64
    weighting::Union{Nothing, Symbol}
    # parameters
    rating_matrix::SparseMatrixCSC
    similarity_matrix::SparseMatrixCSC
    
    kNNRecommender(topK, shrink) = new(topK, shrink)
    kNNRecommender(topK, shrink, weighting) = new(topK, shrink, weighting)
end

function tfidf(rating_matrix::SparseMatrixCSC)
    U, I, R = findnz(rating_matrix)
    n_users, n_items = size(rating_matrix)
        
    bincount = ones(n_users) # to avoid zero devision
    for u in U
        bincount[u] += 1
    end

    idf = log.(n_items ./ bincount)

    for j in 1:length(U)
        R[j] = R[j] * idf[U[j]]
    end

    return sparse(U, I, R)
end

function compute_similarity_matrix(rating_matrix::SparseMatrixCSC, topK::Int, shrink::Float64)
    # (user, item)^T * (user, item) -> (item, item)
    # Return S[i, j] where j is full items, and i is related items at topK
    
    simJ = Int[]
    simI = Int[]
    simS = Float64[]
    
    U, I, R = findnz(rating_matrix)
    n_users, n_items = size(rating_matrix)
    
    norms = sqrt.(sum(rating_matrix.^2, dims=1))
    
    for j in 1:n_items
        Uj, Rj = findnz(rating_matrix[:, j])
        simj = zeros(n_items)
        for (u, ruj) in zip(Uj, Rj)
            Iu, Ri = findnz(rating_matrix[u, :])
            for (i, rui) in zip(Iu, Ri)
                s = rui * ruj
                s /= norms[j] * norms[i] + shrink + 1e-6
                simj[i] += s
            end
        end
        
        arg_sort_i = sortperm(simj, rev=true)[2:topK+1]
        append!(simI, arg_sort_i)
        append!(simS, simj[arg_sort_i])
        append!(simJ, fill(j, length(arg_sort_i)))
    end
    
    return sparse(simI, simJ, simS)
end

function compute_similarity_matrix(knn::kNNRecommender)
    rating_matrix = knn.rating_matrix
    
    if knn.weighting == :tfidf
        rating_matrix = tfidf(rating_matrix)
    end
    
    knn.similarity_matrix = compute_similarity_matrix(rating_matrix, knn.topK, knn.shrink)
    
    return nothing
end

function predict(knn::kNNRecommender, uidx::Int, topN::Int)
    preds = sortperm((knn.rating_matrix[uidx, :]' * knn.similarity_matrix)', rev=true)
    
    I, R  = findnz(knn.rating_matrix[uidx, :])
    filter!(p->!(p in I), preds)
    
    return preds[1:min(topN, length(preds))]
end

function predict(knn::kNNRecommender, uidxs::Array{Int64,1}, topN::Int)
    return map(uidx->predict(knn, uidx, topN), uidxs)
end

function evaluate_recall(knn::kNNRecommender, topN::Int, rating_matrix_test::SparseMatrixCSC)
    recalls = Float64[]
    
    for uidx in 1:size(knn.rating_matrix)[1]
        gts, = findnz(rating_matrix_test[uidx, :])
        if length(gts) == 0 continue end
        
        preds = predict(knn, uidx, topN)
        tps = intersect(Set(preds), Set(gts))
        
        push!(recalls, length(tps) / length(gts))
    end
    
    return recalls
end