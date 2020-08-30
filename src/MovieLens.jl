mutable struct MovieLens
    dirpath
    
    df_rating_master
    df_user_master
    df_item_master
    
    item2iidx
    user2uidx
    
    df_rating_train
    df_rating_valid
    df_rating_test
    rating_matrix_train
    rating_matrix_valid
    rating_matrix_test
    
    MovieLens(dirpath = "../dataset/ml-1m") = new(dirpath)
end

function load_data(mldata::MovieLens)
    mldata.df_rating_master = CSV.read(joinpath(mldata.dirpath, "ratings.dat"), delim="::", 
                                        header=[:user_id, :item_id, :rating, :timestamp], 
                                        types=Dict(:user_id=>String, :item_id=>String, :rating=>Float64))
    mldata.df_user_master = CSV.read(joinpath(mldata.dirpath, "users.dat"), delim="::", 
                                        header=[:user_id, :gender, :age, :occupation, :zipcode], 
                                        types=Dict(:user_id=>String, :gender=>String, :age=>Int, :occupation=>Int))
    mldata.df_item_master = CSV.read(joinpath(mldata.dirpath, "movies.dat"), delim="::", 
                                        header=[:item_id, :title, :genres], 
                                        types=Dict(:item_id=>String, :title=>String, :genres=>String))
    
    mldata.user2uidx = Dict(uid=>i for (i, uid) in enumerate(mldata.df_user_master[:, :user_id]))
    mldata.item2iidx = Dict(iid=>i for (i, iid) in enumerate(mldata.df_item_master[:, :item_id]))
    
    return nothing
end

function split_data(mldata::MovieLens; testsize=0.2, validsize=0.1, random_state=46)
    trainsize = 1 - testsize - validsize
    
    fullsize = size(mldata.df_rating_master)[1]
    index_array = randperm(MersenneTwister(random_state), fullsize)
    
    train_last_index = floor(Int, fullsize*trainsize)
    valid_last_index = floor(Int, fullsize*(trainsize + validsize))
    
    mldata.df_rating_train = mldata.df_rating_master[index_array[1:train_last_index], :]
    mldata.df_rating_valid = mldata.df_rating_master[index_array[train_last_index+1:valid_last_index], :]
    mldata.df_rating_test = mldata.df_rating_master[index_array[valid_last_index+1:end], :]
    
    return nothing
end

function transform_to_rating_matrix(df_rating, user2uidx, item2iidx)
    U = Int[]
    I = Int[]
    R = Float64[]
    
    for i in 1:size(df_rating)[1]
        row = df_rating[i, :]
        
        uid = row.user_id
        iid = row.item_id
        r = row.rating
        
        if r != 0
            push!(U, user2uidx[uid])
            push!(I, item2iidx[iid])
            push!(R, r)
        end
    end 
    
    return sparse(U, I, R)
end

function transform_to_rating_matrix(df_rating, user2uidx, item2iidx, threshold_rating::Real)
    U = Int[]
    I = Int[]
    R = Float64[]
    
    for i in 1:size(df_rating)[1]
        row = df_rating[i, :]
        
        uid = row.user_id
        iid = row.item_id
        r = row.rating
        
        if r >= threshold_rating
            push!(U, user2uidx[uid])
            push!(I, item2iidx[iid])
            push!(R, 1)
        end
    end 
    
    return sparse(U, I, R)
end

function transform_to_rating_matrix(mldata::MovieLens)
    mldata.rating_matrix_train = transform_to_rating_matrix(mldata.df_rating_train, mldata.user2uidx, mldata.item2iidx)
    mldata.rating_matrix_valid = transform_to_rating_matrix(mldata.df_rating_valid, mldata.user2uidx, mldata.item2iidx)
    mldata.rating_matrix_test = transform_to_rating_matrix(mldata.df_rating_test, mldata.user2uidx, mldata.item2iidx)
    
    return
end 

function transform_to_rating_matrix(mldata::MovieLens, threshold_rating)
    mldata.rating_matrix_train = transform_to_rating_matrix(mldata.df_rating_train, mldata.user2uidx, mldata.item2iidx, threshold_rating)
    mldata.rating_matrix_valid = transform_to_rating_matrix(mldata.df_rating_valid, mldata.user2uidx, mldata.item2iidx, threshold_rating)
    mldata.rating_matrix_test = transform_to_rating_matrix(mldata.df_rating_test, mldata.user2uidx, mldata.item2iidx, threshold_rating)
    
    return
end 