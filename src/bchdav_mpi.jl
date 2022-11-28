# module DBchdav


using Printf
using Statistics
using LowRankApprox
using MPI
using LinearAlgebra
using Test

# export dbchdav



# function dbchdav(DA, dim, nwant, comm_info, user_opts;X, Y, X_gather, X_gather_T, X_gather_vbuf, Y_gather, Y_gather_T, Y_gather_T_vbuf, verb=false)
function dbchdav(DA, dim, nwant, comm_info, user_opts; verb=false)    
    # bchdav_mpi.jl implements the multi-process block chebyshev-davidson method using MPI for 
    # computing the smallest eigenpairs of sparse symmetric/Hermitian problems.

    # Usage:
    #      [evals, eigV, kconv, history] = bchdav_mpi(DA, dim, nwant, comm_info, user_opts; verb=false)

    # DA is the corresponding part of the distributed matrix A for the eigenproblem, (dim=size(A,1)), 
    #   (the A can be input as a function for matrix-vector products)

    # nwant is the number of wanted eigen pairs of matrix A,

    # comm_info is a structure coontaining the followinf fields:

    #       comm -- the initial communicator of all processors;

    #   comm_row -- the communicator of a row of the 2D processor grid;

    #   comm_col -- the communicator of a column of the 2D processor grid;

    #  comm_size -- the size of the communicator comm;

    # comm_size_sq -- the square root of comm_size;

    #       rank -- the rank (0-indexed) of the processor in comm;

    #   rank_row -- the rank (0-indexed) of the processor in the communicator comm_row;

    #   rank_col -- the rank (0-indexed) of the processor in the communicator comm_col; 

    # info_cols_dist -- the partition scheme of the matrix of the eigenvectors, info_cols_dist[rank+1] denotes the number of rows owned by the processor;
    # 


    # opts is a structure containing the following fields: (order of field names 
    # does not matter. there is no need to specify all fields since most of the time
    # the default values should suffice.
    # the more critical fields that can affect performance are blk, polm, vimax)

    #        blk -- block size, 
    #               by default: blk = 3.
    #      polym -- the degree of the Chebyshev polynomial; 
    #               by default:  polym=20.
    #      vimax -- maximum inner-restart subspace (the active subspace) dimension, 
    #               by default:  vimax = max(max(5*blk, 30), ceil(nwant/4)).

    #   do_outer -- do outer restart (if no outer restart, inner restart may gradually 
    #               increase subspace dim until convergence if necessary)
    #               by default:  do_outer = logical(1).
    #      vomax -- maximum outer-restart subspace dimension;
    #               by default:  vomax= nwant+30.  

    #     filter -- filter method to be used;
    #               currently only two Chebshev filters are implemented,
    #               by default: filter=2. (the one with simple scaling)
    #        tol -- convergence tolerance for the residual norm of each eigenpair;
    #               by default:  tol =1e-8.
    #      itmax -- maximum iteration number;
    #               by default:  itmax= max(floor(dim/2), 300).
    #      ikeep -- number of vectors to keep during inner-restart,
    #               by default:  ikeep= max(floor(vimax/2), vimax-3*blk).

    #         v0 -- the initial vectors (can be of any size);
    #               by default: v0 = randn(dim,blk).
    #      displ -- information display level; 
    #               (<=0 --no output; 1--some output; >=2 --more output, 
    #                when displ>5, some expensive debugging calculations will be performed)
    #               by default: displ=1.
    #     chksym -- check the symmetry of the input matrix A.
    #               if chksym==1 and A is numeric, then isequal(A,A') is called.
    #               the default is not to check symmetry of A. 
    #      kmore -- additional number of eigen-pairs to be computed.
    #               by default, kmore=3.
    #        upb -- upper bound of all the eigenvalues of the input matrix A.
    #               by default: upb = 2.
    #        lwb -- lower bound of all the eigenvalues of the input matrix A.
    #               by default: lwb = 0.
    #    low-nwb -- lower bound of the unwanted eigenvalues of the inpure matrix A.
    #               by default: low_nwb = lwb + (upb - lwb) / 10.
    #    augment -- choose how many filtered vectors to keep in the basis,  
    #               by default augment=1,  only the last blk filtered vectors are kept;
    #               if augment=2,  the last 2*blk  filtered vectors are kept;
    #               if augment=3,  the last 3*blk  filtered vectors are kept.


    # ========== Output variables:

    #       evals:  converged eigenvalues (optional).

    #       eigV:  converged eigenvectors (optional, but since eigenvectors are
    #              always computed, not specifying this output does not save cputime).

    #      kconv:  number of converged eigenvalues (optional).

    #    history:  log information (optional)
    #              log the following info at each iteration step:

    #              history(:,1) -- iteration number (the current iteration step)
    #              history(:,2) -- cumulative number of matrix-vector products 
    #                              at each iteration, history(end,2) is the total MVprod.
    #              history(:,3) -- residual norm at each iteration
    #              history(:,4) -- current approximation of the wanted eigenvalues

    # ---------------------------------------------------------------------------
    

    global MVprod
    global MVcpu
    # initialize mat-vect-product count and mat-vect-product cpu to zero
    MVprod = 0
    MVcpu = 0

    global filt_non_mv_cput
    global filt_mv_cput 
    filt_non_mv_cput = 0
    filt_mv_cput = 0
    returnhere = 0

    # cputotal = cputime   #init the total cputime count
    global elapsedTime = Dict("total"=>0.0, "Cheb_filter"=>0.0, "Cheb_filter_n"=>0.0, "Cheb_filter_scal"=>0.0,
                "Cheb_filter_scal_n"=>0.0, "SpMM"=>0.0, "SpMM_n"=>0.0, "main_loop"=>0.0, "TSQR"=>0.0, "TSQR_n"=>0.0,
                "Inner_prod"=>0.0, "Inner_prod_n"=>0.0, "Hn"=>0.0, "Hn_n"=>0.0, "Norm"=>0.0, "Norm_n"=>0.0
    )

    
    comm = comm_info["comm"]
    # comm_T = comm_info["comm_T"]
    comm_row = comm_info["comm_row"]
    comm_rol = comm_info["comm_col"]
    rank = comm_info["rank"]
    rank_row = comm_info["rank_row"]
    rank_col = comm_info["rank_col"]
    info_cols_dist = comm_info["info_cols_dist"]
    comm_size = comm_info["comm_size"]
    comm_size_sq = comm_info["comm_size_sq"]
    r2c, counts_info, l2r, process_tree, process_tree_lvl = setup_process_tree(comm_size)
    comms_tree = create_communicators(r2c, rank, comm)

    elapsedTime["total"] = @elapsed begin

        #
        # Process inputs and do error-checking
        #

        # DA = varargin[1]

        # dim = size(DA,1)

        Anumeric = 1

        #  Set default values and apply existing input options:
        #  default opt values will be overwritten by user input opts (if exist).
        
        #  there are a few unexplained parameters which are mainly used to
        #  output more info for comparision purpose, these papameters can be 
        #  safely neglected by simply using the defaults.

        blk = 3
        opts=Dict([("blk", blk), ("filter", 2), ("polym", 20), ("tol", 1e-8),
                    ("vomax", nwant+30),  ("do_outer", true),
                    ("vimax", max(max(5*blk, 30), ceil(nwant/4))),
                    ("adjustvimax", true), 
                    ("itmax", max(floor(dim/2), 300)), ("augment", 1), 
                    ("chksym", false),  ("displ", 1),  ("kmore", 3), ("forArray", rand(5,5))])

        if !isa(user_opts,Dict)
            if rank == 0
                @printf("Options must be a dictionary. (note bchdav does not need mode)")
            end
        else
            # overwrite default options by user input options
            # opts = merge(user_opts, opts) 
            for key in keys(user_opts)
                opts[key] = user_opts[key]
                
            end       
        end

        # save opt values in local variables 
        blk = opts["blk"];  filter=opts["filter"];  polym=opts["polym"];  tol=opts["tol"];
        vomax=opts["vomax"];  vimax=opts["vimax"];  itmax=opts["itmax"];  
        augment=opts["augment"];  kmore=opts["kmore"];  displ=opts["displ"]; 

        X = zeros(Float64, blk, info_cols_dist[rank+1])

        if  haskey(opts, "v0")
            sizev0 = size(opts["v0"],1)
            if sizev0 < blk
            # @printf("*** input size(v0,2)=%i, blk=%i, augment %i random vectors\n",
                    # sizev0, blk, blk-sizev0)
                DV0 = vcat(opts["v0"], rand(blk-sizev0, info_cols_dist[rank+1]))
            else
                DV0 = opts["v0"]
            end
        else
            DV0 = randn(nwant, info_cols_dist[rank+1])
            sizev0 = blk
        end


        if opts["do_outer"] 
            vomaxtmp = max(min(nwant + 6*blk, nwant+30), ceil(nwant*1.2))
            if  vomax < vomaxtmp 
                if rank == 0
                    @printf("--> Warning: vomax=%i, nwant=%i, blk=%i\n", vomax, nwant, blk)
                end
                vomax = vomaxtmp
                if rank == 0
                    @printf("--> Warnning: increase vomax to %i\n",vomax)
                end
            end  
            if  vimax > vomax
                if rank == 0
                    @printf("--> Warning:  (vimax > vomax)  vimax=%i, vomax=%i\n", vimax, vomax)
                end
                vimax = max(min(6*blk, nwant), ceil(nwant/4))  #note vomax > nwant
                if rank == 0
                    @printf("--> reduce vimax to %i\n", vimax)
                end
            end
        end

        if  vimax < 5*blk
            if rank == 0 
                @printf("--> Warning:  (vimax < 5*blk)  vimax=%i, blk=%i\n", vimax, blk)
            end
            if  opts["adjustvimax"] 
                vimax = 5*blk
                if rank == 0
                    @printf("--> increase vimax to %i\n", vimax)   
                end     
            elseif 3*blk > vimax
                vimax = 3*blk
                if rank == 0
                    @printf("--> adjust vimax to %i\n", vimax)
                end
            end
        end
        if vimax > vomax 
            vomax = vimax+2*blk
        end
        ikeep = trunc(Int64, max(floor(vimax/2), vimax-3*blk))


        # ##################################################################

        #  Now start the main algorithm:

        # ##################################################################

        #  Comment: when the matrix A is large, passing A explicitly as a 
        #  variable in the function interface is not as efficient as using 
        #  A as a global variable. 
        #  In this code A is passed as a global variable when A is numeric. 


        longlog = 1

        #  Preallocate memory, useful if dim and vomax are large
        DV = randn(vomax, info_cols_dist[rank+1])
        DW = zeros(Float64, trunc(Int64, vimax), size(DV,2))
        Hn = zeros(trunc(Int64, vimax), trunc(Int64, vimax))
        evals = zeros(nwant)   
        resnrm = zeros(nwant,1)


        #  get the very important filtering upper bound. 
        #  if there is a user input upb, make sure the input upb is an upper bound, 
        #  otherwise convergence can be very slow. (safer to just let the LANCZ_bound() estimate an upper bound without a user input upb)
        #  an estimated lower bound can also be obtained via LANCZ_bound()

        lancz_step = 4
        # if  Anumeric > 0
        # elapsedTime["LANCZ_bound"] += @elapsed begin
        # upb, low_nwb, lowb, maxritz = LANCZ_bound(dim, lancz_step, DA)  
        # end
        # elapsedTime["LANCZ_bound_n"] += 1
        upb = 2.0
        lowb = 0.0

        if haskey(opts, "upb")
            # if opts["upb"] < upb
            #     if rank == 0
            #         @printf("user input upperbound may be too small, may NOT converge!! \n")
            #     end
            #     upb = opts["upb"]   #still use the user input upb, run at your own risk
            # end
            upb = opts["upb"]
        end

        if haskey(opts, "lwb")
            # if opts["lwb"] > lowb
            #     if rank == 0
            #         @printf("user input lowerbound may be too large, may NOT converge!! \n")
            #     end
            #     lowb = opts["lwb"]   #still use the user input lwb, run at your own risk
            # end
            lowb = opts["lwb"]
        end

        low_nwb = lowb + (upb - lowb)/20
        if haskey(opts, "low_nwb")
            low_nwb = opts["low_nwb"]   #still use the user input upb, run at your own risk
        end
        maxritz = low_nwb

        #
        # add some variables to measure computational cost
        #
        iter_total = 0           # init the total iteration number count
        orth_cputotal = 0        # init the total cputime for orthogonalization
        orth_flopstotal = 0      # init the total flops for orthogonalization
        refinement_cputotal = 0  # init the total cputime for rayleigh-ritz refinement
        nlog = 1

        kinner = 0;  kouter = 0;  # count the number of inner and outer restarts
        Hn_cpu=0; conv_cpu=0;  # temp variables, used only for measuring cpu

        # -----------------------------------------------------------------------------
        #  the two-indeces trick: use n and kact,  where n = kact+kconv, 
        #  n is the same as ksub in the paper, note that n is NOT the dimension 'dim'
        # -----------------------------------------------------------------------------
        n = 0        # n stores column # in subspace V 
        kact = 0     # kact counts the dimension of the active subspace
        kconv = 0    # init number of converged eigenvalues
        kconv1 = 1   # auxiliary, kconv1 always stores kconv+1 
        history = zeros(0,4)
        
        elapsedTime["TSQR"] += @elapsed begin
            # DGKS doesn't scale
            # DV[1:blk, :] = DGKS_1(DV0[1:blk, :], dim, comm_row, comm_col)
            DV[1:blk, :] = TSQR_1(DV0[1:blk, :], dim, process_tree_lvl, comms_tree, counts_info, rank, comm_row, comm_col)
        end
        elapsedTime["TSQR_n"] += 1

        ec = 0
        kinit = blk

        # global X = zeros(Float64, blk, size(DV,2))
        # global Y = zeros(Float64, blk, info_cols_dist[rank_col*comm_size_sq+rank_row+1])
        # Y_gather = zeros(Float64, size(DA, 1), blk)
        # X_gather_T = zeros(Float64, size(DA, 2), blk)
        # local_info_cols = info_cols_dist[rank_row*comm_size_sq+1:rank_row*comm_size_sq+comm_size_sq]
        # X_gather = Array{Float64}(undef, (blk, sum(local_info_cols)))
        # _counts = vcat([blk for i in 1:length(local_info_cols[:])]', info_cols_dist[rank_row*comm_size_sq+1:(rank_row+1)*comm_size_sq]')
        # global X_gather_vbuf = VBuffer(X_gather, vec(prod(_counts, dims=1)))
        # global Y_gather_T = zeros(Float64, blk, size(DA,1))
        # _counts = vcat([blk for i in 1:length(local_info_cols[:])]', info_cols_dist[rank_col*comm_size_sq+1:(rank_col+1)*comm_size_sq]')
        # global Y_gather_T_vbuf = VBuffer(Y_gather_T, vec(prod(_counts, dims=1)))
        inds = [ind for ind = 1:size(DA,1)]
        DE = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(DA, 1)), size(DA,1), size(DA,2)) : sparse([1], [1], [0.0], size(DA,1), size(DA,2))
        
        MPI.Barrier(comm)

        elapsedTime["main_loop"] = @elapsed begin
            while  iter_total <= itmax 
                
                iter_total = iter_total +1
                
                # DVtmp = zeros(Float64, blk, size(DV, 2))
                if  ec > 0  &&  kinit + ec <= sizev0
                    if  displ > 4
                        if rank == 0
                            @printf("---> using column [%i : %i] of initial vectors\n",kinit+1, kinit+ec) 
                        end
                    end
                    # Vtmp = hcat(Matrix{Float64}(opts["v0"][:,kinit+1:kinit+ec]), Matrix{Float64}(V[:,kconv1:kconv+blk-ec]))
                    X = vcat(DV0[kinit+1:kinit+ec, :], DV[kconv1:kconv+blk-ec, :])
                    kinit = kinit + ec
                else
                    # Vtmp = Matrix{Float64}(V[:,kconv1:kconv+blk])
                    X = DV[kconv1:kconv+blk, :]
                end

                elapsedTime["Cheb_filter_scal"] += @elapsed begin
                # Vtmp = Cheb_filter_scal(DA, Vtmp, polym, low_nwb, upb, lowb, augment)
                # X = Cheb_filter_scal(polym, X, Y, DA, DE, X_gather, X_gather_T, X_gather_vbuf, Y_gather, Y_gather_T, Y_gather_T_vbuf, 0, comm_row, comm_col)
                    X, _ = Cheb_filter_scal_1(polym, low_nwb, upb, lowb, X, DA, DE, info_cols_dist, rank_row, rank_col, comm_size_sq, 0, comm_row, comm_col)
                end
                elapsedTime["Cheb_filter_scal_n"] += 1
                
                #
                # make t orthogonal to all vectors in V
                #
                n = kconv + kact
                # orth_cpu = cputime
                orth_cputotal = @elapsed begin
                    # DGKS doesn't scale
                    orth_flops = 0 # delete this later
                    # X = DGKS(DV, 1:n, X, dim, comm_row, comm_col)
                    X, _ = TSQR(DV, 1:n, X, dim, process_tree_lvl, comms_tree, counts_info, rank, comm_row, comm_col)
                end
                orth_flopstotal = orth_flopstotal + orth_flops
                elapsedTime["TSQR"] += orth_cputotal
                elapsedTime["TSQR_n"] += 1
                kb = size(X,1)
                n1 = n+1
                kact = kact + kb
                n = kconv + kact   
                # V[:, n1:n] = Vtmp
               
                DV[n1:n,:] = X
                
                #
                # compute new matrix-vector product.
                #
                # if  Anumeric > 0
                elapsedTime["SpMM"] += @elapsed begin
                    DW[kact-kb+1:kact, :] = SpMM_A_1_w_E(DV[n1:n, :], DA, DE, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                end
                elapsedTime["SpMM_n"] += 1
                # else
                #     W[:, kact-kb+1:kact],_ =  duser_Hx(V[:, n1:n], A_operator)  
                # end

                
                # Hn_cpu0=cputime
                #
                # update Hn, compute only the active part (Hn does not include any deflated part)
                #
                # Hn[1:kact, kact-kb+1:kact] = V[:, kconv1:n]'* W[:, kact-kb+1:kact]
                Hn_cpu = @elapsed begin

                    pHn = zeros(Float64, kact, kb)
                    mul!(pHn, DV[kconv1:n,:], DW[kact-kb+1:kact, :]')
                    MPI.Allreduce!(pHn, +, comm_row)
                    MPI.Allreduce!(pHn, +, comm_col)
                    Hn[1:kact, kact-kb+1:kact] = pHn
                    #
                    # symmetrize the diagonal block of Hn, this is very important since
                    # otherwise Hn may not be numerically symmetric from last step 
                    # (the orthonormality of eigenvectors of Hn will be poor without this step)
                    #
                    Hn[kact-kb+1:kact,kact-kb+1:kact]=(Hn[kact-kb+1:kact,kact-kb+1:kact] + Hn[kact-kb+1:kact,kact-kb+1:kact]')/2.0
                    if  kact > kb  # symmetrize Hn
                        Hn[kact-kb+1:kact, 1:kact-kb] = Hn[1:kact-kb, kact-kb+1:kact]'
                    end
                end
                elapsedTime["Hn"] += Hn_cpu
                elapsedTime["Hn_n"] += 1

                # refinement_cpu = cputime
                refinement_cputotal += @elapsed begin
                    #
                    # Rayleigh-Ritz refinement (at each ietration)
                    # First compute the eigen-decomposition of the rayleigh-quotient matrix
                    # (sorting is unnecessary since eig(Hn) already sorts the Ritz values). 
                    # Then rotate the subspace basis.
                    #
                    d_eig, Eig_vec = eigen(Hn[1:kact,1:kact])  
                    Eig_val = Diagonal(d_eig)
                    
                    kold = kact
                    if  kact + blk > vimax
                        #
                        # inner-restart, it can be easily controlled by the two-indeces (kact,n) trick
                        #
                        if  displ > 4 
                            if rank == 0
                                @printf("==> Inner-restart: vimax=%i, n=%i, kact from %i downto %i \n", vimax, n, kact, ikeep)
                            end
                        end
                        kact = ikeep
                        n = kconv+kact   # should always keep n=kconv+kact
                        kinner = kinner+1  #used only for counting
                    end 
                    
                    # V[:,kconv1:kconv+kact] = V[:,kconv1:kconv+kold]*Eig_vec[1:kold,1:kact]
                    Eig_vec_p = Eig_vec[1:kold, 1:kact]
                    # @spawnat V.pids[i] localpart(V)[:,kconv1:kconv+kact] = localpart(V)[:,kconv1:kconv+kold]*Eig_vec[1:kold, 1:kact]
                    DV[kconv1:kconv+kact, :] = Eig_vec_p'*DV[kconv1:kconv+kold, :]
                    if  displ > 5  #can be expensive 
                        # @printf("Refinement: n=%i, kact=%i, kconv=%i,", n, kact, kconv)
                        # orth_err = norm(V[:,kconv1:kconv+kact]'*V[:,kconv1:kconv+kact] - Matrix{Float64}(I, kact, kact))
                        orth_err_local = zeros(Float64, kact, kact)
                        elapsedTime["Inner_prod"] += @elapsed begin
                            mul!(orth_err_local, DV[kconv1:kconv+kact, :], DV[kconv1:kconv+kact, :]')
                            MPI.Allreduce!(orth_err_local, +, comm_row)
                            MPI.Allreduce!(orth_err_local, +, comm_col)
                        end
                        elapsedTime["Inner_prod_n"] += 1
                        orth_err = norm(orth_err_local - Matrix{Float64}(I, kact, kact))
                        if  orth_err > 1e-10
                            if rank == 0
                                @printf("After RR refinement: orth-error = %e\n", orth_err)
                            end
                        end
                    end
                    # W[:,1:kact]=W[:,1:kold]*Eig_vec[1:kold,1:kact]
                    # @spawnat W.pids[i] localpart(W)[:,1:kact] = localpart(W)[:,1:kold]*Eig_vec[1:kold, 1:kact]
                    DW[1:kact,:] = Eig_vec_p'*DW[1:kold,:]
                end 
                
                beta1 = maximum(broadcast(abs, d_eig))
                #--deflation and restart may make beta1 too small, (the active subspace
                #--dim is small at the end), so use beta1 only when it is larger.     
                if  beta1 > maxritz
                    maxritz = beta1 
                end
                tolr = tol*maxritz     

                # test convergence     
                ec = 0    #ec conunts # of converged eigenpair at current step

                # conv_cpu0=cputime
                # CPUtic()
                # check convergence only for the smallest blk # of Ritz pairs. 
                # i.e., check the first blk Ritz values (since d_eig is in increasing order)
                kconv0 = kconv
                # it was 1:blk
                for jc = 1:kb  
                    
                    rho = d_eig[jc]
                    elapsedTime["Norm"] += @elapsed begin
                        # r = W[:, jc]  - rho*V[:,kconv0+jc]
                        # normr = norm(r)
                        normr = zeros(1)
                        normr[1] = norm(DW[jc,:]-rho*DV[kconv0+jc,:],2)^2
                        MPI.Allreduce!(normr, +, comm_row)
                        MPI.Allreduce!(normr, +, comm_col)
                        normr[1] = sqrt(normr[1])
                    end
                    elapsedTime["Norm_n"] += 1
                    if  displ >= 4
                        if rank == 0
                            @printf(" n = %i,  rho= %e,  rnorm= %e\n", n, rho, normr[1])
                        end
                    end
                    swap = false

                    if  longlog == 1
                        historyhere = zeros(1,4)
                        historyhere[1] = iter_total
                        historyhere[2] = MVprod
                        historyhere[3] = normr[1]
                        historyhere[4] = rho
                        history = vcat(history, historyhere)
                        nlog = nlog+1
                    end
                    
                    if  normr[1] < tolr
                        kconv = kconv +1
                        kconv1 = kconv +1
                        ec=ec+1
                        if  displ >= 1
                            if rank == 0
                                @printf("#%i converged in %i steps, ritz(%3i)=%e, rnorm= %6.4e\n", kconv, iter_total, kconv, rho, normr[1])
                            end
                        end
                        evals[kconv] = rho
                        resnrm[kconv] = normr[1]
                        
                        #
                        ##--compare with converged eigenvalues (sort in increasing order)
                        #
                        # determine to which position we should move up the converged eigenvalue
                        imove = kconv - 1
                        while  imove > 0
                            if rho < evals[imove] 
                                imove = imove -1 
                            else
                                break
                            end
                        end
                        imove = imove+1  #the position to move up the current converged pair
                        if  imove < kconv  
                            swap = true  
                            if  displ > 3
                                if rank == 0
                                    @printf(" ==> swap %3i  upto %3i\n", kconv, imove)
                                end
                            end
                            # vtmp =  V[:,kconv]
                            # for i = kconv:-1:imove+1
                            #     V[:,i]=V[:,i-1]
                            #     evals[i]=evals[i-1]
                            # end
                            # V[:,imove]=vtmp
                            
                            vtmp = DV[kconv,:]
                            DV[imove+1:kconv,:] = DV[imove:kconv-1,:]
                            DV[imove,:] = vtmp
                            evals[imove+1:kconv] = evals[imove:kconv-1]
                            evals[imove]=rho
                        end

                        if (kconv >= nwant && !swap && blk>1 ) || (kconv >= nwant+kmore)
                            if  displ > 1
                                if rank == 0
                                    @printf("The converged eigenvalues and residual_norms are:\n")
                                    for i = 1:kconv
                                        @printf("  eigval(%3i) = %11.8e,   resnrm(%3i) = %8.5e \n", i, evals[i], i, resnrm[i])
                                    end
                                end
                            end
                            # change to output V= V(:, 1:kconv); later 
                            # eigV = Matrix{Float64}(V[:, 1:kconv])
                            # eigV = DV[1:kconv, :]
                            
                            if  displ > 0 #these output info may be useful for profiling
                                if rank == 0
                                    @printf("#converged eig=%i,  #wanted eig=%i,  #iter=%i, kinner=%i, kouter=%i\n", kconv,  nwant,  iter_total, kinner, kouter)

                                    @printf(" info of the eigenproblem and solver parameters :  dim=%i\n", dim)
                                    @printf(" polym=%i,  blk=%i,  vomax=%i,  vimax=%i,  n=%i, augment=%i, tol=%4.2e\n",polym, blk, vomax, vimax, n, augment, tol)
                                    @printf(" ORTH-cpu=%6.4e, ORTH-flops=%i,  ORTH-flops/dim=%6.4e\n",orth_cputotal, orth_flopstotal,  orth_flopstotal/dim)
                                    @printf(" mat-vect-cpu=%6.4e  #mat-vect-prod=%i,  mat-vec-cpu/#mvprod=%6.4e\n",MVcpu, MVprod, MVcpu/MVprod)
                                    
                                    
                                    # cputotal = CPUtoq()
                                    # @printf(" filt_MVcpu=%6.4e, filt_non_mv_cpu=%6.4e, refinement_cpu=%6.4e\n",filt_mv_cput, filt_non_mv_cput, refinement_cputotal)
                                    # @printf(" CPU%%: MV=%4.2f%%(filt_MV=%4.2f%%), ORTH=%4.2f%%, refinement=%4.2f%%\n", MVcpu/cputotal*100, filt_mv_cput/cputotal*100, orth_cputotal/cputotal*100, refinement_cputotal/cputotal*100)
                                    # @printf("       other=%4.2f%% (filt_nonMV=%4.2f%%, Hn_cpu=%4.2f%%, conv_cpu=%4.2f%%)\n",(cputotal-MVcpu-orth_cputotal-refinement_cputotal)/cputotal*100,
                                            # filt_non_mv_cput/cputotal*100, Hn_cpu/cputotal*100, conv_cpu/cputotal*100)
                                    # @printf(" TOTAL CPU seconds = %e\n", cputotal)
                                end
                            end
                            
                            returnhere = 1
                            break
                            # return evals, eigV, kconv, history
                        end
                    else
                        break # exit when the first non-convergent Ritz pair is detected
                    end
                end

                if returnhere == 1
                    break
                end
                
                if  ec > 0
                    # W[:,1:kact-ec] = W[:,ec+1:kact] 
                    DW[1:kact-ec,:] = DW[ec+1:kact,:]
                    # update the current active subspace dimension 
                    kact = kact - ec   
                end
                
                # save only the non-converged Ritz values in Hn
                Hn[1:kact,1:kact] = Eig_val[ec+1:kact+ec,ec+1:kact+ec]     
                
                #
                # update lower_nwb  (ritz_nwb) to be the mean value of d_eig. 
                # from many practices this choice turn out to be efficient,
                # but there are other choices for updating the lower_nwb.
                # (the convenience in adapting this bound without extra computation
                # shows the reamrkable advantage in integrating the Chebbyshev 
                # filtering in a Davidson-type method)
                #
                #low_nwb = median(d_eig(1:max(1, length(d_eig)-1)));
                low_nwb = median(d_eig)
                lowb = minimum(d_eig)
                
                #
                # determine if need to do outer restart (only n need to be updated)
                #
                if  n + blk > vomax && opts["do_outer"]
                    nold = n
                    n = max(kconv+blk, vomax - 2*blk)
                    if  displ > 4
                        if rank == 0
                            @printf("--> Outer-restart: n from %i downto %i, vomax=%i, vimax=%i, kact=%i\n", nold, n, vomax, vimax, kact)
                        end
                    end
                    kact = n-kconv
                    kouter = kouter+1  #used only for counting
                end
                # conv_cpu = conv_cpu + CPUtoq()

                # @printf("%i, %i\n", iter_total, itmax)
            
            end
        end

        if  iter_total > itmax
            #
            # the following should rarely happen unless the problem is
            # extremely difficult (highly clustered eigenvalues)
            # or the vomax or vimax is set too small
            #
            if rank == 0
                @printf("***** itmax=%i, it_total=%i\n", itmax, iter_total)
                @printf("***** bchdav.jl:  Maximum iteration exceeded\n")
                @printf("***** nwant=%i, kconv=%i, vomax=%i\n", nwant, kconv, vomax)
                @printf("***** it could be that your vimax/blk, or vomax, or polym is too small")
            end
        end

    

    end

    elapsedTime["Pre_loop"] = elapsedTime["total"] - elapsedTime["main_loop"]

    if verb
        if rank == 0
            @printf("\n")
            @printf("+++++++++++++++++++++ runtime details for debugging +++++++++++++++++++++++++\n")
            @printf("total runtime:                          %.2e \n", elapsedTime["total"])
            @printf("runtime preloop                         %.2e \n", elapsedTime["Pre_loop"])
            @printf("runtime main_loop:                      %.2e \n", elapsedTime["main_loop"])
            if verb
                @printf("   runtime TSQR:                    %.2e / %i \n", elapsedTime["TSQR"], elapsedTime["TSQR_n"])
                @printf("   runtime Cheb_filter_scal:        %.2e / %i \n", elapsedTime["Cheb_filter_scal"], elapsedTime["Cheb_filter_scal_n"])
                @printf("   runtime SpMM:                    %.2e / %i \n", elapsedTime["SpMM"], elapsedTime["SpMM_n"])
                @printf("   runtime Inner_prod:              %.2e / %i \n", elapsedTime["Inner_prod"], elapsedTime["Inner_prod_n"])
                @printf("   runtime Norm:                    %.2e / %i \n", elapsedTime["Norm"], elapsedTime["Norm_n"])   
                @printf("   runtime Hn:                      %.2e / %i \n", elapsedTime["Hn"], elapsedTime["Hn_n"])            
            end
            @printf("+++++++++++++++++++++++++++++++ END +++++++++++++++++++++++++++++++++++++++++\n")
        end
    end

    
    return evals[1:kconv], DV[1:kconv, :], kconv, iter_total, elapsedTime

end

function DGKS_1(V, N, comm_row, comm_col)
    colv, ndim = size(V)
    nrmv = zeros(Float64, 1)
    nrmv[1] = norm(V[1,:],2)^2
    MPI.Allreduce!(nrmv, +, comm_row)
    MPI.Allreduce!(nrmv, +, comm_col)
    nrmv[1] = sqrt(nrmv[1])
    V[1, 1:ndim] = V[1, 1:ndim]/nrmv[1]
    if colv == 1
        W = zeros(Float64, 0, ndim)
    else
        W = DGKS(V[1:1, :], 1:1, V[2:colv, :], N, comm_row, comm_col)
    end
    vcat(V[1:1, 1:ndim], W)
end

function DGKS(X, ids, V, N, comm_row, comm_col)
    epsbig = 2.22045e-16  
    reorth=0.717
    one=1.0e0
    colv, ndim = size(V)
    colx = length(ids)
    orth_flops = 0
    vrank = 0

    for k = 1:colv
        tV = zeros(Float64, size(V[k:k, :]))
        nrmv = zeros(Float64, 1)
        nrmv[1] = norm(V[k,:],2)^2
        orth_flops += ndim
        MPI.Allreduce!(nrmv, +, comm_row)
        MPI.Allreduce!(nrmv, +, comm_col)
        nrmv[1] = sqrt(nrmv[1])
        if (nrmv[1] < epsbig*sqrt(N))
            continue 
        end

        if (nrmv[1] <= 2*epsbig || nrmv[1] >= 300)
            V[k, 1:ndim] = V[k, 1:ndim]/nrmv[1]
            orth_flops += ndim
            nrmv[1] = one
        end

        h = zeros(Float64, colx+vrank, 1)
        if colx == 0
            mul!(h, V[1:vrank, :], V[k:k,:]')
        else
            mul!(h, vcat(X[ids, :], V[1:vrank, :]), V[k:k, :]')
        end
        MPI.Allreduce!(h, +, comm_row)
        MPI.Allreduce!(h, +, comm_col)
        if colx == 0
            mul!(tV, h', V[1:vrank, :])
        else
            mul!(tV, h', vcat(X[ids, :], V[1:vrank, :]))
        end
        if colx+vrank > 0
            V[k:k, :] -= tV
        end
        # V(1:ndim,k) = V(1:ndim,k) -  [X, V(1:ndim,1:vrank)]*h;
        orth_flops += ndim*((colx+vrank)*2 + 1)

        nrmproj = zeros(Float64, 1)
        nrmproj[1] = norm(V[k,:],2)^2
        orth_flops=orth_flops+ndim
        MPI.Allreduce!(nrmproj, +, comm_row)
        MPI.Allreduce!(nrmproj, +, comm_col)
        nrmproj[1] = sqrt(nrmproj[1])

        if (nrmproj[1] > reorth*nrmv[1])
            vrank = vrank +1
            if (abs(nrmproj[1] - one) > epsbig)
                V[k,:] = V[k,:]/nrmproj[1]
                orth_flops += ndim
            end
            if (vrank != k)
                V[vrank, :] = V[k, :]
            end
        else
            nrmv[1] = nrmproj[1];      

            h = zeros(Float64, colx+vrank, 1)
            if colx == 0
                mul!(h, V[1:vrank, :], V[k:k, :]')
            else
                mul!(h, vcat(X[ids, :], V[1:vrank, :]), V[k:k, :]')
            end
            MPI.Allreduce!(h, +, comm_row)
            MPI.Allreduce!(h, +, comm_col)
            if colx == 0
                mul!(tV, h', V[1:vrank, :])
            else
                mul!(tV, h', vcat(X[ids, :], V[1:vrank, :]))
            end
            if colx+vrank > 0
                V[k:k, :] -= tV
            end
            orth_flops += ndim*((colx+vrank)*2 + 1)
            
            nrmproj[1] = norm(V[k, :],2)^2
            orth_flops=orth_flops+ndim
            MPI.Allreduce!(nrmproj, +, comm_row)
            MPI.Allreduce!(nrmproj, +, comm_col)
            nrmproj[1] = sqrt(nrmproj[1])    
            if (nrmproj[1] > reorth*nrmv[1]  && nrmproj[1] >= sqrt(N)*epsbig)
                vrank = vrank +1
                if (abs(nrmproj[1] - one) > epsbig)
                    V[k, :] = V[k, :]/nrmproj[1]
                    orth_flops += ndim
                end
                if (vrank != k)
                    V[vrank, :] = V[k, :]
                end	
            else
                
                # fail the 2nd reorthogonalization test,
                #    V(:,k) is numerically dependent in V(:, 1:vrank),
                # do not increase vrank, but go for the next k 
                
            end
        end
    end

    if vrank > 0
        V = V[1:vrank, :]
    else #recursively call DGKS_blk to make output V not a zero vector
        # fprintf('DGKS: # of columns replaced by random vectors =%i\n', colv-vrank); 
        V[vrank+1:colv, :] = DGKS(vcat(X[ids, :], V[1:vrank, :]), 1:(length(ids)+vrank), randn(colv-vrank, ndim), N, comm_row, comm_col)
    end

    return V
end

function SpMM_A_0(V, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    blk = size(V, 1)
    # W = zeros(Float64, blk, info_cols_dist[rank_col*comm_size_sq+rank_row+1])
    W_gather = zeros(Float64, size(A, 1), blk)
    V_gather_T = zeros(Float64, size(A, 2), blk)
    local_info_cols = info_cols_dist[rank_row*comm_size_sq+1:rank_row*comm_size_sq+comm_size_sq]
    V_gather = Array{Float64}(undef, (blk, sum(local_info_cols)))
    _counts = vcat([blk for i in 1:length(local_info_cols[:])]', local_info_cols')
    V_gather_vbuf = VBuffer(V_gather, vec(prod(_counts, dims=1)))
    # W_gather_T = zeros(Float64, blk, size(A,1))
    # _counts = vcat([blk for i in 1:length(local_info_cols[:])]', local_info_cols')
    # W_gather_T_vbuf = VBuffer(W_gather_T, vec(prod(_counts, dims=1)))

    MPI.Allgatherv!(V, V_gather_vbuf, comm_col)

    mul!(W_gather, A, V_gather', 1.0, 0.0)

    MPI.Allreduce!(W_gather, +, comm_row)
    # W_gather_T = W_gather'

    # W = MPI.Scatterv!(W_gather_T_vbuf, zeros(Float64, size(W)), root, comm_row)
    
    # MPI.Allgatherv!(W, W_gather_T_vbuf, comm_row)
    
    mul!(V_gather_T, E', W_gather, 1.0, 0.0)

    MPI.Allreduce!(V_gather_T, +, comm_col)
    V_gather = V_gather_T'
    
    Y = V_gather[:, sum(local_info_cols[1:rank_col])+1:sum(local_info_cols[1:rank_col+1])]
    # Y = zeros(Float64, size(V))
    # MPI.Scatterv!(V_gather_vbuf, Y, root, comm_col)  
end

function SpMM_A(X, A, X_gather, X_gather_vbuf, Y_gather, Y_gather_T, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    blk = size(X, 1)
    MPI.Allgatherv!(X, X_gather_vbuf, comm_col)

    mul!(Y_gather, A, X_gather', 1.0, 0.0)

    MPI.Reduce!(Y_gather, +, root, comm_row)
    Y_gather_T = zeros(Float64, size(Y_gather, 2), size(Y_gather, 1))
    Y_gather_T = copy(Y_gather')
    
    local_info_rows = info_cols_dist[rank_col*comm_size_sq+1:rank_col*comm_size_sq+comm_size_sq]
    _counts = vcat([blk for i in 1:length(local_info_rows[:])]', local_info_rows')
    _counts = vec(prod(_counts, dims=1))
    Y_gather_T_vbuf = VBuffer(Y_gather_T, _counts)
    Y = MPI.Scatterv!(Y_gather_T_vbuf, zeros(Float64, blk, local_info_rows[rank_row+1]), root, comm_row)

    # MPI.Allreduce!(Y_gather, +, comm_row)
    # Y_gather_T = Y_gather'
    # local_info_cols = info_cols_dist[rank_col*comm_size_sq+1:rank_col*comm_size_sq+comm_size_sq]
    # Y = Y_gather_T[:, sum(local_info_cols[1:rank_row])+1:sum(local_info_cols[1:rank_row+1])]
end


function SpMM_A_1D(X, A, info_cols_dist, rank, comm_size, root, comm)
    cputime = Dict()

    cputime["pre"] = @elapsed begin
    blk, dim = size(X)
    N = size(A,2)
    # W = zeros(Float64, blk, info_cols_dist[rank_col*comm_size_sq+rank_row+1])
    YT = zeros(Float64, dim, blk)
    Y = zeros(Float64, blk, dim)
    X_gather = Array{Float64}(undef, (blk, N))
    _counts = vcat([blk for i in 1:length(info_cols_dist[:])]', info_cols_dist')
    X_gather_vbuf = VBuffer(X_gather, vec(prod(_counts, dims=1)))
    end

    cputime["Allgatherv!"] = @elapsed begin
    MPI.Allgatherv!(X, X_gather_vbuf, comm)
    end
    
    cputime["mul!"] = @elapsed begin
    mul!(YT, A, X_gather')
    end
    transpose!(Y, YT)
    Y, cputime
end


function SpMM_A_1(X, A, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)

    cputime = Dict()

    cputime["pre"] = @elapsed begin
    blk = size(X, 1)
    # W = zeros(Float64, blk, info_cols_dist[rank_col*comm_size_sq+rank_row+1])
    Y_gather = zeros(Float64, size(A, 1), blk)
    local_info_cols = info_cols_dist[rank_row*comm_size_sq+1:rank_row*comm_size_sq+comm_size_sq]
    X_gather = Array{Float64}(undef, (blk, sum(local_info_cols)))
    _counts = vcat([blk for i in 1:length(local_info_cols[:])]', local_info_cols')
    X_gather_vbuf = VBuffer(X_gather, vec(prod(_counts, dims=1)))
    end
    
    cputime["Allgatherv!"] = @elapsed begin
    MPI.Allgatherv!(X, X_gather_vbuf, comm_col)
    end

    cputime["mul!"] = @elapsed begin
    mul!(Y_gather, A, X_gather')
    end

    # cputime["Reduce!"] = @elapsed begin
    # MPI.Reduce!(Y_gather, +, root, comm_row)
    # end

    # cputime["transpose!"] = @elapsed begin
    # local_info_rows = info_cols_dist[rank_col*comm_size_sq+1:rank_col*comm_size_sq+comm_size_sq]    
    # Y_gather_T = zeros(Float64, size(Y_gather, 2), size(Y_gather, 1))
    # if rank_row == 0
    #     transpose!(Y_gather_T, Y_gather)
        
    #     _counts = vcat([blk for i in 1:length(local_info_rows[:])]', local_info_rows')
    #     _counts = vec(prod(_counts, dims=1))
    #     Y_gather_T_vbuf = VBuffer(Y_gather_T, _counts)
    # else
    #     Y_gather_T_vbuf = VBuffer(nothing)
    # end
    # end

    # cputime["Scatterv!"] = @elapsed begin
    # Y = MPI.Scatterv!(Y_gather_T_vbuf, zeros(Float64, blk, local_info_rows[rank_row+1]), root, comm_row)
    # end
    
    cputime["Reduce_scatter"] = @elapsed begin
    local_info_rows = info_cols_dist[rank_col*comm_size_sq+1:rank_col*comm_size_sq+comm_size_sq]
    Y_gather_T = zeros(Float64, size(Y_gather, 2), size(Y_gather, 1))
    transpose!(Y_gather_T, Y_gather)
    _counts = vcat([blk for i in 1:length(local_info_rows[:])]', local_info_rows')
    _counts = vec(prod(_counts, dims=1))
    Y_gather_T_vbuf = VBuffer(Y_gather_T, _counts)
    Y = zeros(Float64, blk, local_info_rows[rank_row+1])
    Y_buf = MPI.Buffer(Y)
    op = MPI.Op(+, Float64, iscommutative=true)
    MPI.API.MPI_Reduce_scatter(Y_gather_T, Y, Y_gather_T_vbuf.counts, Y_buf.datatype, op, comm_row)
    end
    Y, cputime
end

function SpMM_A_2(X, A, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    blk = size(X, 1)
    # W = zeros(Float64, blk, info_cols_dist[rank_col*comm_size_sq+rank_row+1])
    Y_gather = zeros(Float64, blk, size(A, 1))
    local_info_cols = info_cols_dist[rank_row*comm_size_sq+1:rank_row*comm_size_sq+comm_size_sq]
    X_gather = Array{Float64}(undef, (blk, sum(local_info_cols)))
    _counts = vcat([blk for i in 1:length(local_info_cols[:])]', local_info_cols')
    X_gather_vbuf = VBuffer(X_gather, vec(prod(_counts, dims=1)))

    MPI.Allgatherv!(X, X_gather_vbuf, comm_col)

    mul!(Y_gather', A, X_gather', 1.0, 0.0)

    MPI.Reduce!(Y_gather, +, root, comm_row)
    
    local_info_rows = info_cols_dist[rank_col*comm_size_sq+1:rank_col*comm_size_sq+comm_size_sq]
    if rank_row == 0
        _counts = vcat([blk for i in 1:length(local_info_rows[:])]', local_info_rows')
        _counts = vec(prod(_counts, dims=1))
        Y_gather_vbuf = VBuffer(Y_gather, _counts)
    else
        Y_gather_vbuf = VBuffer(nothing)
    end
    Y = MPI.Scatterv!(Y_gather_vbuf, zeros(Float64, blk, local_info_rows[rank_row+1]), root, comm_row)
end

function SpMM_A_3(X, A, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, X_gather, X_gather_vbuf, Y_gather, Y_gather_T, Y_gather_T_vbuf)
    cputime = Dict()
    cputime["Allgatherv!"] = @elapsed begin
    MPI.Allgatherv!(X, X_gather_vbuf, comm_col)
    end
    
    # if rank_row == 0 && rank_col == 0
    #     @printf("size(A): (%i, %i), nnz(A): %i, size(X): (%i, %i) \n", size(A, 1), size(A, 2), nnz(A), size(X_gather,1), size(X_gather,2))
    # end
    cputime["mul!"] = @elapsed begin
    mul!(Y_gather, A, X_gather')
    # Y_gather = A*X_gather'
    end
    
    cputime["Reduce!"] = @elapsed begin
    MPI.Reduce!(Y_gather, +, root, comm_row)
    end

    cputime["transpose!"] = @elapsed begin
    if rank_row == 0
        transpose!(Y_gather_T, Y_gather)
    end
    end

    cputime["Scatterv!"] = @elapsed begin
    local_info_rows = info_cols_dist[rank_col*comm_size_sq+1:rank_col*comm_size_sq+comm_size_sq]
    Y = MPI.Scatterv!(Y_gather_T_vbuf, zeros(Float64, blk, local_info_rows[rank_row+1]), root, comm_row)
    end

    Y, cputime
end

function SpMM_A_4(A, root, comm, X_gather, Y_gather, Y_gather_T)
    
    cputime = Dict()
    cputime["mul!"] = @elapsed begin
    mul!(Y_gather, A, X_gather')
    end

    cputime["Allreduce!"] = @elapsed begin
    MPI.Allreduce!(Y_gather, +, comm)
    end

    cputime["transpose!"] = @elapsed begin
    transpose!(Y_gather_T, Y_gather)
    end

    cputime
end

function SpMM_A_5(X, A, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, X_gather, X_gather_vbuf, Y_gather, Y_gather_T, Y_gather_T_vbuf)
    cputime = Dict()
    cputime["Allgatherv!"] = @elapsed begin
    # MPI.Allgatherv!(X, X_gather_vbuf, comm_col)
    end
    
    if rank_row == 0 && rank_col == 0
        @printf("size(A): (%i, %i), nnz(A): %i, size(X): (%i, %i), size(Y): (%i, %i) \n", size(A, 1), size(A, 2), nnz(A), size(X_gather,1), size(X_gather,2), size(Y_gather,1), size(Y_gather,2))
    end
    cputime["mul!"] = @elapsed begin
        mul!(Y_gather, A, X_gather')
    # Y_gather = A*X_gather'
    end
    
    cputime["Reduce!"] = @elapsed begin
    # MPI.Reduce!(Y_gather, +, root, comm_row)
    end

    cputime["transpose!"] = @elapsed begin
    if rank_row == 0
        # transpose!(Y_gather_T, Y_gather)
    end
    end

    cputime["Scatterv!"] = @elapsed begin
        local_info_rows = info_cols_dist[rank_col*comm_size_sq+1:rank_col*comm_size_sq+comm_size_sq]
        Y = zeros(blk, local_info_rows[rank_row+1])
    # Y = MPI.Scatterv!(Y_gather_T_vbuf, zeros(Float64, blk, local_info_rows[rank_row+1]), root, comm_row)
    end

    Y, cputime
end


function SpMM_A_4_w_E(A, E, root, comm_row, comm_col, X_gather, X_gather_T, Y_gather, Y_gather_T)
    SpMM_A_4(A, root, comm_row, X_gather, Y_gather, Y_gather_T)
    SpMM_A_4(E', root, comm_col, Y_gather_T, X_gather_T, X_gather)
end

function SpMM_A_1_w_E(X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    Z, _ = SpMM_A_1(X, A, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    Y, _ = SpMM_A_1(Z, E', info_cols_dist, rank_col, rank_row, comm_size_sq, root, comm_col, comm_row)
    Y
end

# function Cheb_filter_scal(deg, low, high, leftb, X, A, E, X_gather, X_gather_T, X_gather_vbuf, Y_gather, Y_gather_T, Y_gather_T_vbuf, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
function Cheb_filter_scal(deg, low, high, leftb, X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
# deg should always be an odd number

    e = (high - low)/2
    center= (high+low)/2
    sigma = e/(leftb - center)
    tau = 2/sigma

    Y = SpMM_A_0(X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    Y = (Y - center*X)*(sigma/e)
    for kk = 2:deg-1
        sigma_new = 1 /(tau - sigma);
        Y1 = SpMM_A_0(Y, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        Y1 = (Y1 - center*Y)*(2*sigma_new/e) - (sigma*sigma_new)*X
        X = copy(Y)
        Y = copy(Y1)
        sigma = sigma_new
    end
    
    Y1 = SpMM_A_0(Y, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    sigma_new = 1 /(tau - sigma)
    Y1 = (Y1 - center*Y)*(2*sigma_new/e) - (sigma*sigma_new)*X
    
end


function Cheb_filter_scal_1D(deg, low, high, leftb, X, A, info_cols_dist, rank, comm_size, root, comm)
    e = (high - low)/2
    center= (high+low)/2
    sigma = e/(leftb - center)
    tau = 2/sigma

    cputime = Dict("copy"=>0.0)
    cputime["SpMM"] = @elapsed begin
    Y, _ = SpMM_A_1D(X, A, info_cols_dist, rank, comm_size, root, comm)
    end
    cputime["local_computation"] = @elapsed begin
    Y = (Y - center*X)*(sigma/e)
    end
    for kk = 2:deg-1
        sigma_new = 1 /(tau - sigma)

        cputime["SpMM"] += @elapsed begin
        Y1, _ = SpMM_A_1D(Y, A, info_cols_dist, rank, comm_size, root, comm)
        end
        
        cputime["local_computation"] += @elapsed begin
        Y1 = (Y1 - center*Y)*(2*sigma_new/e) - (sigma*sigma_new)*X
        end

        cputime["copy"] += @elapsed begin
        X = deepcopy(Y)
        Y = deepcopy(Y1)
        end
        sigma = sigma_new
    end

    cputime["SpMM"] += @elapsed begin
    Y1, _ = SpMM_A_1D(Y, A, info_cols_dist, rank, comm_size, root, comm)
    end
    sigma_new = 1 /(tau - sigma)
    cputime["local_computation"] += @elapsed begin
    Y1 = (Y1 - center*Y)*(2*sigma_new/e) - (sigma*sigma_new)*X
    end
    
    Y1, cputime
end

function Cheb_filter_scal_1(deg, low, high, leftb, X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
# deg should always be an odd number

    e = (high - low)/2
    center= (high+low)/2
    sigma = e/(leftb - center)
    tau = 2/sigma
    
    cputime = Dict("copy"=>0.0)
    cputime["SpMM"] = @elapsed begin
    Y = SpMM_A_1_w_E(X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    end
    cputime["local_computation"] = @elapsed begin
    Y = (Y - center*X)*(sigma/e)
    end
    for kk = 2:deg-1
        sigma_new = 1 /(tau - sigma)

        cputime["SpMM"] += @elapsed begin
        Y1 = SpMM_A_1_w_E(Y, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
        
        cputime["local_computation"] += @elapsed begin
        Y1 = (Y1 - center*Y)*(2*sigma_new/e) - (sigma*sigma_new)*X
        end

        cputime["copy"] += @elapsed begin
        X = deepcopy(Y)
        Y = deepcopy(Y1)
        end
        sigma = sigma_new
    end
    
    cputime["SpMM"] += @elapsed begin
    Y1 = SpMM_A_1_w_E(Y, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    end
    sigma_new = 1 /(tau - sigma)
    cputime["local_computation"] += @elapsed begin
    Y1 = (Y1 - center*Y)*(2*sigma_new/e) - (sigma*sigma_new)*X
    end
    
    Y1, cputime
end

function Cheb_filter_scal_4(deg, low, high, leftb, X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, X_gather, X_gather_T, Y_gather, Y_gather_T, X_gather_vbuf)
# deg should always be an odd number

    e = (high - low)/2
    center= (high+low)/2
    sigma = e/(leftb - center)
    tau = 2/sigma

    cputime = Dict("copy"=>0.0)
    cputime["Allgatherv!"] = @elapsed begin
    MPI.Allgatherv!(X, X_gather_vbuf, comm_col)
    end

    cputime["copy"] += @elapsed begin
    X_gather0 = copy(X_gather)
    end
    cputime["SpMM"] = @elapsed begin
    SpMM_A_4_w_E(A, E, root, comm_row, comm_col, X_gather, X_gather_T, Y_gather, Y_gather_T)
    end
    cputime["local_computation"] = @elapsed begin
    X_gather = (X_gather - center*X_gather0)*(sigma/e)
    end
    for kk = 2:deg-1
        sigma_new = 1 / (tau - sigma)
        cputime["copy"] += @elapsed begin
        X_gather1 = copy(X_gather)
        end
        cputime["SpMM"] += @elapsed begin
        SpMM_A_4_w_E(A, E, root, comm_row, comm_col, X_gather, X_gather_T, Y_gather, Y_gather_T)
        end
        cputime["local_computation"] += @elapsed begin
        X_gather = (X_gather - center*X_gather1)*(2*sigma_new/e) - (sigma*sigma_new)*X_gather0
        end
        cputime["copy"] += @elapsed begin
        X_gather0 = copy(X_gather1)
        end
        sigma = sigma_new
    end
    
    cputime["copy"] += @elapsed begin
    X_gather1 = copy(X_gather)
    end
    cputime["SpMM"] += @elapsed begin
    SpMM_A_4_w_E(A, E, root, comm_row, comm_col, X_gather, X_gather_T, Y_gather, Y_gather_T)
    end
    sigma_new = 1 /(tau - sigma)
    cputime["local_computation"] += @elapsed begin
    X_gather = (X_gather - center*X_gather1)*(2*sigma_new/e) - (sigma*sigma_new)*X_gather0
    end
    
    cputime["Scatterv!"] = @elapsed begin
    local_info_rows = info_cols_dist[rank_row*comm_size_sq+1:rank_row*comm_size_sq+comm_size_sq]
    X = MPI.Scatterv!(X_gather_vbuf, zeros(Float64, blk, local_info_rows[rank_col+1]), root, comm_col)
    end
    
    X, cputime
end

function Cheb_filter_scal_sq(deg, low, high, leftb, X, A)
# deg should always be an odd number

    e = (high - low)/2
    center= (high+low)/2
    sigma = e/(leftb - center)
    tau = 2/sigma
    Y = zeros(Float64, size(X))
    Y1 = zeros(Float64, size(X))


    mul!(Y, A, X)
    Y = (Y - center*X)*(sigma/e)
    for kk = 2:deg-1
        sigma_new = 1 /(tau - sigma)
        mul!(Y1, A, Y)
        Y1 = (Y1 - center*Y)*(2*sigma_new/e) - (sigma*sigma_new)*X
        X = deepcopy(Y)
        Y = deepcopy(Y1)
        sigma = sigma_new
    end
    
    mul!(Y1, A, Y)
    sigma_new = 1 /(tau - sigma)
    Y1 = (Y1 - center*Y)*(2*sigma_new/e) - (sigma*sigma_new)*X

    Y1
end

function split_count(N::Integer, n::Integer)
    q,r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

function split_count_local(N::Integer, n::Integer)
    counts = zeros(Int64, n*n)
    counts1 = split_count(N, n)
    info = zeros(Int64, 1, n*n*n)
    for i in 1:n
        counts2 = split_count(counts1[i], n)
        counts[(i-1)*n+1:i*n] = counts2
        for j in 1:n
            info[(i-1)*n*n+(j-1)*n+1:(i-1)*n*n+j*n] = counts2
        end
    end
    return counts, info
end

function findnz_local(A, comm_size_sq, nnz)
    N = size(A,1)
    counts = split_count(N, comm_size_sq)
    I = zeros(Int64, nnz)
    J = zeros(Int64, nnz)
    V = zeros(Float64, nnz)
    S = zeros(Int64, 2, comm_size_sq*comm_size_sq)
    col_start = col_end = 0
    row_start = row_end = 0
    counts_local = zeros(Int64, comm_size_sq, comm_size_sq)
    n = 0
    for j in 1:comm_size_sq
        col_start = j == 1 ? 1 : sum(counts[1:j-1])+1
        col_end = sum(counts[1:j])
        for i in 1:comm_size_sq
            row_start = i == 1 ? 1 : sum(counts[1:i-1])+1
            row_end = sum(counts[1:i])
            I1, J1, V1 = findnz(A[row_start:row_end, col_start:col_end])
            m = length(I1)
            I[n+1:n+m] = I1
            J[n+1:n+m] = J1
            V[n+1:n+m] = V1
            counts_local[i, j] = m
            S[1, (j-1)*comm_size_sq + i] = row_end - row_start + 1
            S[2, (j-1)*comm_size_sq + i] = col_end - col_start + 1
            n += m
        end
    end
    I, J, V, counts_local[:], S
end


function ranks_division(ranks, q)
    n = length(ranks)
    ans = []
    if n >= q
        dvd = split_count(n, q)
        cs = [0]
        s = 0
        for i in 1:length(dvd)
            s += dvd[i]
            append!(cs, s)
        end
        ans = [ranks[cs[1]+1:cs[2]]]
        for i in 2:q
            append!(ans, [ranks[cs[i]+1:cs[i+1]]])
        end
    elseif n > 0
        ans = [[ranks[1]]]
        for i in 2:n
            append!(ans, [ranks[i:i]])
        end
    end
    ans
end

function recursive_tree(ranks, l, L, q)
    node = Dict("element"=> ranks, "children"=> [])
    if l < L
        subranks = ranks_division(ranks, q)
        for i in 1:length(subranks)
            ranks_i = subranks[i]
            child = recursive_tree(ranks_i, l+1, L, q)
            append!(node["children"], [child])
        end
    end
    node
end

function level_trasverse(ranks, L, q)
    l = 0
    ans = Dict(l=> [ranks])
    l2r = Dict(l=> [[ranks]])
    while l < L
        l += 1
        ans[l] = []
        l2r[l] = []
        for i in 1:length(ans[l-1])
            ranks_i = ans[l-1][i]
            subranks_i = ranks_division(ranks_i, q)
            append!(ans[l], subranks_i)
            append!(l2r[l], [subranks_i])     
        end
    end
    l2r
end

function rank2communicator_tree(l2r)
    lvls = keys(l2r)
    r2c = Dict()
    counts = Dict()
    s = 0
    for lvl in lvls
        parents = l2r[lvl]
        r2c[lvl] = Dict("bcast1"=>Dict(), "bcast2"=>Dict())
        counts[lvl] = Dict("bcast1"=>Dict(), "bcast2"=>Dict(), "backward"=>Dict())
        for i in 1:length(parents)
            nodes = parents[i]
            num_rk_pernode = [length(node) for node in nodes]
            maxrk = maximum(num_rk_pernode)
            minrk = minimum(num_rk_pernode)
            @assert minrk <= maxrk <= minrk + 1
            sumrk = sum(num_rk_pernode)
            num_node_mrk = sumrk - minrk*length(nodes)
            for j in 1:minrk
                for k in 1:length(nodes)
                    r2c[lvl]["bcast1"][nodes[k][j]] = nodes[1][j]
                    r2c[lvl]["bcast2"][nodes[k][j]] = nodes[k][j]
                    counts[lvl]["bcast1"][nodes[k][j]] = length(nodes)
                    counts[lvl]["bcast2"][nodes[k][j]] = length(nodes)
                    counts[lvl]["backward"][nodes[k][j]] = k-1
                end
            end
            if num_node_mrk > 0
                for k in 1:num_node_mrk
                    r2c[lvl]["bcast1"][nodes[k][maxrk]] = nodes[k][maxrk]
                    r2c[lvl]["bcast2"][nodes[k][maxrk]] = nodes[1][1]
                    counts[lvl]["bcast1"][nodes[k][maxrk]] = 1
                    counts[lvl]["bcast2"][nodes[k][maxrk]] = length(nodes)
                    counts[lvl]["backward"][nodes[k][maxrk]] = k-1
                end
            end
        end
    end
    r2c, counts     
end

function create_communicators(r2c, rank, comm)
    comms = Dict()
    for lvl in keys(r2c)
        comms[lvl] = Dict()
        comms[lvl]["bcast1"] = MPI.Comm_split(comm, r2c[lvl]["bcast1"][rank], rank)
        MPI.Barrier(comm)
        comms[lvl]["bcast2"] = MPI.Comm_split(comm, r2c[lvl]["bcast2"][rank], rank)
        MPI.Barrier(comm)
    end
    comms
end

function setup_process_tree(P, q=4)
# number of levels L = log_q P
    L = Int32(ceil(log(q, P)))
    ranks = [i for i in 0:P-1]
    l2r = level_trasverse(ranks, L, q)
    tree = recursive_tree(ranks, 0, L, q)
    r2c, counts = rank2communicator_tree(l2r)
    r2c, counts, l2r, tree, L
end

function TSQR_1(V, N, L, comms, counts_info, rank, comm_row, comm_col)
    colv, ndim = size(V)
    nrmv = zeros(Float64, 1)
    nrmv[1] = norm(V[1,:],2)^2
    MPI.Allreduce!(nrmv, +, comm_row)
    MPI.Allreduce!(nrmv, +, comm_col)
    nrmv[1] = sqrt(nrmv[1])
    V[1, 1:ndim] = V[1, 1:ndim]/nrmv[1]
    if colv == 1
        W = zeros(Float64, 0, ndim)
    else
        W, _ = TSQR(V[1:1, :], 1:1, V[2:colv, :], N, L, comms, counts_info, rank, comm_row, comm_col)
    end
    vcat(V[1:1, 1:ndim], W)
end

function TSQR(X, ids, V, N, L, comms, counts_info, rank, comm_row, comm_col)
    cputime = Dict("Allgatherv!"=>0.0, "Bcast!"=>0.0, "Allreduce!"=>0.0)
    W = vcat(X[ids,:], V)
    kb0 = length(ids)
    kb = size(W,1)
    l = L
    F = qr(W')
    Qs = Dict(l=>Matrix(F.Q))
    RT = zeros(size(F.R,2), size(F.R,1))
    transpose!(RT, F.R)
    while l > 0
        RT_gather = zeros(kb, counts_info[l]["bcast1"][rank]*kb)
        RT_gather_vbuf = VBuffer(RT_gather, vec([kb*kb for i in 1:counts_info[l]["bcast1"][rank]]))
        cputime["Allgatherv!"] += @elapsed begin
        MPI.Allgatherv!(RT, RT_gather_vbuf, comms[l]["bcast1"])
        end
        RT_gathered = zeros(kb, counts_info[l]["bcast2"][rank]*kb)
        RT_gathered[:, 1:size(RT_gather, 2)] = RT_gather
        cputime["Bcast!"] += @elapsed begin
        MPI.Bcast!(RT_gathered, comms[l]["bcast2"])
        end
        F = qr(RT_gathered')
        Qs[l-1] = Matrix(F.Q)
        RT = zeros(size(F.R,2), size(F.R,1))
        transpose!(RT, F.R)
        l -= 1
    end
    l = 0
    while l < L
        Qs[l+1] = Qs[l+1]*Qs[l][counts_info[l+1]["backward"][rank]*kb+1:counts_info[l+1]["backward"][rank]*kb+kb, :]
        l += 1
    end
    norm_partial = vec(sum(RT[kb0+1:kb,kb0+1:kb] .^ 2, dims=2))
    cputime["Allreduce!"] += @elapsed begin
    MPI.Allreduce!(norm_partial, +, comm_row)
    MPI.Allreduce!(norm_partial, +, comm_col)
    end
    norm_whole = broadcast(sqrt, norm_partial)
    conv = (norm_whole .>= 2.22045e-16)
    if sum(conv) > 0
        indexes = findall((x)->x==1, conv)
        V = Matrix(Qs[L][:, kb0 .+ indexes]')
    else
        V, cputime1 = TSQR(X, ids, randn(kb-kb0,size(X,2)), N, L, comms, counts_info, rank, comm_row, comm_col)
        for (key,val) in cputime1
            cputime[key] += cputime1[key]
        end
    end
    V, cputime
end
# end # module


