using Test, MPIClusterManagers
using MPI

ks = [16, 32, 64]
SQNPROCS = [1]
Ns = [5000, 20000, 50000, 200000, 1000000]
BLAS_NTHS = [parse(Int64, ARGS[2])]
classes = [ARGS[1]]
types = ["static"]
names = Dict("LBOLBSV"=> "lowOverlap_lowBlockSizeVar", "LBOHBSV"=>"lowOverlap_highBlockSizeVar", "HBOLBSV"=>"highOverlap_lowBlockSizeVar", "HBOHBSV"=>"highOverlap_highBlockSizeVar")
f = "comp_mp3.jl"
nfail = 0
# @info "Running SpMM tests using MPI on 2D processor grids"
for N in Ns
    for sqnprocs in SQNPROCS
        nprocs = sqnprocs^2
        for class in classes
            for t in types
                for blas_nths in BLAS_NTHS
                    name = names[class]
                    for k in ks
                        try
                            # run(`julia $(joinpath(@__DIR__, f)) $nprocs $N`)
                            # run(`$(Base.julia_cmd()) $(joinpath(@__DIR__, f1)) $N $sqnprocs $what`)
                            mpiexec() do cmd
                                run(`$cmd --mca opal_warn_on_missing_libcuda 0 -n $nprocs --map-by NUMA:PE=$blas_nths $(Base.julia_cmd()) $(joinpath(@__DIR__, f)) $N $k $class $t $name`)
                            end
                            Base.with_output_color(:green,stdout) do io
                                println(io,"\tSUCCESS: $f")
                            end
                        catch ex
                            Base.with_output_color(:red,stderr) do io
                                println(io,"\tError: $f")
                                showerror(io,ex,backtrace())
                            end
                            # nfail += 1
                        end
                    end
                end
            end
        end
    end
end

