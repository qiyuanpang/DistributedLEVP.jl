using Test, MPIClusterManagers
using MPI

ks = [16, 32, 64]
SQNPROCS = [parse(Int, ARGS[3])]
itermaxs = [40]
Ns = [5000 20000 50000 200000 1000000]
tols = [1e-1, 1e-2, 1e-3]
classes = [ARGS[1]]
types = [ARGS[2]]
names = Dict("LBOLBSV"=> "lowOverlap_lowBlockSizeVar", "LBOHBSV"=>"lowOverlap_highBlockSizeVar", "HBOLBSV"=>"highOverlap_lowBlockSizeVar", "HBOHBSV"=>"highOverlap_highBlockSizeVar")
f = "comp_sg.jl"
nfail = 0
# @info "Running SpMM tests using MPI on 2D processor grids"
for N in Ns
    for sqnprocs in SQNPROCS
        nprocs = sqnprocs^2
        for class in classes
            for t in types
                name = names[class]
                # run(`python $(joinpath(@__DIR__, f1)) $N $sqnprocs $class $t $name`)
                for k in ks
                    for tol in tols
                        try
                            # run(`julia $(joinpath(@__DIR__, f)) $nprocs $N`)
                            # run(`$(Base.julia_cmd()) $(joinpath(@__DIR__, f1)) $N $sqnprocs $what`)
                            mpiexec() do cmd
                                run(`$cmd -n $nprocs $(Base.julia_cmd()) $(joinpath(@__DIR__, f)) $N $k $tol $class $t $name`)
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
                # run(`$(Base.julia_cmd()) $(joinpath(@__DIR__, f2)) $N $sqnprocs $class $t $name`)
            end
        end
    end
end

