
using JuLIP, Test, Printf
using JuLIP.Testing

include("aux.jl")
verbose=true

## check whether on CI
isCI = haskey(ENV, "CI")
notCI = !isCI
eam_W4 = nothing

## check whether ASE is available
global hasase = true
try
   import ASE
catch
   global hasase = false
end

julip_tests = [
   ("testaux.jl", "Miscellaneous"),
   ("test_atoms.jl", "Atoms"),
   ("test_build.jl", "Build"),
   ("testanalyticpotential.jl", "Analytic Potential"),
   ("testpotentials.jl", "Potentials"),
   ("test_ad.jl", "AD Potentials"),
   ("testvarcell.jl", "Variable Cell"),
   ("testhessian.jl", "Hessian"),
   ("testsolve.jl", "Solve"),
   ("test_fio.jl", "File IO"),
]

# remove testsolve if on Travis
if isCI
   julip_tests = julip_tests[1:end-1]
end

# "testexpvarcell.jl";  # USE THIS TO WORK ON EXPCELL IMPLEMENTATION

## ===== some prototype potentials ======
print("Loading some interatomic potentials . .")
data = joinpath(dirname(pathof(JuLIP)), "..", "data") * "/"
eam_Fe = JuLIP.Potentials.EAM(data * "pfe.plt", data * "ffe.plt", data * "F_fe.plt")
print(" .")
eam_W = JuLIP.Potentials.FinnisSinclair(data*"W-pair-Wang-2014.plt", data*"W-e-dens-Wang-2014.plt")
print(" .")
global eam_W4
try
   global eam_W4 = JuLIP.Potentials.EAM(data * "w_eam4.fs")
catch
   global eam_W4 = nothing
end
println(" done.")

##
h0("Starting JuLIP Tests")

@testset "JuLIP" begin
   for (testfile, testid) in julip_tests
      h1("Testset $(testid)")
      @testset "$(testid)" begin include(testfile); end
   end
end


at = bulk(:W, cubic=true) * 2
set_pbc!(at, false)
set_constraint!(at, FixedCell(at))
set_calculator!(at, eam_W4)
fdtest( x -> energy(at, x), x -> JuLIP.gradient(at, x), dofs(at) )
fdtest_hessian( x->gradient(at, x), x->hessian(at, x), dofs(at) )

eam = eam_W4
println("test a single stencil")
r = []
R = []
for (_1, _2, r1, R1) in sites(at, cutoff(eam))
   global r = r1
   global R = R1
   break
end

r = r[1:3]
R = R[1:3]

# evaluate site gradient and hessian
using LinearAlgebra
dVs = JuLIP.Potentials.evaluate_d(eam, r, R)
hVs = hess(eam, r, R)
# and convert them to vector form
dV = mat(dVs)[:]
hV = zeros(3*size(hVs,1), 3*size(hVs,2))
for i = 1:size(hVs,1), j = 1:size(hVs,2)
   hV[3*(i-1).+(1:3), 3*(j-1).+(1:3)] = hVs[i,j]
end
matR = mat(R)

for p = 3:9
   h = 0.1^p
   hVh = fill(0.0, size(hV))
   for n = 1:length(matR)
      matR[n] += h
      r = norm.(R)
      dVh = mat(JuLIP.Potentials.evaluate_d(eam, r, R))[:]
      hVh[:, n] = (dVh - dV) / h
      matR[n] -= h
   end
   @printf("%1.1e | %4.2e \n", h, norm(hVh - hV, Inf))
end
