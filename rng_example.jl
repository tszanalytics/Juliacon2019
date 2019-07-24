using Distributions, Plots
# Knowing the RNG algorithm,the seed and a sample, we
# can determine theta?

# Not knowing the algorithm and seed, we only
# have a random sample, so our knowledge is uncertain.

# theta is not random, the posterior density represents
# our uncertain knowledge of theta.
M = 100000    # number of psuedo-random draws from the posterior
theta = 0.3
n10 = 10   # start with only 10 observations
Random.seed!(41)
x10 = rand(Bernoulli(theta),n10)
@show(x10)
s = sum(x10)
post_theta10 = rand(Beta(s+1,n10-s+1),M)  # Beta posterior
plt = plot(post_theta10,st=:density,label="n = 10",xaxis=[0.0,0.7])

N = collect(100:200:1000)
for n in N
    print(n," ")
    x = rand(Bernoulli(theta),n)
    s = sum(x)
    post_theta = rand(Beta(s+1,n-s+1),M)
    plot!(post_theta, st=:density,alpha=0.5,label="n = $n")
end
plot(plt)

n = 10000
x = rand(Bernoulli(theta),n)
s = sum(x)
post_theta = rand(Beta(s+1,n-s+1),M)
plot!(post_theta, st=:density,linecolor=:black,linewidth=2,label="n = $n")
vline!([0.3],linecolor=:red,linewidth=0.3,alpha=0.8,label="")
#savefig("approaching_certainty_03.png")
