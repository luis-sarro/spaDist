require("expm")
options("scipen"=100, "digits"=1)

a <- read.table("Samples/KingMS_Ctr_7/2-foo.csv",sep=",")

nn <- knn.dist(a,25)
idx <- which.min(nn[,25])
print(a[idx,],digits=10)

sigma=68.27
n <- dim(a)[1]
nt <- floor(n*sigma/100)
nn2 <- get.knnx(a,a[idx,],k=nt)
cvmat <- cov(a[nn2$nn.index,])
print(cvmat[c(2,3,1,4,5,6),c(2,3,1,4,5,6)])

