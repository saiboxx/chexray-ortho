library(mgcv)
library(corpcor)
library(readr)
library(reticulate)
library(Metrics)
np <-import("numpy")

## Read meta data
meta <- read_csv("mimic_meta.csv")

## Label values
# 1.0 :  The label was positively mentioned in the associated study, and is 
#        present in one or more of the corresponding images, e.g. 
#        "A large pleural effusion"
# 0.0 :  The label was negatively mentioned in the associated study, and 
#        therefore should not be present in any of the corresponding images
#        e.g. "No pneumothorax."
# -1.0 : The label was either: 
#        (1) mentioned with uncertainty in the report, and therefore may or 
#            may not be present to some degree in the corresponding image, or 
#        (2) mentioned with ambiguous language in the report and it is unclear 
#            if the pathology exists or not
# Explicit uncertainty: "The cardiac size cannot be evaluated."
# Ambiguous language: "The cardiac contours are stable."
# Missing (empty element) - No mention of the label was made in the report

## ViewPosition: Filter for PA and AP -> already the case

meta$gender <- as.factor(meta$gender)

## race: Merge Asian, White, Black
meta$race <- as.factor(meta$race)
levels(meta$race) <- sapply(levels(meta$race),
                            function(lev){
                              if(grepl("ASIAN", lev)) return("ASIAN")
                              if(grepl("BLACK", lev)) return("BLACK")
                              if(grepl("HISPANIC", lev)) return("HISPANIC")
                              if(grepl("WHITE", lev)) return("WHITE")
                              return("OTHER")
                            })

## split: recommended train/val/test splits
train_ind <- meta$split != "test"
test_ind <- meta$split == "test"
meta_test <- meta[test_ind,]
meta <- meta[train_ind,]

## Read embedding
# how many of the singular values should be used
# (based on SVD gives the following explained variances
#  for different amount of columns)
# red <- 19 # 95% variance
# red <- 73 # 97,5% variance
red <- 111 # 98% variance
# red <- 311 # 99% variance

if(file.exists("emb_svd.RDS") & file.exists("emb_svd_test.RDS")){
  
  emb_svd <- readRDS("emb_svd.RDS")
  emb_svd_test <- readRDS("emb_svd_test.RDS")

}else{

  emb <- np$load("embeddings.npy")
  emb_test <- emb[test_ind,]
  emb <- emb[train_ind,]

  ## SVD
  emb_svd <- fast.svd(emb)
  saveRDS(emb_svd, file="emb_svd.RDS")
  plot(cumsum(emb_svd$d^2/sum(emb_svd$d^2)), type="b")
  # (red <- min(which(cumsum(emb_svd$d^2/sum(emb_svd$d^2))>0.98)))

  # for test
  emb_svd_test <- fast.svd(emb_test)
  saveRDS(emb_svd_test, file="emb_svd_test.RDS")

}

# create reduced embedding from SVD
emb <- emb_svd$u[,1:red]%*%diag(emb_svd$d[1:red])%*%emb_svd$v[1:red,1:red]
emb_test <- emb_svd_test$u[,1:red]%*%diag(emb_svd_test$d[1:red])%*%
  emb_svd_test$v[1:red,1:red]

## Combine
emb <- cbind(as.data.frame(emb), meta)
emb_test <- cbind(as.data.frame(emb_test), meta_test)
rm(meta, meta_test); gc()

## Create response
name_resp <- "Pleural Effusion"
# for this remove uncertain cases
resp_eff_uncertain <- which(emb[,name_resp] == -1)
emb <- emb[-resp_eff_uncertain,]
# and NAs
emb <- emb[!is.na(emb[,name_resp]),]
emb$resp <- emb[,name_resp]
# same for test
resp_uncertain_test <- which(emb_test[,name_resp] == -1)
emb_test <- emb_test[-resp_uncertain_test,]
# and NAs
emb_test <- emb_test[!is.na(emb_test[,name_resp]),]
emb_test$resp <- emb_test[,name_resp]

## Check for other NAs
emb <- emb[!is.na(emb$gender),]
emb <- emb[!is.na(emb$race),]
# test
emb_test <- emb_test[!is.na(emb_test$gender),]
emb_test <- emb_test[!is.na(emb_test$race),]

## Fit model with protected features
formla <- paste0("resp ~ 1 + anchor_age + gender + race + ",
                 paste(paste0("V", 1:red), collapse = " + "))

mod <- bam(as.formula(formla), family = "binomial",
           data = emb)

# raceBLACK, genderM significant for alpha = 0.05
# and age
summary(mod)

## Prediction performance
pred <- predict(mod, emb_test, type = "response")
(auc_all <- auc(emb_test$resp, c(pred)))

## Fit model without protected features
formla2 <- paste0("resp ~ 1 + ",
                  paste(paste0("V", 1:red), collapse = " + "))

mod2 <- bam(as.formula(formla2), family = "binomial",
            data = emb)

## Prediction performane
pred2 <- predict(mod2, emb_test, type = "response")
(auc_2 <- auc(emb_test$resp, c(pred2)))

# check how much predictions can be explained
# by protected features
emb$pred_mod2 <- predict(mod2)

mod2_explained_protected <- bam(
  pred_mod2 ~ 1 + anchor_age + gender + race,
  data = emb
)

# all significant
summary(mod2_explained_protected)
anova(mod2_explained_protected)

## Remove protected features from predictions
emb$adjusted_pred <- resid(lm(pred_mod2 ~ 1 + anchor_age + gender + race,
                              data = emb))

# check: p-values should all be 1
summary(bam(
  adjusted_pred ~ 1 + anchor_age + gender + race,
  data = emb
))

## Now adjust embedding
feat_mat <- model.matrix(~ -1 + anchor_age + gender + race,
                         data = emb) # creates X matrix
q_mat <- qr.Q(qr(feat_mat)) # => X = QR

# replace embedding
rhs <- crossprod(q_mat, as.matrix(emb[,paste0("V", 1:red)])) # Q^T U
proj_emb <- q_mat%*%rhs # = Q Q^T U = X(X^TX)^{-1} X^T U
emb[,paste0("V", 1:red)] <- emb[,paste0("V", 1:red)] - proj_emb  # = U - X(X^TX)^{-1} X^T U = P_X^bot U

## Now try again to see how much protected features are in the predictions
mod2_fixed <- bam(as.formula(formla2), family = "binomial",
                  data = emb)

## Prediction performane
pred2_fixed <- predict(mod2_fixed, emb_test, type = "response")
(auc_fixed <- auc(emb_test$resp, c(pred2_fixed)))

# check how much predictions can be explained
# by protected features
emb$pred_mod2_fixed <- predict(mod2_fixed)

mod2_explained_fixed <- bam(
  pred_mod2_fixed ~ 1 + anchor_age + gender + race,
  data = emb
)

# run inference
summary(mod2_explained_fixed)
anova(mod2_explained_fixed)
# => all p-values are 1, nothing can be explained by protected features anymore