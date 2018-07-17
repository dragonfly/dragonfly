#########################################################
# config.pb: This file demonstrates the way to pass     #
#            hyperparameters info to dragonfly script.  #
#                                                       #
# Required Fields:                                      #
# name: "name of the python file"                       #
#                                                       #
# variable {                                            #
#   name: "variable name"                               #
#   type: "int or float data type"                      #
#   dim:  "dimension of the hyperparameter"             #
#   min:  "min value of the domain"                     #
#   max:  "max value of the domain"                     #
# }                                                     #
#                                                       #
# Write the above "variable" structure for each         #
# hyperparameter present in the function.               #
#                                                       #
# -- kvysyara@andrew.cmu.edu                            #
#########################################################
name: "branin"

variable {
 name: "x1"
 type: FLOAT
 dim: 1
 min:  -5.0
 max:  10.0
}

variable {
 name: "x2"
 type: FLOAT
 dim: 1
 min:  0.0
 max:  15.0
}
