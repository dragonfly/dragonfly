#########################################################
# config.pb: This file demonstrates the way to pass     #
#            hyperparameters info to dragonfly script.  #
#                                                       #
# Required Fields:                                      #
# name: "name of the python file"                       #
#                                                       #
# domain {                                              #
#   name: "hyperparameter name"                         #
#   type: "int or float data type"                      #
#   min:  "min value of the domain"                     #
#   max:  "max value of the domain"                     #
#   dim:  "dimension of the hyperparameter"             #
# }                                                     #
#                                                       #
# Write the above "domain" structure for each           #
# hyperparameter present in the function.               #
#                                                       #
# -- kvysyara@andrew.cmu.edu                            #
#########################################################
name: "branin"

domain {
 name: "x1"
 type: "float"
 min: -5.0
 max: 10.0
 dim: "1"
}

domain {
 name: "x2"
 type: "float"
 min: 0.0
 max: 15.0
 dim: "1"
}
