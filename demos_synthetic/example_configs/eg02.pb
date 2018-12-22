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
name: "face_rec"

domain {
 name: "N"
 type: "int"
 min: 1
 max: 500
 dim: "1"
}

domain {
 name: "C"
 type: "float"
 min: 0
 max: 1000
 dim: "1"
}

domain {
 name: "gamma"
 type: "float"
 min: 0
 max: 1
 dim: "1"
}
