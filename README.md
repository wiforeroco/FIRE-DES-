# FIRE-DES++

Development, application and analysis of the FIRE-DES++ technique for the dynamic selection of sets of classifiers.

This work is framed in the context of the combination of classifiers for classification problems. More specifically, it focuses on the techniques of dynamic selection of classifiers DES. The DES techniques aim to select one or more suitable and competent classifiers for the classification of a new test instance. Most DES techniques estimate the competence of the classifiers using a given criterion on the region of competence of the test instance, generally defined as the nearest neighbors to the example to be classified. The technique Frienemy Indecision Region Dynamic Ensem- ble Selection (FIRE-DES) uses for this type of problems the “frienemies” (instances of different classes) that pre-selects classifiers, and then correctly classifies the less a pair of frienemy, allowing to generate a filter of competent classifiers. However, the FIRE-DES technique confuses noisy regions with regions of indecision, leading to the pre-selection of noncompetent classifiers leaving examples in different regions without any pre-selection. 

Therefore, the same authors propose an improvement called FIRE-DES ++ based on the article of “FIRE-DES ++ Enhanced online pruning of base classifiers for dynamic ensemble selection” of the authors Rafael MO Cruz, Dayvid V. R. Oliveira, George D. C. Cavalcanti and Robert Sabourin. They propose the FIRE-DES++ technique which is analogous to the FIRE-DES technique but improved, eliminating the noise and reducing the overlap of classes in the validation set. It also defines the region of compe- tence using the K-Nearest Neighbors Equality (KNNE) algorithm.
