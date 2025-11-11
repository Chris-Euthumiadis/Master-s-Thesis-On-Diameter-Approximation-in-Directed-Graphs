# -*- coding: utf-8 -*-
import numpy as np

import algorithms
import math

# Αλγόριθμος για προσεγγιστικό υπολογισμό της διαμέτρου
# Παίρνει σαν παράμετρο μη αρνητικό ακέραιο αριθμό D
def algA_d(A, D):
    print("D=" + str(D))
    # Αριθμός κόμβων αρχικού γράφου
    n = A.shape[0]
    (d, d1, d2) = algorithms.calcDegrees(A)
    # Αριθμός ακμών του αρχικού γράφου
    m = int(sum(d)/2)
    
    # Βήμα 1
    # Δημιουργία γράφου με μέγιστο βαθμό κορυφών 3
    # print("Step 1")
    # print("Δημιουργία γράφου με μέγιστο βαθμό κορυφής ίσο με 3")
    B, no_nodes, no_edges = algorithms.max3degree_graph(A, d, n)
    # print("B=")
    # print(B)
    
    # Υπολογισμός έσω-εκκεντρότητας για κάθε μία από τις κορυφές
    inner_eccentricities = []
    # print("**Έσω-Εκκεντρότητα")
    for node in range(no_nodes):
        e = algorithms.inner_eccentricity(node, B, no_nodes)
        inner_eccentricities.append(e)
    # print(inner_eccentricities)
    
    # Υπολογισμός έξω-εκκεντρότητας για κάθε μία από τις κορυφές
    outer_eccentricities = []    
    # print("Έξω-Εκκεντρότητα")
    for node in range(no_nodes):
        e = algorithms.outer_eccentricity(node, B, no_nodes)
        outer_eccentricities.append(e)
    # print(outer_eccentricities)

    if math.inf in inner_eccentricities or math.inf in inner_eccentricities:
        return None

    # Παράμετροι α και ω
    # ω = fast matrix multiplication exponent
    omega = 2.37286
    alpha = (omega+1)/(omega+5)
    # print("alpha="+str(alpha))
    # Η λίστα με όλες τις κορυφές
    list_of_nodes = [x for x in range(no_nodes)]
    # print("List of nodes=" + str(list_of_nodes))
    # print("Πλήθος κορυφών στον νέο γράφο:" + str(no_nodes))
    # Βήμα 2
    # Επιλέγουμε δείγμα με 4m^αlogm κορυφές
    # print("Step 2")
    sample_size = int(4*m**alpha*math.log10(m))
    ### sample_size = int(4 * m ** alpha * math.log(m))
    # print("sample_size=" + str(sample_size))
    # Επιλέγουμε με τυχαίο τρόπο κάποιες κορυφές
    selected = algorithms.choose_random(list_of_nodes, sample_size)
    # print(selected)
    
    # Αν κάποια από τις επιλεγμένες κορυφές έχει έσω ή έξω εκκεντρρότητα >= 4D/7
    # ο αλγόριθμος αποδέχεται
    limit = 4*D/7

    # print("4D/7 = "+str(limit))
    # print("Έλεγχος αν η εσωτερική ή εξωτερική εκκεντρότητα μίας κορυφής του δείγματος είναι τουλάχιστον 4D/7")
    for i in selected:    
        # print("i="+str(i))
        # print("inner eccentricity=" + str(inner_eccentricities[i]))

        if inner_eccentricities[i] >= limit or outer_eccentricities[i] >= limit:
            # print("Βρέθηκε τέτοια κορυφή")
            # print("Accept: Step 2")
            return True
    # print("Δεν βρέθηκε τέτοια κορυφή")
    # Βήμα 3
    # Ελέγχουμε αν υπάρχει κορυφή v όπου το πλήθος κορυφών στο Bout(v) για απόσταση
    # D/7 είναι <= m^a
    # για κάθε μία κορυφή
    # print("Step 3")
    list_of_nodes = [x for x in range(no_nodes)]
    # print("m**alpha="+str(m**alpha))
    # print("Για κάθε κόμβο υπολογίζουμε το πλήθος των κόμβων στο Bout για D/7")
    # print("Αν για κάποιον κόμβο είναι <= m^a και στο Bout+ αν υπάρχει κάποιος κόμβος έχει εκκεντρότητα τυολάχιστον 4D/7")
    # print("τότε ο αλγόριθμος αποδέχεται")
    for v in list_of_nodes:
        Bout_set, count = algorithms.find_Bout1(B, no_nodes, v, D/7)
        # print("COUNT for "+str(v)+"="+str(count))
        if count <= m**alpha:           
            # Αν υπάρχει τέτοια κορυφή, τότε ελέγχουμε αν υπάρχει κορυφή στο σύνολο
            # Bout_plus με εκκεντρότητα τουλάχιστον 4D/7            
            # Βρίσκουμε το σύνολο Bout_plus
            Bout_plus, count = algorithms.find_Bout_plus1(B, no_nodes, v, D/7)
            for node in Bout_plus:
                if inner_eccentricities[node] >= limit or outer_eccentricities[node] >= limit:
                    # print("Βρέθηκε τέτοια κορυφή")
                    # print("Accept: Step 3")
                    return True
    # print("Δεν βρέθηκε τέτοια κορυφή")

    # Βήμα 4
    # Ελέγχουμε αν υπάρχει κορυφή v όπου το πλήθος κορυφών στο Bin(v) για απόσταση
    # D/7 είναι <= m^a
    # για κάθε μία κορυφή
    # list_of_nodes = [x for x in range(no_nodes)]
    # print("Step 4")
    # print("Για κάθε κόμβο υπολογίζουμε το πλήθος των κόμβων στο Bin για D/7")
    # print("Αν για κάποιον κόμβο είναι <= m^a και στο Bout+ αν υπάρχει κάποιος κόμβος έχει εκκεντρότητα τυολάχιστον 4D/7")
    # print("τότε ο αλγόριθμος αποδέχεται")
    for v in list_of_nodes:
        Bin_set, count = algorithms.find_Bin1(B, no_nodes, v, D/7)
        if count <= m**alpha:           
            # Αν υπάρχει τέτοια κορυφή, τότε ελέγχουμε αν υπάρχει κορυφή στο σύνολο
            # Bin_plus με εκκεντρότητα τουλάχιστον 4D/7            
            # Βρίσκουμε το σύνολο Bin_plus
            Bin_plus, count = algorithms.find_Bin_plus1(B, no_nodes, v, D/7)
            for node in Bin_plus:
                if inner_eccentricities[node] >= limit or outer_eccentricities[node] >= limit:
                    # print("Βρέθηκε τέτοια κορυφή")
                    # print("Accept: Step 4")
                    return True
    # print("Δεν βρέθηκε τέτοια κορυφή")

    # Βήμα 5
    # Παίρνουμε δείγμα S_v με 4m^(1-α)logm κορυφές
    # print("Step 5")

    sample_size = int(4*m**(1-alpha)*math.log10(m))
    ### sample_size = int(4 * m ** (1 - alpha) * math.log(m))
    # print("Sample size = " + str(sample_size))
    # Η λίστα με όλες τις κορυφές
    # list_of_nodes = [x for x in range(no_nodes)]
    S_v = algorithms.choose_random(list_of_nodes, sample_size)
    list_of_nodes = [x for x in range(no_nodes)]
    # print(list_of_nodes)
    # print("S_v=")
    # print(S_v)
    Sout = []
    # print("m**(1-a)="+str(m**(1-alpha)))
    # print("2*D/7="+str(2*D/7))
    for s in S_v:
        # print(s)
        Bout, count = algorithms.find_Bout1(B, no_nodes, s, 2*D/7)
        # print("count="+str(count))
        if count <= m**(1-alpha):
            Sout.append(s)
    # print("Sout="+str(Sout))
    
    Sin = []
    for s in S_v:
        # print(s)
        Bin, count = algorithms.find_Bin1(B, no_nodes, s, 2*D/7)
        # print(count)
        if count <= m**(1-alpha):
            Sin.append(s)
    # print("Sin="+str(Sin))

    # Βήμα 6
    # Δημιουργία πίνακα Aout
    Aout = np.zeros(shape=(len(Sout), no_nodes), dtype=int)
    for s in Sout:
        Bout, count = algorithms.find_Bout1(B, no_nodes, s, 2*D/7)
        for v in range(no_nodes):
            if v in Bout:
                Aout[Sout.index(s),v] = 1

    # Δημιουργία πίνακα Ain
    Ain = np.zeros(shape=(no_nodes, len(Sin)), dtype=int)
    if int(4*D/7) == 2*int(2*D/7):
        for s in Sin:
            Bin, count = algorithms.find_Bin1(B, no_nodes, s, 2 * D / 7)
            for v in range(no_nodes):
                if v in Bin:
                    Ain[v, Sin.index(s)] = 1
    else:
        for s in Sin:
            Bin_plus, count = algorithms.find_Bin_plus1(B, no_nodes, s, 2 * D / 7)
            for v in range(no_nodes):
                if v in Bin_plus:
                    Ain[v, Sin.index(s)] = 1

    product = np.matmul(Aout, Ain)
    # print("product=")
    # print(product)
    if 0 in product:
        return True
    else:
        return False



# Adjacency matrix
#A = np.array([[0, 1, 1, 1, 1, 0, 0], 
#                [1, 0, 1, 0, 0, 0, 0],
#                [1, 1, 0, 0, 1, 1, 1],
#                [1, 0, 0, 0, 1, 0, 0],
#                [1, 0, 1, 1, 0, 1, 0],
#                [0, 0, 1, 0, 1, 0, 1],
#                [0, 0, 1, 0, 0, 1, 0]]) 




A = np.array([[0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0, 1],
                [0, 1, 0, 0, 0, 0]])


n=10
p=0.3
A = algorithms.create_directed_graph(n, p)
print("A=")
print(A)

# Δοκιμή αναζήτησης BFS για τον υπολογισμό της διαμέτρου
diam = algorithms.calc_diameter(A, n)
print("Με BFS η διάμετρος υπολογίστηκε σε:" + str(diam))

lo =0
hi = n
while hi-lo>1:
    mid = int((lo+hi)/2)
    accept = algA_d(A, mid)
    if accept:
        lo = mid
    else:
        hi=mid
print("diam="+str(lo))
# diam = 0
# for D in range(1,n):
#     # print("D="+str(D))
#     accept = algA_d(A,D)
#     if accept == False:
#         diam = D-1
#         break
#     if accept == None:
#         print("Η διάμετρος είναι άπειρη")
#     print(accept)
# print(diam)





    


