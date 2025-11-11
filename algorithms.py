# -*- coding: utf-8 -*-
from asyncio.windows_events import INFINITE
from turtledemo.penrose import inflatedart

import numpy as np
from heapq import heapify, heappop, heappush
import math
import random

# Συνάρτηση που δημιουργεί έναν πίνακα nxn
# Ο πίνακας είναι συμμετρικός και έχει πιθανότητα p
# να έχει 1 σε μία θέση (i,j), διαφορετικά έχει τιμή 0
# Ο πίνακας αντιστοιχεί στον πίνακα γειτνίασης μη κατευθυνόμενου γράφου
def create_graph(n, p):
    A = np.zeros(shape=(n, n), dtype=int)
    for i in range(n):
        A[i,i] = 0
        for j in range(i+1, n):
            x = random.uniform(0, 1)
            if x<p:
                A[i,j] = 1
                A[j,i] = 1
            # else:
            #     A[i, j] = math.inf
            #     A[j, i] = math.inf

    return A

# Συνάρτηση που δημιουργεί έναν πίνακα nxn ως πίνακα γειτνίασης
# ενός κατευθυνόμενου γράφου
def create_directed_graph(n, p):
    A = np.zeros(shape=(n, n), dtype=int)
    for i in range(n):
        A[i,i] = 0
        for j in range(n):
            if i != j:
                x = random.uniform(0, 1)
                if x<p:
                    A[i,j] = 1
                # else:
                #     A[i,j] = math.inf
    return A

# Συνάρτηση για τον υπολογισμό των βαθμών των κορυφών του γράφου
# d1 είναι η λίστα των έσω-βαθμών (άθροισμα ανά γραμμές)
# d2 είναι η λίστα των έξω-βαθμών (άθροισμα ανά στήλες)
# d είναι το άθροισμα έσω και έξω-βαθμών
def calcDegrees(A):
    d1 = np.sum(A, axis=0)
    d2 = np.sum(A, axis=1)
    d = d1 + d2
    return (d, d1, d2)

# Συνάρτηση για τον υπολογισμό της διαμέτρου ενός γράφου χωρίς βάρη
# A είναι ο πίνακας γειτνίασης του γράφου
# n είναι το πλήθος των κορυφών
def calc_diameter(A, n):
    max_distances = []
    # Υπολογισμός απόστασης για κάθε κορυφή i από την πιο απομακρυσμένη κορυφή
    for i in range(n):
        max_dist = bfs(A, n, i)
        max_distances.append(max_dist)
    # Επιστροφή της μέγιστης από τις αποστάσεις, που είναι η διάμετρος
    return max(max_distances)
    

# Συνάρτηση η οποία εφαρμόζει αναζήτηση κατά πλάτος σε έναν γράφο χωρίς βάρη
# ξεκινώντας από συγκεκριμένη κορυφή του γράφου
# A είναι ο πίνακας γειτνίασης του γράφου
# n είναι το πλήθος των κορυφών
# node είναι η κορυφή από την οποία ξεκινά η αναζήτηση
# Η συνάρτηση επιστρέφει την απόσταση από την πιο απομακρυσμένη κορυφή
def bfs(A, n, node):
    # Καταχώρηση των κομβων με τις αντίστοιχες αποστάσεις από τη ρίζα του δέντρου
    # που απεικονίζει την αναζήτηση BFS
    list_of_nodes = {node:0}
    # Λίστα με τους κόμβους που εξετάζουμε 
    open_nodes = [node]
    # Όσο υπάρχουν κόμβοι που είναι υπό-εξέταση
    while len(open_nodes) > 0:
        # Παίρνουμε τον επόμενο κόμβο από την αρχή της λίστας
        current_node = open_nodes[0]
        # Τον αφαιρούμε από τη λίστα
        open_nodes.pop(0)
        for i in range(n):
            # προσθέτουμε τους γείτονές του στη λίστα
            if A[current_node, i] > 0:
                # Αν δεν έχουν ήδη εξεταστεί
                if (i not in open_nodes) and (i not in list_of_nodes.keys()): 
                    open_nodes.append(i)
                    list_of_nodes[i] = list_of_nodes[current_node] + 1
    # Αν ο γράφος είναι μη συνεκτικός έχουμε άπειρη διάμετρο
    if len(list_of_nodes) < n:
        return math.inf
    else:
        return max(list_of_nodes.values())


# Συνάρτηση η οποία εφαρμόζει αναζήτηση κατά πλάτος σε έναν γράφο χωρίς βάρη
# ξεκινώντας από συγκεκριμένη κορυφή node του γράφου και φτάνοντας σε μια άλλη κορυφή target
# A είναι ο πίνακας γειτνίασης του γράφου
# n είναι το πλήθος των κορυφών
# node είναι η κορυφή από την οποία ξεκινά η αναζήτηση
# target είναι η κορυφή-στόχος
# Η συνάρτηση επιστρέφει την απόσταση από την κορυφή node στην κορυφή target
def bfs1(A, n, node, target):
    # Καταχώρηση των κομβων με τις αντίστοιχες αποστάσεις από τη ρίζα του δέντρου
    # που απεικονίζει την αναζήτηση BFS
    list_of_nodes = {node:0}
    # Λίστα με τους κόμβους που εξετάζουμε
    open_nodes = [node]
    # Όσο υπάρχουν κόμβοι που είναι υπό-εξέταση
    while len(open_nodes) > 0:
        # Παίρνουμε τον επόμενο κόμβο από την αρχή της λίστας
        current_node = open_nodes[0]
        # Τον αφαιρούμε από τη λίστα
        open_nodes.pop(0)
        for i in range(n):
            # προσθέτουμε τους γείτονές του στη λίστα
            if A[current_node, i] > 0:
                # Αν βρήκαμε το στόχο
                if i == target:
                    return list_of_nodes[current_node] + 1
                # Αν δεν έχουν ήδη εξεταστεί
                if (i not in open_nodes) and (i not in list_of_nodes.keys()):
                    open_nodes.append(i)
                    list_of_nodes[i] = list_of_nodes[current_node] + 1
    # Αν ο γράφος είναι μη συνεκτικός έχουμε άπειρη διάμετρο
    if len(list_of_nodes) < n:
        return math.inf
    # else:
    #     return max(list_of_nodes.values())
     
# Δημιουργία πίνακα με μέγιστο βαθμών κορυφών το 3
# A είναι ο πίνακας γειτνίασης του γράφου
# d είναι το διάνυσμα βαθμών των κορυφών του γράφου
# n είναι το πλήθος των κορυφών
# το πλήθος κορυφών του νέου γράφου είναι ίσο με nodes
# το πλήθος ακμών (τόξων) του νέου γράφου είναι ίσο με edges
def max3degree_graph(A, d, n):
    # Άθροισμα βαθμών που είναι ίσο με το πλήθος κορυφών του νέου γράφου
    nodes = sum(d)
    # Αρχικά δημιουργούμε πίνακα με τιμές ίσες με άπειρο    
    B = np.ones((nodes, nodes), dtype=int)
    B = math.inf*B

    # Ελέγχουμε όλες τις ακμές του αρχικού πίνακα γειτνίασης
    for i in range(n):
        for j in range(n):
            if A[i,j] == 1:
                # Εύρεση κατάλληλης θέσης στον νέο πίνακα γειτνίασης
                k = getNext(B, d, nodes, i, i)
                l = getNext(B, d, nodes, k, j)
                # Τοποθέτηση 1 (ακμής) στον νέο πίνακα γειτνίασης
                B[k,l] = 1
    # Δημιουργία των ακμών μηδενικού βάρους (κατευθυνόμενοι κύκλοι του νέου γράφου)
    for i in range(n):
        # Εύρεση σωστής θέσης
        pos = 0
        for k in range(i):
            pos += d[k]

        for j in range(d[i]):
            for k in range(d[i]):
                # Κάθε νέα (τεχνητή) κορυφή συνδέεται με ακμή με την επόμενη του κύκλου
                if (k == (j+1)%d[i]):
                    B[pos+j,pos+k] = 0
    # Η κύρια διαγώνιος περιλαμβάνει μηδενικά
    for i in range(nodes):
        B[i,i] = 0
    # Το πλήθος των ακμών
    edges = 3 * nodes / 2
    # Επιστροφή του νέου πίνακα γειτνίασης και του πλήθους κορυφών
    return B, nodes, edges

# Συνάρτηση που επιστρέφει την 1η κορυφή από τον κύκλο που έχει δημιουργηθεί ο οποίος δεν 
# έχει συνδεθεί με κάποια άλλη κορυφή
def getNext(B, d, total, i, j):
    pos = 0
    for k in range(j):
        pos += d[k]

    for k in range(pos, pos+d[j]):
        # Έλεγχος αν η κορυφή έχει ήδη συνδεθεί με άλλη
        if notAssigned(B, k, total):
            return k
    
    return -1

# Συνάρτηση που επιστρέφει true αν η κορυφή k δεν έχει συνδεθεί ακόμη με άλλη 
# κορυφή στον νέο γράφο
def notAssigned(B, k, total):
    # Έλεγχος γραμμών νέου πίνακα γειτνίασης
    for i in range(total):
        if B[k,i] == 1:
            return False
    # Έλεγχος στηλέων νέου πίνακα γειτνίασης
    for i in range(total):
        if B[i,k] == 1:
            return False
    return True


# Αλγόριθμος Dijkstra για τον υπολογισμό συντομότερων αποστάσεων 
# από την κορυφή source σε όλες τις άλλες κορυφές του γράφου
# ο γράφος έχει πίνακα γειτνίασης Α
# n: πλήθος κορυφών γράφου
def shortest_distances_from_source(source, A, n):
    # πίνακας αποστάσεων
    distances = {node: float("inf") for node in range(n)}
    distances[source] = 0
    # Ουρά προτεραιότητας
    pq = [(0, source)]
    heapify(pq)
    # Σύνολο κόμβων που έχουμε επισκεφθεί ήδη
    visited = set()
    while pq:  # Όσο η ουρά δεν είναι άδεια
           current_distance, current_node = heappop(pq) 
           # Αν έχουμε ήδη επισκεφθεί τον κόμβο
           if current_node in visited:
               continue  
           # Αν δεν τον έχουμε επισκεφθεί τον προσθέτουμε στην ουρά
           visited.add(current_node) 
           # Ελέγχουμε τις γειτονικές κορυφές
           for j in range(n):
               if(A[current_node,j] < math.inf):
                   # νέα απόσταση μέσω του άλλου κόμβου
                   new_distance = current_distance + A[current_node,j]
                   # Αν η νέα απόσταση είναι μικρότερη από την προηγούμενη
                   # ενημερώνουμε την ουρά
                   if new_distance < distances[j]:
                       distances[j] = new_distance
                       heappush(pq, (new_distance, j))
    return distances

# Αλγόριθμος Dijkstra για τον υπολογισμό συντομότερων αποστάσεων 
# από όλες τις κορυφές του γράφου προς μία συγκεκριμένη κορυφή-στόχο
# ο γράφος έχει πίνακα γειτνίασης Α
# n: πλήθος κορυφών γράφου
def shortest_distances_to_target(target, A, n):
    # πίνακας αποστάσεων
    distances = {node: float("inf") for node in range(n)}
    distances[target] = 0
    # Ουρά προτεραιότητας
    pq = [(0, target)]
    heapify(pq)
    # Σύνολο κόμβων που έχουμε επισκεφθεί ήδη
    visited = set()
    while pq:  # Όσο η ουρά δεν είναι άδεια
           current_distance, current_node = heappop(pq) 
           # Αν έχουμε ήδη επισκεφθεί τον κόμβο
           if current_node in visited:
               continue  
           # Αν δεν τον έχουμε επισκεφθεί τον προσθέτουμε στην ουρά
           visited.add(current_node) 
           # Ελέγχουμε τις γειτονικές κορυφές
           for j in range(n):
               if(A[j, current_node] < math.inf):
                   # νέα απόσταση μέσω του άλλου κόμβου
                   new_distance = current_distance + A[j, current_node]
                   # Αν η νέα απόσταση είναι μικρότερη από την προηγούμενη
                   # ενημερώνουμε την ουρά
                   if new_distance < distances[j]:
                       distances[j] = new_distance
                       heappush(pq, (new_distance, j))
    return distances

# Συνάρτηση που επιστρέφει την έσω-εκκεντρότητα της κορυφής node
# για έναν γράφο με πίνακα γειτνίασης Α και πλήθος κορυφών n
def inner_eccentricity(node, A, n):
    distances = shortest_distances_to_target(node, A, n)
    return max(distances.values())

# Συνάρτηση που επιστρέφει την έξω-εκκεντρότητα της κορυφής node
# για έναν γράφο με πίνακα γειτνίασης Α και πλήθος κορυφών n
def outer_eccentricity(node, A, n):
    distances = shortest_distances_from_source(node, A, n)
    return max(distances.values())

# Συνάρτηση που παίρνει μία λίστα με κορυφές list
# και επιστρέφει λίστα με k από αυτές επιλεγμένες τυχαία
def choose_random(list, k):
    list1 = []
    for i in range(k):
        x = random.choice(list)
        list1.append(x)
        list.remove(x)
    return list1

# Υπολογισμός πλήθους κορυφών u οι οποίες απέχουν από την κορυφή v 
# απόσταση το πολύ r
def find_Bout(A, n, v, r):
    # Υπολογισμός συντομότερων αποστάσεων από την κορυφή v
    distances = shortest_distances_from_source(v, A, n)
    set_of_nodes = set()
    # print(distances)
    # Ελέγχουμε όλες τις κορυφές και όσες απέχουν απόσταση 
    # μέχρι r τις προσθέτουμε στο σύνολο
    for d in distances:
        if(distances[d]<=r):
            set_of_nodes.add(d)
    return set_of_nodes, len(set_of_nodes)
    

# Υπολογισμός πλήθους κορυφών οι οποίες απέχουν από την κορυφή v
# απόσταση το πολύ r με τη βοήθεια αναζήτησης κατά πλάτος
def find_Bout1(A, n, v, r):
    # Υπολογισμός συντομότερων αποστάσεων από την κορυφή v
    set_of_nodes = set()
    for i in range(n):
        d = bfs1(A, n, v, i)
        if(d <= r):
            set_of_nodes.add(d)
    return set_of_nodes, len(set_of_nodes)



# Συνάρτηση που επιστρέφει τις κορυφές του συνόλου Bout μίας κορυφής v
# μαζί με αυτές που γειτονεύουν μαζί τους
def find_Bout_plus(A, n, v, r):
    # Υπολογισμός συντομότερων αποστάσεων από την κορυφή v
    distances = shortest_distances_from_source(v, A, n)
    set_of_nodes = set()
    # print(distances)
    # Ελέγχουμε όλες τις κορυφές και όσες απέχουν απόσταση 
    # μέχρι r τις προσθέτουμε στο σύνολο
    for d in distances:
        if(distances[d]<=r):
            set_of_nodes.add(d)
        # Ελέγχουμε τους γείτονες όσων έχουν απόσταση r 
        # και τους προσθέτουμε στο σύνολο
        if(distances[d]==r):
            for u in range(n):
                if(A[d,u] == 1):
                    set_of_nodes.add(u)
    return set_of_nodes, len(set_of_nodes)
    

# Υπολογισμός πλήθους κορυφών οι οποίες απέχουν από την κορυφή v
# απόσταση το πολύ r με τη βοήθεια αναζήτησης κατά πλάτος
def find_Bout_plus1(A, n, v, r):
    # Υπολογισμός συντομότερων αποστάσεων από την κορυφή v
    set_of_nodes = set()
    for i in range(n):
        d = bfs1(A, n, v, i)
        if(d <= r):
            set_of_nodes.add(d)
        # προσθέτουμε και τους γείτονες αν η απόσταση είναι ίση με r
        if (d == r):
            for u in range(n):
                if (A[v, u] == 1):
                    set_of_nodes.add(u)
    return set_of_nodes, len(set_of_nodes)

# Υπολογισμός πλήθους κορυφών u για τις οποίες η απόσταση προς την κορυφή v 
# το πολύ r
def find_Bin(A, n, v, r):
    # Υπολογισμός συντομότερων αποστάσεων προς την κορυφή v
    distances = shortest_distances_to_target(v, A, n)
    set_of_nodes = set()
    # print(distances)
    # Ελέγχουμε όλες τις κορυφές και όσες απέχουν απόσταση 
    # μέχρι r τις προσθέτουμε στο σύνολο
    for d in distances:
        if(distances[d]<=r):
            set_of_nodes.add(d)
    return set_of_nodes, len(set_of_nodes)    
    
# Υπολογισμός πλήθους κορυφών u για τις οποίες η απόσταση προς την κορυφή v
# το πολύ r
def find_Bin1(A, n, v, r):
    # Υπολογισμός συντομότερων αποστάσεων από την κορυφή v
    set_of_nodes = set()
    for i in range(n):
        d = bfs1(A, n, i, v)
        if(d <= r):
            set_of_nodes.add(d)
    return set_of_nodes, len(set_of_nodes)

# Συνάρτηση που επιστρέφει τις κορυφές του συνόλου Bin μίας κορυφής v
# μαζί με αυτές που γειτονεύουν μαζί τους
def find_Bin_plus(A, n, v, r):
    # Υπολογισμός συντομότερων αποστάσεων από την κορυφή v
    distances = shortest_distances_to_target(v, A, n)
    set_of_nodes = set()
    # print(distances)
    # Ελέγχουμε όλες τις κορυφές και όσες απέχουν απόσταση 
    # μέχρι r τις προσθέτουμε στο σύνολο
    for d in distances:
        if(distances[d]<=r):
            set_of_nodes.add(d)
        # Ελέγχουμε τους γείτονες όσων έχουν απόσταση r 
        # και τους προσθέτουμε στο σύνολο
        if(distances[d]==r):
            for u in range(n):
                if(A[u,d] == 1):
                    set_of_nodes.add(u)
    return set_of_nodes, len(set_of_nodes)
    
# Συνάρτηση που επιστρέφει τις κορυφές του συνόλου Bin μίας κορυφής v
# μαζί με αυτές που γειτονεύουν μαζί τους
def find_Bin_plus1(A, n, v, r):
    # Υπολογισμός συντομότερων αποστάσεων από την κορυφή v
    set_of_nodes = set()
    for i in range(n):
        d = bfs1(A, n, i,v)
        if(d <= r):
            set_of_nodes.add(d)
        # προσθέτουμε και τους γείτονες αν η απόσταση είναι ίση με r
        if (d == r):
            for u in range(n):
                if (A[u, v] == 1):
                    set_of_nodes.add(u)
    return set_of_nodes, len(set_of_nodes)
    
    
    
    
    
    
    
    
    
    
