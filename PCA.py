import numpy as np

def PCA(X, eigvec_num):

    X = (X - X.mean(axis = 0)) / np.std(X, axis = 0) # standardizacija podataka, tako sto se oduzme prosek dimenzije od svakog podatka i devijacija za svaku dimenziju

    """ OVAJ PCA NE RADI NA KOMPLEKSNIM BROJEVIMA TREBALO BI X DA BUDE KONJUGOVANO KOMPLEKSNA MATRICA """
    # matrica kovarijanse, treba da se dobije kovarijansa nezavisnih u odnosu na zavisnu odnosno matrica koja ce biti (ako je originalna NxM) MxM
    C = ( X.T.dot(X) ) / X.shape[0]

    v, w = np.linalg.eig(C) # eigenvec/vals jer numpy opet ima f i za to

    # sortiranje eigvec i eigval
    idx = v.argsort()[::-1] # opadajuci
    v = v[idx] # eigval 
    w = w[:, idx] # eigvec

    """ pravljenje novog prostora """
    return X.dot(w[:, :eigvec_num])